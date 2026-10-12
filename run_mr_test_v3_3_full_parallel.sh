#!/usr/bin/env bash
set -euo pipefail

project_root="/home/ubuntu/MTrain"
python_bin="/home/ubuntu/anaconda3/bin/python"
output_root="${project_root}/mr_test_data_v3_3_full"
run_root="${project_root}/mr_test_runs_v3_3_full"
generator="${project_root}/generate_mr_test_v3_3.py"

mkdir -p "$output_root" "$run_root"

datasets=(mnlim mnlimm sick snli)

server_model_for() {
    case "$1" in
        mnlim|mnlimm|snli) printf '%s\n' "gemma-3-4b-it-qat" ;;
        sick) printf '%s\n' "gemma-3-4b-it-qat-worker2" ;;
        *) return 1 ;;
    esac
}

workers_for() {
    case "$1" in
        mnlim|mnlimm) printf '%s\n' 3 ;;
        sick) printf '%s\n' 4 ;;
        snli) printf '%s\n' 2 ;;
        *) return 1 ;;
    esac
}

request_timeout_for() {
    case "$1" in
        mnlim|mnlimm) printf '%s\n' 120 ;;
        sick|snli) printf '%s\n' 600 ;;
        *) return 1 ;;
    esac
}

for dataset in "${datasets[@]}"; do
    input="${project_root}/mr_test_sources_v3_3/${dataset}.jsonl"
    output="${output_root}/${dataset}.jsonl"
    generation_log="${output_root}/${dataset}.generation_log.jsonl"
    report="${output_root}/${dataset}.report.json"
    run_log="${run_root}/${dataset}.stdout.log"
    pid_file="${run_root}/${dataset}.pid"
    server_model="$(server_model_for "$dataset")"
    workers="$(workers_for "$dataset")"
    request_timeout="$(request_timeout_for "$dataset")"
    tmux_session="MTrain-MR-v33-${dataset}"

    if [[ ! -s "$input" ]]; then
        printf 'missing input for %s: %s\n' "$dataset" "$input" >&2
        exit 1
    fi

    if [[ -s "$report" ]] && grep -q '"status": "complete"' "$report"; then
        printf '%s already complete; skipping\n' "$dataset"
        continue
    fi

    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        printf '%s already running in tmux session %s; skipping\n' "$dataset" "$tmux_session"
        continue
    fi

    resume_args=()
    if [[ -s "$output" && -s "$generation_log" ]]; then
        resume_args=(--resume)
    elif [[ -e "$output" || -e "$generation_log" ]]; then
        printf '%s has an incomplete output/log pair; refusing unsafe restart\n' "$dataset" >&2
        exit 1
    fi

    {
        printf '\n[%s] launch dataset=%s server_model=%s workers=%s timeout=%s mode=%s\n' \
            "$(date '+%F %T %z')" "$dataset" "$server_model" "$workers" \
            "$request_timeout" \
            "$([[ ${#resume_args[@]} -gt 0 ]] && printf resume || printf fresh)"
    } >> "$run_log"

    command=("$python_bin" -u "$generator" \
        --dataset "$dataset" \
        --input "$input" \
        --output "$output" \
        --all \
        --seed 42 \
        --candidate-count 3 \
        --retries 2 \
        --workers "$workers" \
        --request-timeout "$request_timeout" \
        --checkpoint-every 10 \
        --server-model "$server_model" \
        "${resume_args[@]}")
    printf -v command_string '%q ' "${command[@]}"
    printf -v run_log_quoted '%q' "$run_log"
    tmux new-session -d -s "$tmux_session" \
        "exec ${command_string} >> ${run_log_quoted} 2>&1"
    pid="$(tmux display-message -p -t "$tmux_session" '#{pane_pid}')"
    printf '%s\n' "$pid" > "$pid_file"
    printf '%s started pid=%s model=%s workers=%s tmux=%s\n' \
        "$dataset" "$pid" "$server_model" "$workers" "$tmux_session"
done

monitor_session="MTrain-MR-v33-monitor"
if ! tmux has-session -t "$monitor_session" 2>/dev/null; then
    tmux new-session -d -s "$monitor_session" \
        "exec ${project_root}/monitor_mr_test_v3_3_full.sh"
    monitor_pid="$(tmux display-message -p -t "$monitor_session" '#{pane_pid}')"
    printf '%s\n' "$monitor_pid" > "${run_root}/monitor.pid"
    printf 'monitor started pid=%s tmux=%s\n' "$monitor_pid" "$monitor_session"
fi
