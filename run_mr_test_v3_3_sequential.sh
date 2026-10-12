#!/usr/bin/env bash
set -uo pipefail

project_root="/home/ubuntu/MTrain"
python_bin="/home/ubuntu/anaconda3/bin/python"
generator="${project_root}/generate_mr_test_v3_3.py"
checkpoint_guard="${project_root}/verify_mr_test_checkpoint.py"
source_root="${project_root}/mr_test_sources_v3_3"
output_root="${project_root}/mr_test_data_v3_3_full"
run_root="${project_root}/mr_test_runs_v3_3_full"
queue_log="${run_root}/sequential_queue.log"
server_model="gemma-3-4b-it-qat"
# Main-model lane: mnlim ONLY. SICK runs independently on
# gemma-3-4b-it-qat-worker2. mnlimm and snli are report-only (both stopped on
# 2026-09-19) and must never be auto-restarted by this queue or by
# monitor_mr_test_v3_3_full.sh.
datasets=(mnlim)

mkdir -p "$output_root" "$run_root"

is_complete() {
    local report="$1"
    [[ -s "$report" ]] && grep -q '"status": "complete"' "$report"
}

pid_is_alive() {
    local pid_file="$1"
    local dataset="$2"
    local pid
    local index
    local -a argv=()
    [[ -s "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"
    kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    # Match the --dataset VALUE as an exact argv token. A substring test on the
    # joined command line is wrong here: "--dataset mnlim" is a prefix of
    # "--dataset mnlimm".
    mapfile -d '' -t argv < "/proc/${pid}/cmdline"
    for (( index = 0; index < ${#argv[@]}; index++ )); do
        if [[ "${argv[index]}" == "--dataset" && "${argv[index+1]:-}" == "$dataset" ]]; then
            [[ "${argv[*]}" == *"generate_mr_test_v3_3.py"* ]] && return 0
            return 1
        fi
    done
    return 1
}

timeout_for() {
    case "$1" in
        sick) printf '%s\n' 600 ;;
        mnlim|mnlimm) printf '%s\n' 120 ;;
        *) return 1 ;;
    esac
}

for dataset in "${datasets[@]}"; do
    input="${source_root}/${dataset}.jsonl"
    output="${output_root}/${dataset}.jsonl"
    generation_log="${output_root}/${dataset}.generation_log.jsonl"
    report="${output_root}/${dataset}.report.json"
    run_log="${run_root}/${dataset}.stdout.log"
    pid_file="${run_root}/${dataset}.pid"
    request_timeout="$(timeout_for "$dataset")"

    if [[ ! -s "$input" || ! -s "$output" || ! -s "$generation_log" ]]; then
        printf '[%s] unsafe or missing input/checkpoint pair for %s; stopping queue\n' \
            "$(date '+%F %T %z')" "$dataset" | tee -a "$queue_log" >&2
        exit 1
    fi

    while pid_is_alive "$pid_file" "$dataset"; do
        sleep 60
    done

    if is_complete "$report"; then
        printf '[%s] %s already complete; skipping\n' \
            "$(date '+%F %T %z')" "$dataset" | tee -a "$queue_log"
        continue
    fi

    attempt=0
    while ! is_complete "$report"; do
        attempt=$((attempt + 1))
        if (( attempt > 5 )); then
            printf '[%s] %s failed five resume attempts; stopping queue\n' \
                "$(date '+%F %T %z')" "$dataset" | tee -a "$queue_log" >&2
            exit 1
        fi

        if ! checkpoint_summary="$(
            "$python_bin" "$checkpoint_guard" --dataset "$dataset" --snapshot
        )"; then
            printf '[%s] %s checkpoint protection/validation failed; stopping queue\n' \
                "$(date '+%F %T %z')" "$dataset" | tee -a "$queue_log" >&2
            exit 1
        fi
        printf '[%s] %s checkpoint protected and valid: %s\n' \
            "$(date '+%F %T %z')" "$dataset" "$checkpoint_summary" >> "$queue_log"

        printf '[%s] launching %s attempt=%d model=%s workers=3 timeout=%s resume=true\n' \
            "$(date '+%F %T %z')" "$dataset" "$attempt" "$server_model" \
            "$request_timeout" | tee -a "$queue_log"

        "$python_bin" -u "$generator" \
            --dataset "$dataset" \
            --input "$input" \
            --output "$output" \
            --all \
            --seed 42 \
            --candidate-count 3 \
            --retries 2 \
            --workers 3 \
            --request-timeout "$request_timeout" \
            --checkpoint-every 10 \
            --server-model "$server_model" \
            --resume >> "$run_log" 2>&1 &
        child_pid=$!
        printf '%s\n' "$child_pid" > "$pid_file"

        wait "$child_pid"
        rc=$?
        if is_complete "$report"; then
            printf '[%s] %s complete\n' \
                "$(date '+%F %T %z')" "$dataset" | tee -a "$queue_log"
            break
        fi

        printf '[%s] %s exited rc=%d without complete report; retrying from checkpoint in 60s\n' \
            "$(date '+%F %T %z')" "$dataset" "$rc" | tee -a "$queue_log" >&2
        sleep 60
    done
done

printf '[%s] sequential queue complete\n' "$(date '+%F %T %z')" | tee -a "$queue_log"
