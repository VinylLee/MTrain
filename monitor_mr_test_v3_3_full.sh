#!/usr/bin/env bash
set -uo pipefail

project_root="/home/ubuntu/MTrain"
python_bin="/home/ubuntu/anaconda3/bin/python"
run_root="${project_root}/mr_test_runs_v3_3_full"
output_root="${project_root}/mr_test_data_v3_3_full"
status_script="${project_root}/status_mr_test_v3_3_full.py"
checkpoint_guard="${project_root}/verify_mr_test_checkpoint.py"
queue_script="${project_root}/run_mr_test_v3_3_sequential.sh"
history="${run_root}/status_history.log"
supervisor_log="${run_root}/supervisor.log"
state_file="${run_root}/supervisor_state.tsv"
stall_state_file="${run_root}/stall_state.tsv"
one_shot_lock="${run_root}/one_shot_monitor.lock"
model_path="lmstudio-community/gemma-3-4B-it-qat-GGUF/gemma-3-4B-it-QAT-Q4_0.gguf"

# Auto-recovery scope is mnlim (see start_main_queue_if_needed) plus sick (see
# start_sick_if_needed) ONLY. mnlimm and snli are report-only: both were stopped
# on 2026-09-19 and must never be auto-restarted by this supervisor. They are
# still reported by record_progress and status_mr_test_v3_3_full.py.

# Record-only thresholds. These annotate the log; they never trigger recovery.
gpu_temp_warn_c=78
gpu_temp_alert_c=85
gpu_free_warn_mib=256
gpu_free_alert_mib=96

mkdir -p "$run_root"

# Both the tmux loop (run_mr_test_v3_3_monitor_loop.sh) and the systemd timer
# mtrain-mr-v33-monitor.timer (OnCalendar=*:00,30) invoke this one-shot. Without
# this lock the two can run concurrently and both snapshot + relaunch, which is
# what produces "checkpoint changed while snapshot was being copied" and double
# relaunches. Non-blocking on purpose: a hung holder must never wedge recovery
# permanently (the loop wraps its call in `timeout 900` as a backstop).
exec 8> "$one_shot_lock"
if ! flock -n 8; then
    printf '[%s] another one-shot supervisor check is in progress; skipping this invocation\n' \
        "$(date '+%F %T %z')" >> "$supervisor_log"
    exit 0
fi

log() {
    printf '[%s] %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$supervisor_log"
}

report_complete() {
    local dataset="$1"
    "$python_bin" - "$output_root/${dataset}.report.json" <<'PY' 2>/dev/null
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and json.loads(path.read_text()).get("status") == "complete" else 1)
PY
}

pid_is_dataset_generator() {
    local dataset="$1"
    local pid_file="${run_root}/${dataset}.pid"
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

model_loaded() {
    local identifier="$1"
    timeout 30 lms ps --json 2>/dev/null | "$python_bin" -c '
import json, sys
identifier = sys.argv[1]
try:
    models = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if any(model.get("identifier") == identifier for model in models) else 1)
' "$identifier"
}

ensure_model() {
    local identifier="$1"
    if model_loaded "$identifier"; then
        return 0
    fi
    log "model instance missing: ${identifier}; attempting safe reload"
    if timeout 180 lms load "$model_path" --exact --yes --gpu max \
        --context-length 4096 --identifier "$identifier" >> "$supervisor_log" 2>&1; then
        log "model instance reloaded: ${identifier}"
        return 0
    fi
    log "ERROR model reload failed: ${identifier}"
    return 1
}

start_main_queue_if_needed() {
    # Auto-recovery scope is mnlim ONLY. mnlimm is report-only and is never
    # auto-restarted, so it is deliberately absent from the target selection.
    local target="mnlim"
    local checkpoint_summary

    if report_complete "$target"; then
        if ! pid_is_dataset_generator "$target" &&
            tmux has-session -t MTrain-MR-v33-sequential 2>/dev/null; then
            # The running sequential queue holds the previous `datasets=(mnlim
            # mnlimm)` array in memory and would launch the report-only mnlimm
            # lane next. Record it; do not act (record-only by decision).
            log "ALERT report_only_launch_pending dataset=mnlimm detail=sequential queue still alive after mnlim completed and will attempt the report-only mnlimm lane; manual decision required"
        fi
        log "main lane complete: dataset=${target}; auto-recovery idle"
        return 0
    fi

    if pid_is_dataset_generator "$target"; then
        ensure_model gemma-3-4b-it-qat || return 1
        log "main lane healthy: dataset=${target} process=running queue=$(tmux has-session -t MTrain-MR-v33-sequential 2>/dev/null && printf running || printf stopped)"
        return 0
    fi
    if tmux has-session -t MTrain-MR-v33-sequential 2>/dev/null; then
        ensure_model gemma-3-4b-it-qat || return 1
        log "main lane generator not active; sequential queue remains responsible for guarded recovery"
        return 0
    fi

    log "main lane stopped: dataset=${target}; protecting and validating checkpoint"
    if ! checkpoint_summary="$(
        "$python_bin" "$checkpoint_guard" --dataset "$target" --snapshot 2>> "$supervisor_log"
    )"; then
        log "ERROR ${target} checkpoint protection/validation failed; automatic resume blocked"
        return 1
    fi
    log "${target} checkpoint protected and valid: ${checkpoint_summary}"
    ensure_model gemma-3-4b-it-qat || return 1
    log "main sequential queue stopped; restarting guarded queue"
    if ! bash -n "$queue_script" >> "$supervisor_log" 2>&1; then
        log "ERROR main queue script validation failed"
        return 1
    fi
    tmux new-session -d -s MTrain-MR-v33-sequential "exec ${queue_script}"
    log "main sequential queue restarted"
}

start_sick_if_needed() {
    local dataset="sick"
    local checkpoint_summary
    local pid
    local command
    local run_log="${run_root}/sick.stdout.log"

    if report_complete "$dataset"; then
        log "secondary lane complete: dataset=sick"
        return 0
    fi
    if pid_is_dataset_generator "$dataset"; then
        ensure_model gemma-3-4b-it-qat-worker2 || return 1
        log "secondary lane healthy: dataset=sick process=running"
        return 0
    fi

    log "secondary lane stopped: dataset=sick; protecting and validating checkpoint"
    if ! checkpoint_summary="$(
        "$python_bin" "$checkpoint_guard" --dataset sick --snapshot 2>> "$supervisor_log"
    )"; then
        log "ERROR SICK checkpoint protection/validation failed; automatic resume blocked"
        return 1
    fi
    log "SICK checkpoint protected and valid: ${checkpoint_summary}"
    ensure_model gemma-3-4b-it-qat-worker2 || return 1

    if tmux has-session -t MTrain-MR-v33-sick 2>/dev/null; then
        tmux kill-session -t MTrain-MR-v33-sick
    fi
    command=("$python_bin" -u "${project_root}/generate_mr_test_v3_3.py"
        --dataset sick
        --input "${project_root}/mr_test_sources_v3_3/sick.jsonl"
        --output "${output_root}/sick.jsonl"
        --all
        --seed 42
        --candidate-count 3
        --retries 2
        --workers 4
        --request-timeout 600
        --checkpoint-every 10
        --server-model gemma-3-4b-it-qat-worker2
        --resume)
    printf -v command_string '%q ' "${command[@]}"
    printf -v run_log_quoted '%q' "$run_log"
    tmux new-session -d -s MTrain-MR-v33-sick \
        "exec ${command_string} >> ${run_log_quoted} 2>&1"
    pid="$(tmux display-message -p -t MTrain-MR-v33-sick '#{pane_pid}')"
    printf '%s\n' "$pid" > "${run_root}/sick.pid"
    log "secondary lane resumed: dataset=sick pid=${pid} workers=4 timeout=600"
}

record_progress() {
    local temporary_state="${state_file}.tmp"
    local temporary_stall_state="${stall_state_file}.tmp"
    local dataset
    local summary
    local values
    local completed
    local output_bytes
    local log_bytes
    local error_count
    local previous
    local previous_completed=0
    local previous_output_bytes=0
    local previous_log_bytes=0
    local previous_error_count=0
    local latest
    local last_error
    local new_errors
    local latest_pair=-1
    local lane_alive=0
    local current_pid=""
    local current_epoch
    local stall_row
    local previous_pair=-1
    local previous_streak=0
    local previous_change_epoch=0
    local previous_pid=""
    local streak
    local change_epoch
    local stalled_minutes
    local stall_warn_minutes
    local stall_alert_minutes
    local -a log_details

    : > "$temporary_state"
    : > "$temporary_stall_state"
    for dataset in mnlim mnlimm sick; do
        previous_completed=0
        previous_output_bytes=0
        previous_log_bytes=0
        previous_error_count=0
        latest_pair=-1
        log_details=()
        last_error="none"
        if ! summary="$("$python_bin" "$checkpoint_guard" --dataset "$dataset" 2>> "$supervisor_log")"; then
            log "ERROR periodic checkpoint validation failed: dataset=${dataset}"
            continue
        fi
        values="$(printf '%s' "$summary" | "$python_bin" -c '
import json, sys
row = json.load(sys.stdin)
print(row["completed_pairs"], row["output_bytes"], row["log_bytes"])
')"
        read -r completed output_bytes log_bytes <<< "$values"
        mapfile -t log_details < <(
            "$python_bin" - "${run_root}/${dataset}.stdout.log" "$dataset" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
dataset = sys.argv[2]
errors = 0
latest = "none"
last_error = "none"
latest_pair = -1
if path.is_file():
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("RuntimeError:"):
                errors += 1
                last_error = line.rstrip()
            if line.startswith(f"[{dataset} ") and "] pair_id=" in line:
                latest = line.rstrip()
                try:
                    latest_pair = int(latest.split("] pair_id=", 1)[1].split(None, 1)[0])
                except (IndexError, ValueError):
                    pass
print(errors)
print(latest)
print(last_error)
print(latest_pair)
PY
        )
        error_count="${log_details[0]:-0}"
        latest="${log_details[1]:-none}"
        last_error="${log_details[2]:-none}"
        latest_pair="${log_details[3]:--1}"

        if [[ -s "$state_file" ]]; then
            previous="$(awk -F '\t' -v dataset="$dataset" '$1 == dataset {print $2, $3, $4, $5}' "$state_file")"
            if [[ -n "$previous" ]]; then
                read -r previous_completed previous_output_bytes previous_log_bytes previous_error_count <<< "$previous"
            fi
        fi

        new_errors=$((error_count - previous_error_count))
        if (( new_errors < 0 )); then
            new_errors="$error_count"
        fi
        log "progress dataset=${dataset} checkpoint_pairs=${completed} delta_pairs=$((completed - previous_completed)) ckpt_lag=$((latest_pair + 1 - completed)) output_growth=$((output_bytes - previous_output_bytes)) log_growth=$((log_bytes - previous_log_bytes)) errors=${error_count} new_errors=${new_errors} latest=${latest:-none}"
        if (( new_errors > 0 )); then
            log "error dataset=${dataset} last=${last_error}"
        fi

        # Stall detection is keyed on the `latest=` pair_id, never on
        # delta_pairs: the generator flushes a checkpoint only every
        # --checkpoint-every sources, so delta_pairs reads 0 for whole cycles
        # while the lane is in fact progressing. The predicate is "changed",
        # not "greater", because a resume re-prints a progress line for every
        # already-completed source and rewinds the counter to zero.
        case "$dataset" in
            # Worst case for one source pair is
            # ceil(11 operations / workers) * retries(2) * 2 calls * request-timeout
            # = 4*4*120s = 32 min for mnlim (w=3,t=120); 3*4*600s = 120 min for
            # sick (w=4,t=600). WARN is 2x that, ALERT 3x.
            mnlim) stall_warn_minutes=64; stall_alert_minutes=96 ;;
            sick) stall_warn_minutes=240; stall_alert_minutes=360 ;;
            *) stall_warn_minutes=64; stall_alert_minutes=96 ;;
        esac
        if [[ "$dataset" == mnlim || "$dataset" == sick ]]; then
            lane_alive=0
            if pid_is_dataset_generator "$dataset"; then
                lane_alive=1
                current_pid="$(<"${run_root}/${dataset}.pid")"
            fi
            current_epoch="$(date +%s)"
            stall_row=""
            if [[ -s "$stall_state_file" ]]; then
                stall_row="$(awk -F '\t' -v d="$dataset" '$1 == d {print $2"\t"$3"\t"$4"\t"$5}' "$stall_state_file")"
            fi
            previous_pair=-1
            previous_streak=0
            previous_change_epoch=0
            previous_pid=""
            if [[ -n "$stall_row" ]]; then
                IFS=$'\t' read -r previous_pair previous_streak previous_change_epoch previous_pid <<< "$stall_row"
            fi
            streak="$previous_streak"
            change_epoch="$previous_change_epoch"
            if [[ "$current_pid" != "$previous_pid" ]] || (( latest_pair != previous_pair )); then
                streak=0
                change_epoch="$current_epoch"
            elif (( lane_alive == 1 && latest_pair >= 0 )); then
                streak=$((previous_streak + 1))
            fi
            stalled_minutes=$(( (current_epoch - change_epoch) / 60 ))
            if (( lane_alive == 0 )); then
                log "ALERT lane_down dataset=${dataset} latest_pair=${latest_pair} stalled_minutes=${stalled_minutes}"
            elif (( stalled_minutes >= stall_alert_minutes )); then
                log "ALERT stall dataset=${dataset} latest_pair=${latest_pair} stalled_minutes=${stalled_minutes} threshold=${stall_alert_minutes} no_progress_cycles=${streak}"
            elif (( stalled_minutes >= stall_warn_minutes )); then
                log "WARN stall dataset=${dataset} latest_pair=${latest_pair} stalled_minutes=${stalled_minutes} threshold=${stall_warn_minutes} no_progress_cycles=${streak}"
            else
                log "stall dataset=${dataset} state=ok latest_pair=${latest_pair} stalled_minutes=${stalled_minutes}"
            fi
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$dataset" "$latest_pair" "$streak" "$change_epoch" "$current_pid" >> "$temporary_stall_state"
        fi

        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$dataset" "$completed" "$output_bytes" "$log_bytes" "$error_count" >> "$temporary_state"
    done
    mv "$temporary_state" "$state_file"
    mv "$temporary_stall_state" "$stall_state_file"
}

log "one-shot supervisor check begin"
start_main_queue_if_needed
start_sick_if_needed

{
    printf '\n[%s]\n' "$(date '+%F %T %z')"
    "$python_bin" "$status_script"
} >> "$history" 2>&1
"$python_bin" "$status_script" 2>&1 | while IFS= read -r line; do
    log "status ${line}"
done

record_progress

models="$(timeout 30 lms ps --json 2>&1 || true)"
log "models ${models}"

# Record-only GPU thresholds. A here-string is used instead of a pipeline so the
# loop body runs in this shell rather than a subshell.
if ! gpu_output="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>&1)"; then
    log "WARN gpu_query_failed output=$(printf '%s' "$gpu_output" | tr '\n' ';')"
else
    while IFS= read -r gpu_line; do
        [[ -n "$gpu_line" ]] || continue
        log "gpu ${gpu_line}"
        IFS=',' read -r gpu_index gpu_name gpu_used gpu_total gpu_free gpu_util gpu_temp <<< "$gpu_line"
        gpu_index="${gpu_index// /}"
        gpu_free="${gpu_free// /}"
        gpu_temp="${gpu_temp// /}"
        if [[ "$gpu_temp" =~ ^[0-9]+$ ]]; then
            if (( gpu_temp >= gpu_temp_alert_c )); then
                log "ALERT gpu_temp index=${gpu_index} temp_c=${gpu_temp} threshold=${gpu_temp_alert_c}"
            elif (( gpu_temp >= gpu_temp_warn_c )); then
                log "WARN gpu_temp index=${gpu_index} temp_c=${gpu_temp} threshold=${gpu_temp_warn_c}"
            fi
        fi
        if [[ "$gpu_free" =~ ^[0-9]+$ ]]; then
            if (( gpu_free <= gpu_free_alert_mib )); then
                log "ALERT gpu_free index=${gpu_index} free_mib=${gpu_free} threshold=${gpu_free_alert_mib} note=shared_tenant"
            elif (( gpu_free <= gpu_free_warn_mib )); then
                log "WARN gpu_free index=${gpu_index} free_mib=${gpu_free} threshold=${gpu_free_warn_mib} note=shared_tenant"
            fi
        fi
    done <<< "$gpu_output"
fi

if report_complete mnlim && report_complete mnlimm && report_complete sick; then
    log "all local datasets complete"
fi
log "one-shot supervisor check complete"
