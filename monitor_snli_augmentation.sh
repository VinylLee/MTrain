#!/usr/bin/env bash
set -u

pid="${1:?usage: $0 PID OUTPUT LOG}"
output="${2:?usage: $0 PID OUTPUT LOG}"
monitor_log="${3:?usage: $0 PID OUTPUT LOG}"
generation_log="${output%.json}.generation_log.jsonl"
target_rows=5340

snapshot() {
    local now output_rows generation_log_rows output_mtime process_state
    now="$(date '+%F %T %z')"
    output_rows=0
    generation_log_rows=0
    output_mtime="missing"
    process_state="stopped"

    if [[ -f "$output" ]]; then
        output_rows="$(wc -l < "$output")"
        output_mtime="$(stat -c '%y' "$output")"
    fi
    if [[ -f "$generation_log" ]]; then
        generation_log_rows="$(wc -l < "$generation_log")"
    fi
    if kill -0 "$pid" 2>/dev/null; then
        process_state="running"
    fi

    printf '%s pid=%s state=%s output_rows=%s generation_log_rows=%s output_mtime=%s\n' \
        "$now" "$pid" "$process_state" "$output_rows" \
        "$generation_log_rows" "$output_mtime" >> "$monitor_log"

    if (( output_rows >= target_rows )); then
        printf '%s status=target_reached target_rows=%s\n' "$now" "$target_rows" >> "$monitor_log"
        return 1
    fi
    if [[ "$process_state" == "stopped" ]]; then
        printf '%s status=generator_stopped_before_target target_rows=%s\n' \
            "$now" "$target_rows" >> "$monitor_log"
        return 1
    fi
    return 0
}

while snapshot; do
    sleep 3600
done
