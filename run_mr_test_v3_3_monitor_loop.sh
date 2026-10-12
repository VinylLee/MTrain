#!/usr/bin/env bash
set -uo pipefail

project_root="/home/ubuntu/MTrain"
monitor="${project_root}/monitor_mr_test_v3_3_full.sh"
run_root="${project_root}/mr_test_runs_v3_3_full"
loop_log="${run_root}/supervisor.log"
lock_file="${run_root}/supervisor.lock"
interval_seconds=1800
# Bound each one-shot so a wedged probe cannot hold the one-shot lock
# (mr_test_runs_v3_3_full/one_shot_monitor.lock) forever and starve recovery.
monitor_timeout_seconds=900

mkdir -p "$run_root"
exec 9> "$lock_file"
if ! flock -n 9; then
    printf '[%s] another supervisor loop already holds %s\n' \
        "$(date '+%F %T %z')" "$lock_file" >> "$loop_log"
    exit 1
fi

while true; do
    printf '[%s] supervisor cycle begin interval_seconds=%s\n' \
        "$(date '+%F %T %z')" "$interval_seconds" >> "$loop_log"
    timeout "$monitor_timeout_seconds" "$monitor"
    rc=$?
    if (( rc != 0 )); then
        printf '[%s] ERROR supervisor one-shot exited rc=%s; next check remains scheduled\n' \
            "$(date '+%F %T %z')" "$rc" >> "$loop_log"
    fi
    printf '[%s] supervisor cycle complete; sleeping %ss\n' \
        "$(date '+%F %T %z')" "$interval_seconds" >> "$loop_log"
    sleep "$interval_seconds"
done
