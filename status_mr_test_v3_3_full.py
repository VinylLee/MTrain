#!/usr/bin/env python3
"""Print resumable full-generation progress for all four MR test datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path("/home/ubuntu/MTrain")
OUTPUT_ROOT = ROOT / "mr_test_data_v3_3_full"
RUN_ROOT = ROOT / "mr_test_runs_v3_3_full"
SOURCE_ROOT = ROOT / "mr_test_sources_v3_3"
DATASETS = ("mnlim", "mnlimm", "sick", "snli")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def completed_pairs(path: Path) -> int:
    if not path.exists():
        return 0
    pair_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pair_ids.add(int(json.loads(line)["pair_id"]))
    return len(pair_ids)


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def report_status(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "pending"


def main() -> int:
    print("dataset  state     pairs/expected  progress   logs   output  report")
    for dataset in DATASETS:
        source = SOURCE_ROOT / f"{dataset}.jsonl"
        output = OUTPUT_ROOT / f"{dataset}.jsonl"
        log = OUTPUT_ROOT / f"{dataset}.generation_log.jsonl"
        report = OUTPUT_ROOT / f"{dataset}.report.json"
        pid = read_pid(RUN_ROOT / f"{dataset}.pid")
        expected = count_lines(source)
        done = completed_pairs(log)
        progress = 100.0 * done / expected if expected else 0.0
        status = report_status(report)
        state = "complete" if status == "complete" else ("running" if is_alive(pid) else "stopped")
        print(
            f"{dataset:<8} {state:<9} {done:>5}/{expected:<5} "
            f"{progress:>7.2f}% {count_lines(log):>7} {count_lines(output):>8} {status}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
