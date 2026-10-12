#!/usr/bin/env python3
"""Validate and optionally snapshot one resumable MR-test checkpoint pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ubuntu/MTrain")
OUTPUT_ROOT = ROOT / "mr_test_data_v3_3_full"
BACKUP_ROOT = ROOT / "mr_test_checkpoint_backups"
DATASETS = {"mnlim", "mnlimm", "sick", "snli"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(output: Path, generation_log: Path) -> dict[str, object]:
    output_rows = 0
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{output}:{line_number}: invalid JSON: {exc}") from exc
            output_rows += 1

    log_rows = 0
    pair_ids: set[int] = set()
    with generation_log.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pair_id = row["pair_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{generation_log}:{line_number}: invalid generation-log row: {exc}"
                ) from exc
            if not isinstance(pair_id, int) or pair_id < 0:
                raise ValueError(
                    f"{generation_log}:{line_number}: invalid pair_id={pair_id!r}"
                )
            pair_ids.add(pair_id)
            log_rows += 1

    completed_pairs = len(pair_ids)
    expected_pair_ids = set(range(completed_pairs))
    if pair_ids != expected_pair_ids:
        missing = sorted(expected_pair_ids - pair_ids)[:10]
        unexpected = sorted(pair_ids - expected_pair_ids)[:10]
        raise ValueError(
            "generation-log pair_ids are not contiguous from zero: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return {
        "output_rows": output_rows,
        "log_rows": log_rows,
        "completed_pairs": completed_pairs,
        "output_bytes": output.stat().st_size,
        "log_bytes": generation_log.stat().st_size,
        "output_sha256": sha256(output),
        "log_sha256": sha256(generation_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()

    output = OUTPUT_ROOT / f"{args.dataset}.jsonl"
    generation_log = OUTPUT_ROOT / f"{args.dataset}.generation_log.jsonl"
    if not output.is_file() or not generation_log.is_file():
        raise FileNotFoundError(
            f"missing checkpoint pair: output={output.exists()} log={generation_log.exists()}"
        )

    if not args.snapshot:
        summary = validate(output, generation_log)
        print(json.dumps({"dataset": args.dataset, "valid": True, **summary}, sort_keys=True))
        return 0

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    destination = BACKUP_ROOT / args.dataset / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    output_copy = destination / output.name
    log_copy = destination / generation_log.name
    shutil.copy2(output, output_copy)
    shutil.copy2(generation_log, log_copy)

    summary = validate(output_copy, log_copy)
    current_hashes = {"output_sha256": sha256(output), "log_sha256": sha256(generation_log)}
    if current_hashes["output_sha256"] != summary["output_sha256"] or current_hashes[
        "log_sha256"
    ] != summary["log_sha256"]:
        raise RuntimeError("checkpoint changed while snapshot was being copied")

    manifest = {
        "dataset": args.dataset,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_output": str(output),
        "source_generation_log": str(generation_log),
        "valid": True,
        **summary,
    }
    manifest_path = destination / "manifest.json"
    temporary_manifest = destination / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps({**manifest, "snapshot": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
