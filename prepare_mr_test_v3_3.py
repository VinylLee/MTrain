"""Snapshot legacy merged MR tests and extract their source examples.

The legacy files are copied byte-for-byte.  Source rows are selected only when
both provenance markers agree (``is_source is True`` and ``mr_id == 'none'``),
then normalized to the four fields consumed by the v3.3 test generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASETS = ("mnlim", "mnlimm", "sick", "snli")
REQUIRED_FIELDS = ("premise", "hypothesis", "label", "pair_id")
DEFAULT_LEGACY_ROOT = Path(
    "/home/ubuntu/LLMTrain/LLMTrain/data/nli/mr_test_data_merged"
)
DEFAULT_SNAPSHOT_ROOT = Path("mr_test_data_merged_legacy_20260918")
DEFAULT_SOURCE_ROOT = Path("mr_test_sources_v3_3")
DEFAULT_ORIGINAL_ROOT = Path("original_dataset")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def normalized_source(row: Dict[str, Any]) -> Dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"source row missing fields {missing}: {row}")
    label = int(row["label"])
    if label not in (0, 1, 2):
        raise ValueError(f"invalid source label {label}: {row}")
    return {
        "premise": str(row["premise"]),
        "hypothesis": str(row["hypothesis"]),
        "label": label,
        "pair_id": int(row["pair_id"]),
    }


def original_triple(row: Dict[str, Any]) -> tuple[str, str, int]:
    return str(row["premise"]), str(row["hypothesis"]), int(row["label"])


def prepare_dataset(
    dataset: str,
    legacy_root: Path,
    snapshot_root: Path,
    source_root: Path,
    original_root: Path,
) -> Dict[str, Any]:
    legacy_path = legacy_root / f"{dataset}.jsonl"
    snapshot_path = snapshot_root / f"{dataset}.jsonl"
    source_path = source_root / f"{dataset}.jsonl"
    original_path = original_root / dataset / "test.json"
    for path in (legacy_path, original_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    snapshot_root.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        if sha256_file(snapshot_path) != sha256_file(legacy_path):
            raise FileExistsError(
                f"snapshot exists with different content: {snapshot_path}"
            )
    else:
        shutil.copy2(legacy_path, snapshot_path)
    if sha256_file(snapshot_path) != sha256_file(legacy_path):
        raise RuntimeError(f"snapshot hash mismatch: {dataset}")

    rows = read_jsonl(snapshot_path)
    source_by_flag = [row for row in rows if row.get("is_source") is True]
    source_by_mr = [row for row in rows if row.get("mr_id") == "none"]
    if source_by_flag != source_by_mr:
        raise ValueError(f"source markers disagree for {dataset}")
    sources = [normalized_source(row) for row in source_by_flag]
    pair_ids = [row["pair_id"] for row in sources]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"duplicate source pair_id in {dataset}")

    original_rows = [
        row
        for row in read_jsonl(original_path)
        if row.get("label") != -1
        and all(field in row for field in ("premise", "hypothesis", "label"))
    ]
    extracted_triples = [original_triple(row) for row in sources]
    original_triples = [original_triple(row) for row in original_rows]
    if extracted_triples != original_triples:
        raise ValueError(f"extracted sources differ from original_dataset/{dataset}")
    expected_pair_ids = [int(row.get("idx", index)) for index, row in enumerate(original_rows)]
    if pair_ids != expected_pair_ids:
        raise ValueError(f"pair_id sequence differs from original dataset for {dataset}")

    write_jsonl(source_path, sources)
    return {
        "dataset": dataset,
        "legacy_path": str(legacy_path.resolve()),
        "snapshot_path": str(snapshot_path.resolve()),
        "source_path": str(source_path.resolve()),
        "legacy_sha256": sha256_file(legacy_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "source_sha256": sha256_file(source_path),
        "total_rows": len(rows),
        "source_rows": len(sources),
        "followup_rows": len(rows) - len(sources),
        "source_label_counts": dict(sorted(Counter(row["label"] for row in sources).items())),
        "unique_pair_ids": len(set(pair_ids)),
        "unique_source_triples": len(set(extracted_triples)),
        "matches_current_original_dataset": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [
        prepare_dataset(
            dataset,
            args.legacy_root,
            args.snapshot_root,
            args.source_root,
            args.original_root,
        )
        for dataset in DATASETS
    ]
    report = {
        "status": "complete",
        "source_selection": "is_source=true AND mr_id=none",
        "datasets": reports,
        "totals": {
            "rows": sum(item["total_rows"] for item in reports),
            "sources": sum(item["source_rows"] for item in reports),
            "followups": sum(item["followup_rows"] for item in reports),
        },
    }
    args.source_root.mkdir(parents=True, exist_ok=True)
    report_path = args.source_root / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
