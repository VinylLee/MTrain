"""Audit an SNLI augmentation JSONL file, optionally with an isolated Gemma review pass.

The report combines deterministic checks (schema, labels, duplicates, artifacts,
and length) with a stratified blind review by a separate local model.  Review
rows are written as JSONL so a person can override or inspect every judgment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI

from augment_snli_v2 import (
    JUDGE_SYSTEM_PROMPT,
    LABELS,
    _completion,
    extract_json,
    normalize_text,
    read_jsonl,
    write_jsonl,
)


DEFAULT_DATA_DIR = Path(
    "nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat"
)
ARTIFACT_PATTERN = re.compile(
    r"(?i)(<think>|```|\bpremise\s*:|\bhypothesis\s*:|target_label|source_label|"
    r"as an ai|i cannot|transformed sentence)"
)


def percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def deterministic_report(
    sources: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    source_map = {
        int(row.get("pair_id", index)): row for index, row in enumerate(sources)
    }
    synthetic = [row for row in rows if row.get("is_syn") is True]
    originals = [row for row in rows if row.get("is_syn") is not True]
    label_counts = Counter(int(row["label"]) for row in rows)
    mr_counts = Counter(str(row.get("mr_id", "missing")) for row in synthetic)
    mr_type_counts = Counter(str(row.get("mr_type", "missing")) for row in synthetic)
    duplicate_rows = 0
    duplicate_pairs = 0
    exact_seen = set()
    pair_seen = set()
    unchanged = 0
    missing_source = 0
    artifacts = 0
    word_lengths: List[int] = []

    for row in synthetic:
        exact_key = (
            normalize_text(str(row["premise"])),
            normalize_text(str(row["hypothesis"])),
            int(row["label"]),
            str(row.get("mr_id", "")),
        )
        pair_key = exact_key[:3]
        if exact_key in exact_seen:
            duplicate_rows += 1
        if pair_key in pair_seen:
            duplicate_pairs += 1
        exact_seen.add(exact_key)
        pair_seen.add(pair_key)
        pair_id = int(row.get("pair_id", -1))
        source = source_map.get(pair_id)
        if source is None:
            missing_source += 1
        elif normalize_text(str(row["hypothesis"])) == normalize_text(
            str(source["hypothesis"])
        ):
            unchanged += 1
        if ARTIFACT_PATTERN.search(str(row["hypothesis"])):
            artifacts += 1
        word_lengths.append(len(str(row["hypothesis"]).split()))

    return {
        "source_file_count": len(sources),
        "row_count": len(rows),
        "original_rows": len(originals),
        "synthetic_rows": len(synthetic),
        "label_counts": {LABELS[key]: value for key, value in sorted(label_counts.items())},
        "mr_counts": dict(sorted(mr_counts.items())),
        "mr_type_counts": dict(sorted(mr_type_counts.items())),
        "exact_duplicate_synthetic_rows": duplicate_rows,
        "duplicate_nli_triples": duplicate_pairs,
        "unchanged_from_source": unchanged,
        "missing_source_pair_id": missing_source,
        "prompt_artifacts": artifacts,
        "hypothesis_words": {
            "min": min(word_lengths) if word_lengths else 0,
            "median": percentile(word_lengths, 0.5),
            "p95": percentile(word_lengths, 0.95),
            "max": max(word_lengths) if word_lengths else 0,
        },
    }


def stratified_sample(
    rows: Sequence[Dict[str, Any]], sample_size: int, seed: int
) -> List[Dict[str, Any]]:
    synthetic = [row for row in rows if row.get("is_syn") is True]
    if sample_size >= len(synthetic):
        return synthetic
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in synthetic:
        buckets[str(row.get("mr_id", "missing"))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: List[Dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < sample_size and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < sample_size:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    rng.shuffle(selected)
    return selected


def build_review_prompt(items: Sequence[Dict[str, Any]]) -> str:
    payload = []
    for index, item in enumerate(items):
        payload.append(
            {
                "index": index,
                "premise": item["premise"],
                "source_hypothesis": item["source_hypothesis"],
                "source_label": LABELS[item["source_label"]],
                "candidate_hypothesis": item["hypothesis"],
                "declared_label": LABELS[item["label"]],
                "mr_id": item["mr_id"],
                "mr_type": item["mr_type"],
            }
        )
    return f"""Blindly review each SNLI augmentation record below:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For every item return:
- label: your strict SNLI label for premise -> candidate_hypothesis;
- grammatical: natural, self-contained English with no generation artifacts;
- source_equivalent: candidate and source hypothesis have exactly the same truth
  conditions (apply strictly for mr_type=inv; still report it for other types);
- operation_valid: inv preserves meaning, flip creates contradiction, and neutral
  adds only a plausible unsupported fact without negating the supported core;
- An object being mentioned never entails an unstated color, material, breed,
  accessory, or other attribute; an appended such attribute is normally neutral;
- reason: one short tag from ok, wrong_label, meaning_drift, grammar, artifact,
  not_changed, or other.

Return only this JSON shape with one entry per index:
{{"evaluations":[{{"index":0,"label":"entailment","grammatical":true,
"source_equivalent":true,"operation_valid":true,"reason":"ok"}}]}}"""


def parse_review(raw: str, expected_count: int) -> Dict[int, Dict[str, Any]]:
    value = extract_json(raw)
    if isinstance(value, dict) and isinstance(value.get("evaluations"), list):
        evaluations = value["evaluations"]
    elif isinstance(value, list):
        evaluations = []
        for group in value:
            if isinstance(group, dict) and isinstance(group.get("evaluations"), list):
                evaluations.extend(group["evaluations"])
            elif isinstance(group, dict):
                evaluations.append(group)
    else:
        raise ValueError("review response lacks evaluations")
    parsed = {}
    for item in evaluations:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            continue
        index = item["index"]
        if not 0 <= index < expected_count:
            continue
        label = str(item.get("label", "")).lower().strip()
        if label not in LABELS.values():
            continue
        parsed[index] = {
            "judged_label": label,
            "grammatical": item.get("grammatical") is True,
            "source_equivalent": item.get("source_equivalent") is True,
            "operation_valid": item.get("operation_valid") is True,
            "reason": str(item.get("reason", "other")),
        }
    if len(parsed) != expected_count:
        raise ValueError(f"review returned {len(parsed)}/{expected_count} items")
    return parsed


def judge_rows(
    sources: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    client: OpenAI,
    model: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    source_map = {
        int(row.get("pair_id", index)): row for index, row in enumerate(sources)
    }
    prepared = []
    for row in rows:
        pair_id = int(row.get("pair_id", -1))
        source = source_map.get(pair_id)
        if source is None:
            continue
        prepared.append(
            {
                "pair_id": pair_id,
                "premise": str(row["premise"]),
                "source_hypothesis": str(source["hypothesis"]),
                "source_label": int(source["label"]),
                "hypothesis": str(row["hypothesis"]),
                "label": int(row["label"]),
                "mr_id": str(row.get("mr_id", "missing")),
                "mr_type": str(row.get("mr_type", "missing")),
            }
        )

    reviewed: List[Dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        error = None
        parsed = None
        for _ in range(2):
            try:
                raw = _completion(
                    client,
                    model,
                    JUDGE_SYSTEM_PROMPT,
                    build_review_prompt(batch),
                    temperature=0.0,
                    max_tokens=max(600, 150 * len(batch)),
                )
                parsed = parse_review(raw, len(batch))
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if parsed is None:
            for item in batch:
                reviewed.append({**item, "review_error": error})
        else:
            for index, item in enumerate(batch):
                review = parsed[index]
                review["label_correct"] = review["judged_label"] == LABELS[item["label"]]
                review["fully_accepted"] = (
                    review["label_correct"]
                    and review["grammatical"]
                    and review["operation_valid"]
                    and (item["mr_type"] != "inv" or review["source_equivalent"])
                )
                reviewed.append({**item, **review})
        print(f"reviewed {min(start + batch_size, len(prepared))}/{len(prepared)}", flush=True)
    return reviewed


def rate(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    available = [row for row in rows if field in row]
    if not available:
        return None
    return round(sum(row[field] is True for row in available) / len(available), 4)


def review_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if "judged_label" in row]
    inv = [row for row in valid if row["mr_type"] == "inv"]
    by_mr = {}
    for mr_id in sorted({row["mr_id"] for row in valid}):
        subset = [row for row in valid if row["mr_id"] == mr_id]
        by_mr[mr_id] = {
            "count": len(subset),
            "label_accuracy": rate(subset, "label_correct"),
            "grammatical_rate": rate(subset, "grammatical"),
            "operation_valid_rate": rate(subset, "operation_valid"),
            "fully_accepted_rate": rate(subset, "fully_accepted"),
        }
    metrics = {
        "requested": len(rows),
        "successfully_judged": len(valid),
        "label_accuracy": rate(valid, "label_correct"),
        "grammatical_rate": rate(valid, "grammatical"),
        "operation_valid_rate": rate(valid, "operation_valid"),
        "inv_source_equivalence_rate": rate(inv, "source_equivalent"),
        "fully_accepted_rate": rate(valid, "fully_accepted"),
        "by_mr": by_mr,
    }
    metrics["quality_gates"] = {
        "label_accuracy_gte_0.95": (metrics["label_accuracy"] or 0) >= 0.95,
        "grammatical_rate_gte_0.95": (metrics["grammatical_rate"] or 0) >= 0.95,
        "inv_equivalence_gte_0.95": (
            metrics["inv_source_equivalence_rate"] is None
            or metrics["inv_source_equivalence_rate"] >= 0.95
        ),
        "fully_accepted_rate_gte_0.90": (metrics["fully_accepted_rate"] or 0) >= 0.90,
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA_DIR / "labeled_data.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR / "augmented_data.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--judge-sample", type=int, default=0)
    parser.add_argument("--judge-model", default="gemma-3-4b-it-qat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = read_jsonl(args.source)
    rows = read_jsonl(args.data)
    report = {"data": str(args.data), "deterministic": deterministic_report(sources, rows)}
    if args.judge_sample:
        sample = stratified_sample(rows, args.judge_sample, args.seed)
        client = OpenAI(
            api_key="local",
            base_url=f"http://{args.host}:{args.port}/v1",
            timeout=180.0,
        )
        reviewed = judge_rows(sources, sample, client, args.judge_model, args.batch_size)
        review_path = args.review_output or args.data.with_suffix(".quality_review.jsonl")
        write_jsonl(review_path, reviewed)
        report["judge_model"] = args.judge_model
        report["review_output"] = str(review_path)
        report["llm_review"] = review_report(reviewed)
    report_path = args.report or args.data.with_suffix(".quality_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
