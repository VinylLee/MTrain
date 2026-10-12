"""Generate merged MR test JSONL with the current v3.3 augmentation contract.

This is a schema and orchestration adapter around
``augment_snli_all_label_mrs.py``.  It does not redefine prompts, operation
specifications, eligibility rules, deterministic filters, or composite plans.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

import augment_snli_all_label_mrs as core
from provenance_schema import COMPONENT_MRS_FIELD, sha256_file


def source_record(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "idx": -1,
        "premise": str(source["premise"]),
        "hypothesis": str(source["hypothesis"]),
        "mr_id": "none",
        "pair_id": int(source["pair_id"]),
        "label": int(source["label"]),
        "mr_type": "inv",
        "is_source": True,
    }


def followup_record(
    source: Mapping[str, Any],
    operation: str,
    hypothesis: str,
    component_mrs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "idx": -1,
        "premise": str(source["premise"]),
        "hypothesis": hypothesis,
        "mr_id": operation,
        "pair_id": int(source["pair_id"]),
        "label": core.LABEL_IDS[core.target_label(dict(source), operation)],
    }
    if component_mrs is not None:
        row[COMPONENT_MRS_FIELD] = list(component_mrs)
    row["mr_type"] = core.OPERATION_SPECS[operation]["mr_type"]
    row["is_source"] = False
    return row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_sources(rows: Sequence[Dict[str, Any]]) -> None:
    pair_ids = []
    for position, row in enumerate(rows, 1):
        missing = [key for key in ("premise", "hypothesis", "label", "pair_id") if key not in row]
        if missing:
            raise ValueError(f"source row {position} missing {missing}")
        if int(row["label"]) not in core.LABELS:
            raise ValueError(f"source row {position} has invalid label {row['label']!r}")
        pair_ids.append(int(row["pair_id"]))
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("source pair_id values must be unique")


def accepted_log_hypothesis(log: Mapping[str, Any]) -> Optional[str]:
    value = log.get("selected_hypothesis")
    return str(value) if isinstance(value, str) and value.strip() else None


def collect_generation_errors(value: Any) -> List[str]:
    """Collect nested request errors from atomic and composite generation logs."""
    errors: List[str] = []
    if isinstance(value, Mapping):
        error = value.get("error")
        if isinstance(error, str) and error:
            errors.append(error)
        for child in value.values():
            errors.extend(collect_generation_errors(child))
    elif isinstance(value, list):
        for child in value:
            errors.extend(collect_generation_errors(child))
    return errors


# A single source pair the model cannot answer must not discard the rest of the
# run: the pair is recorded (with its request error) and skipped. A *streak* of
# such pairs, however, means the model or the server is broken rather than one
# pathological source, so the run still fails fast once the streak reaches this.
MAX_CONSECUTIVE_REQUEST_ERROR_SOURCES = 5


def should_abort_on_request_errors(consecutive_sources: int) -> bool:
    """True when a run of failing source pairs indicates a systemic failure."""
    return consecutive_sources >= MAX_CONSECUTIVE_REQUEST_ERROR_SOURCES


def verify_service_model(client: OpenAI, model: str) -> None:
    """Require one successful inference; listing a model does not mean it is loaded."""
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with OK only."}],
        temperature=0.0,
        max_tokens=8,
    )


def materialize_rows(
    selected: Sequence[Dict[str, Any]],
    accepted: Mapping[Tuple[int, str], Dict[str, Any]],
    operations: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in sorted(selected, key=lambda row: int(row["pair_id"])):
        pair_id = int(source["pair_id"])
        rows.append(source_record(source))
        for operation in operations:
            row = accepted.get((pair_id, operation))
            if row is not None:
                rows.append(dict(row))
    for idx, row in enumerate(rows):
        row["idx"] = idx
    return rows


def recover_state(
    output_rows: Sequence[Dict[str, Any]],
    logs: Sequence[Dict[str, Any]],
    selected_by_pair: Mapping[int, Dict[str, Any]],
    operations: Sequence[str],
) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], set[Tuple[int, str]]]:
    operation_set = set(operations)
    completed: set[Tuple[int, str]] = set()
    accepted: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for log in logs:
        pair_id = int(log["pair_id"])
        operation = str(log["operation"])
        key = (pair_id, operation)
        if pair_id not in selected_by_pair or operation not in operation_set:
            raise ValueError(f"resume log contains unexpected key {key}")
        if key in completed:
            raise ValueError(f"resume log contains duplicate key {key}")
        completed.add(key)
        hypothesis = accepted_log_hypothesis(log)
        if log.get("status") == "accepted":
            if hypothesis is None:
                raise ValueError(f"accepted log lacks selected_hypothesis for {key}")
            components = core.component_mrs_from_log(operation, dict(log))
            accepted[key] = followup_record(
                selected_by_pair[pair_id], operation, hypothesis, components
            )

    expected_rows = materialize_rows(list(selected_by_pair.values()), accepted, operations)
    if output_rows:
        expected_without_idx = [{k: v for k, v in row.items() if k != "idx"} for row in expected_rows]
        actual_without_idx = [{k: v for k, v in row.items() if k != "idx"} for row in output_rows]
        if actual_without_idx != expected_without_idx:
            # Logs are checkpointed before output, so only a strict prefix/missing accepted
            # row is recoverable. Any extra or changed output is corruption.
            if len(actual_without_idx) > len(expected_without_idx):
                raise ValueError("resume output has rows not represented by the generation log")
            actual_keys = {
                (int(row["pair_id"]), str(row["mr_id"]), str(row["hypothesis"]))
                for row in actual_without_idx
            }
            expected_keys = {
                (int(row["pair_id"]), str(row["mr_id"]), str(row["hypothesis"]))
                for row in expected_without_idx
            }
            if not actual_keys.issubset(expected_keys):
                raise ValueError("resume output disagrees with the generation log")
    return accepted, completed


def summarize(
    selected: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    logs: Sequence[Dict[str, Any]],
    operations: Sequence[str],
) -> Dict[str, Any]:
    followups = [row for row in rows if not row["is_source"]]
    attempts = Counter(str(log["operation"]) for log in logs)
    accepted = Counter(
        str(log["operation"]) for log in logs if log.get("status") == "accepted"
    )
    return {
        "prompt_version": core.PROMPT_VERSION,
        "provenance_schema_version": core.PROVENANCE_SCHEMA_VERSION,
        "config_fingerprint": core.config_fingerprint(
            core.PROMPT_VERSION, core.OPERATION_SPECS, core.COMPOSITE_STAGE_PLANS
        ),
        "source_count": len(selected),
        "followup_count": len(followups),
        "total_rows": len(rows),
        "source_label_counts": dict(sorted(Counter(int(r["label"]) for r in selected).items())),
        "followup_label_counts": dict(sorted(Counter(int(r["label"]) for r in followups).items())),
        "operation_results": {
            operation: {
                "attempted": attempts[operation],
                "accepted": accepted[operation],
                "acceptance_rate": (
                    round(accepted[operation] / attempts[operation], 4)
                    if attempts[operation]
                    else None
                ),
            }
            for operation in operations
        },
        "component_mrs_rows": sum(COMPONENT_MRS_FIELD in row for row in followups),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-model", default=core.ALLOWED_MODEL)
    parser.add_argument("--judge-model", default=core.ALLOWED_MODEL)
    parser.add_argument(
        "--server-model",
        default=core.ALLOWED_MODEL,
        help="LM Studio identifier for the same model artifact (may be a CPU alias)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--operations", default=",".join(core.OPERATION_SPECS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.generator_model != core.ALLOWED_MODEL or args.judge_model != core.ALLOWED_MODEL:
        raise ValueError(f"generator and judge must both be {core.ALLOWED_MODEL}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"{args.output} exists; use --overwrite or --resume")
    if min(
        args.sample_size,
        args.candidate_count,
        args.retries,
        args.workers,
        args.checkpoint_every,
        args.request_timeout,
    ) < 1:
        raise ValueError("sample and generation controls must be positive")
    operations = [item.strip() for item in args.operations.split(",") if item.strip()]
    unknown = set(operations) - set(core.OPERATION_SPECS)
    if unknown:
        raise ValueError(f"unknown operations: {sorted(unknown)}")

    source_rows = read_jsonl(args.input)
    validate_sources(source_rows)
    selected = list(source_rows) if args.all else core.stratified_sample(source_rows, args.sample_size, args.seed)
    selected_by_pair = {int(row["pair_id"]): row for row in selected}
    log_path = args.output.with_suffix(".generation_log.jsonl")
    if args.resume:
        if not log_path.exists():
            raise FileNotFoundError("resume requires the generation log")
        logs = read_jsonl(log_path)
        output_rows = read_jsonl(args.output)
    else:
        logs = []
        output_rows = []
    accepted, completed = recover_state(
        output_rows, logs, selected_by_pair, operations
    )
    accepted_normalized_keys = {
        (
            core.normalize_text(str(row["premise"])),
            core.normalize_text(str(row["hypothesis"])),
        )
        for row in accepted.values()
    }

    base_url = f"http://{args.host}:{args.port}/v1"
    generator_client = OpenAI(
        api_key="local", base_url=base_url, timeout=args.request_timeout
    )
    judge_client = OpenAI(
        api_key="local", base_url=base_url, timeout=args.request_timeout
    )
    verify_service_model(generator_client, args.server_model)
    started = time.time()
    executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    def checkpoint() -> None:
        rows = materialize_rows(selected, accepted, operations)
        # Log first: recover_state can reconstruct a row if interrupted before output.
        write_jsonl(log_path, logs)
        write_jsonl(args.output, rows)

    processed_sources = 0
    consecutive_request_error_sources = 0
    try:
        for source_index, source in enumerate(selected, 1):
            pair_id = int(source["pair_id"])
            eligible_operations = [op for op in operations if core.eligible(source, op)]
            pending = [op for op in eligible_operations if (pair_id, op) not in completed]
            if executor is None:
                generated = [
                    (
                        operation,
                        core.generate_one(
                            generator_client, judge_client, source, operation,
                            args.server_model, args.server_model,
                            args.candidate_count, args.retries,
                        ),
                    )
                    for operation in pending
                ]
            else:
                futures = {
                    operation: executor.submit(
                        core.generate_one,
                        generator_client, judge_client, source, operation,
                        args.server_model, args.server_model,
                        args.candidate_count, args.retries,
                    )
                    for operation in pending
                }
                generated = [(operation, futures[operation].result()) for operation in pending]

            source_request_error = False
            for operation, (hypothesis, generation_log) in generated:
                key = (pair_id, operation)
                request_errors = collect_generation_errors(generation_log)
                if hypothesis is None and request_errors:
                    # Record the failure and keep going instead of aborting the
                    # whole run. The row is still appended below, so the
                    # generation log keeps its contiguous pair_id sequence and
                    # the pair counts as completed on resume.
                    generation_log["request_error"] = request_errors[0]
                    source_request_error = True
                generation_log.update(
                    {
                        "dataset": args.dataset,
                        "pair_id": pair_id,
                        "source_label": int(source["label"]),
                        "target_label": core.target_label(source, operation),
                    }
                )
                if hypothesis:
                    normalized_key = (
                        core.normalize_text(str(source["premise"])),
                        core.normalize_text(hypothesis),
                    )
                    if normalized_key in accepted_normalized_keys:
                        generation_log["status"] = "duplicate_rejected"
                        generation_log["duplicate_hypothesis"] = hypothesis
                        hypothesis = None
                if hypothesis:
                    generation_log["selected_hypothesis"] = hypothesis
                    components = core.component_mrs_from_log(operation, generation_log)
                    accepted[key] = followup_record(source, operation, hypothesis, components)
                    accepted_normalized_keys.add(normalized_key)
                logs.append(generation_log)
                completed.add(key)
            if source_request_error:
                consecutive_request_error_sources += 1
                if should_abort_on_request_errors(consecutive_request_error_sources):
                    raise RuntimeError(
                        f"model request failed for dataset={args.dataset} on "
                        f"{consecutive_request_error_sources} consecutive source "
                        f"pairs (last pair_id={pair_id}); aborting the run because "
                        f"this indicates a systemic model/service failure"
                    )
            else:
                consecutive_request_error_sources = 0
            processed_sources += 1
            print(
                f"[{args.dataset} {source_index}/{len(selected)}] pair_id={pair_id} "
                f"eligible={len(eligible_operations)} pending={len(pending)}",
                flush=True,
            )
            if processed_sources % args.checkpoint_every == 0 or source_index == len(selected):
                checkpoint()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    checkpoint()
    rows = read_jsonl(args.output)
    report = summarize(selected, rows, logs, operations)
    report.update(
        {
            "status": "complete",
            "dataset": args.dataset,
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "effective_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "server_model": args.server_model,
            "base_url": base_url,
            "operations": operations,
            "selected_pair_ids": [int(row["pair_id"]) for row in selected],
            "elapsed_seconds": round(time.time() - started, 2),
            "input_sha256": sha256_file(args.input),
            "output_sha256": sha256_file(args.output),
            "generation_log_sha256": sha256_file(log_path),
            "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
            "core_generator_sha256": sha256_file(Path(core.__file__).resolve()),
        }
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"generation log: {log_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
