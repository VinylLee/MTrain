"""Backfill the ``component_mrs`` provenance field into an already-generated artifact.

The composite MRs (``composite_inv`` / ``composite_flip`` / ``composite_neutral``) are
built by ``_generate_composite`` from a fixed plan table, and the accepted plan is written
to the generation log as ``applied_steps``. That information never reached the output
JSON, so downstream consumers can only describe a composite row as "several edits".

This tool joins the generation log back onto the artifact and writes the component list
into each composite row. It is the *only* place ``component_mrs`` is produced for the
legacy artifact — the generator emits it natively from now on.

Design rules:

* **No LLM calls.** This repairs provenance, it does not regenerate data. Row text is
  never touched, so existing experiment results that consume this artifact stay valid.
* **Fail closed.** Every assumption is an assertion. If the log and the artifact cannot be
  put in exact 1:1 correspondence, the tool aborts without writing anything.
* **Line preserving.** Rows that already lack the field are rewritten one at a time; all
  other lines keep their original bytes. A second run is a byte-identical no-op.
* **Backup first.** The original is copied and its hash verified before any write.

Usage::

    python3 backfill_component_mrs.py --check      # validate only, write nothing
    python3 backfill_component_mrs.py              # apply
    python3 backfill_component_mrs.py --data <path> --generation-log <path> ...

``--force`` overrides the refusal to rewrite a row whose existing ``component_mrs``
disagrees with the log. Do not use it to paper over a mismatch — investigate first.
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from provenance_schema import (
    COMPONENT_MRS_FIELD,
    PROVENANCE_SCHEMA_VERSION,
    component_mrs_from_log,
    config_fingerprint,
    sha256_file,
)

DEFAULT_DATA = Path(
    "nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat"
    "/augmented_data_all_label_mrs_v3_3_full.json"
)
DEFAULT_GENERATOR_SOURCE = Path(__file__).resolve().parent / "augment_snli_all_label_mrs.py"

# Mirrors the whitelist used by the consuming repo (convert_nli_to_ft.compute_data_signature
# and sample_mettrain_pairid.SIGNATURE_FIELDS). It is duplicated here on purpose: this tool's
# central promise is that adding component_mrs does NOT change a split manifest's signature,
# and that promise should be enforced by the tool making the change rather than checked by
# hand. If the downstream list ever changes, revisit this constant.
SIGNATURE_FIELDS = (
    "pair_id", "mr_id", "premise", "hypothesis", "label", "idx", "id", "_source"
)

# Expected key order of an enriched composite row. The field sits directly after mr_id so
# that provenance travels with MR identity and the leading keys keep their relative order.
COMPOSITE_ROW_KEY_ORDER = (
    "premise", "hypothesis", "label", "pair_id", "is_syn", "weight",
    "mr_id", COMPONENT_MRS_FIELD, "mr_type",
)

# Recovered from a codex session transcript (see the plan). Never recorded anywhere in this
# repo, which is precisely the gap the provenance work closes. Marked unofficial because it
# was reconstructed from a third-party log rather than emitted by the run itself.
LEGACY_RUN_COMMAND_RECOVERED = (
    "python -u augment_snli_all_label_mrs.py --all --seed 42 "
    "--generator-model gemma-3-4b-it-qat --judge-model gemma-3-4b-it-qat "
    "--output nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat"
    "/augmented_data_all_label_mrs_v3_3_full.json "
    "--candidate-count 3 --retries 2 --workers 8 --min-total-rows 5340 "
    "--max-fill-rounds 4 --checkpoint-every 10 --resume"
)


class BackfillError(RuntimeError):
    """Raised when any fail-closed assertion fails. Nothing is written."""


class Checks:
    """Collects named pass/fail assertions so one run reports every problem at once."""

    def __init__(self) -> None:
        self.results: "collections.OrderedDict[str, bool]" = collections.OrderedDict()
        self.details: Dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: Any = None) -> None:
        self.results[name] = bool(ok)
        if detail is not None:
            self.details[name] = detail

    def enforce(self) -> None:
        failed = [name for name, ok in self.results.items() if not ok]
        if failed:
            lines = [f"  FAILED {name}: {self.details.get(name)!r}" for name in failed]
            raise BackfillError(
                f"{len(failed)} assertion(s) failed, nothing written:\n" + "\n".join(lines)
            )

    @property
    def all_passed(self) -> bool:
        return all(self.results.values())


# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────

def read_jsonl_lines(path: Path) -> Tuple[List[bytes], bytes]:
    """Return (lines_without_terminator, terminator) preserving original bytes."""
    raw = path.read_bytes()
    if b"\r" in raw:
        raise BackfillError(f"{path}: CR found; this tool expects LF-only line endings")
    if not raw.endswith(b"\n"):
        raise BackfillError(f"{path}: expected a trailing newline")
    return raw.split(b"\n")[:-1], b"\n"


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def snapshot_generator_constants(source_path: Path) -> Dict[str, Any]:
    """Recover the generation contract from the generator source via AST.

    Used when the artifact's report.json predates ``config_fingerprint``. Only literal
    assignments are read, so the generator is never imported (``openai`` may be absent).

    Both ``Assign`` and ``AnnAssign`` are handled: OPERATION_SPECS and
    COMPOSITE_STAGE_PLANS carry annotations, only PROMPT_VERSION does not.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"PROMPT_VERSION", "OPERATION_SPECS", "COMPOSITE_STAGE_PLANS"}
    found: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    if missing:
        raise BackfillError(f"{source_path}: could not recover constants {sorted(missing)}")
    return found


def resolve_config_fingerprint(
    report: Dict[str, Any], snapshot: Dict[str, Any], generator_source: Path
) -> Tuple[str, str]:
    """Return (fingerprint, provenance-of-the-fingerprint)."""
    if report.get("config_fingerprint"):
        return str(report["config_fingerprint"]), "report.json"
    return (
        config_fingerprint(
            snapshot["PROMPT_VERSION"],
            snapshot["OPERATION_SPECS"],
            snapshot["COMPOSITE_STAGE_PLANS"],
        ),
        "ast_snapshot_of_generator_source",
    )


def signature_of(rows: Sequence[Dict[str, Any]]) -> str:
    """Reproduce the downstream split-manifest signature."""
    canonical = [{k: r.get(k) for k in SIGNATURE_FIELDS if k in r} for r in rows]
    canonical.sort(key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":")))
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def insert_after(row: Dict[str, Any], anchor: str, key: str, value: Any) -> Dict[str, Any]:
    """Rebuild ``row`` with ``key`` inserted immediately after ``anchor``."""
    if anchor not in row:
        raise BackfillError(f"row has no anchor key {anchor!r} to insert {key!r} after")
    if key in row:
        # Copying the existing key while iterating would overwrite the inserted value with
        # the stale one; callers that update in place must not come through here.
        raise BackfillError(f"row already has {key!r}; use in-place assignment instead")
    rebuilt: Dict[str, Any] = {}
    for existing_key, existing_value in row.items():
        rebuilt[existing_key] = existing_value
        if existing_key == anchor:
            rebuilt[key] = value
    return rebuilt


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = Path(args.data)
    log_path = Path(args.generation_log) if args.generation_log else data_path.with_suffix(".generation_log.jsonl")
    report_path = Path(args.report) if args.report else data_path.with_suffix(".report.json")
    backup_path = Path(args.backup) if args.backup else data_path.with_name(
        f"{data_path.stem}.pre_component_mrs_{datetime.date.today():%Y%m%d}{data_path.suffix}"
    )
    backfill_report_path = Path(args.backfill_report) if args.backfill_report else data_path.with_suffix(".component_mrs_backfill.json")
    generator_source = Path(args.generator_source)

    for required in (data_path, log_path, report_path, generator_source):
        if not required.exists():
            raise BackfillError(f"missing required input: {required}")

    checks = Checks()

    # ── Provenance of the run this artifact came from ──────────────────────────
    snapshot = snapshot_generator_constants(generator_source)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    composite_operations = set(snapshot["COMPOSITE_STAGE_PLANS"])
    operation_specs = snapshot["OPERATION_SPECS"]
    fingerprint, fingerprint_source = resolve_config_fingerprint(report, snapshot, generator_source)
    generator_source_sha256 = sha256_file(generator_source)

    checks.check(
        "report.prompt_version matches generator source",
        report.get("prompt_version") == snapshot["PROMPT_VERSION"],
        {"report": report.get("prompt_version"), "source": snapshot["PROMPT_VERSION"]},
    )
    checks.check(
        "report.label_mrs matches generator OPERATION_SPECS order",
        list(report.get("label_mrs") or []) == list(operation_specs),
        {"report": report.get("label_mrs"), "source": list(operation_specs)},
    )

    log_lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    operation_results = report.get("operation_results") or {}
    attempted_ok, accepted_ok = True, True
    attempted_detail, accepted_detail = {}, {}
    for operation in operation_specs:
        logged = sum(1 for entry in log_lines if entry.get("operation") == operation)
        accepted = sum(1 for entry in log_lines
                       if entry.get("operation") == operation and entry.get("status") == "accepted")
        want = (operation_results.get(operation) or {})
        attempted_detail[operation] = [want.get("attempted"), logged]
        accepted_detail[operation] = [want.get("accepted"), accepted]
        if want.get("attempted") != logged:
            attempted_ok = False
        if want.get("accepted") != accepted:
            accepted_ok = False
    checks.check("report.attempted counts match generation log", attempted_ok, attempted_detail)
    checks.check("report.accepted counts match generation log", accepted_ok, accepted_detail)
    checks.check(
        "per-operation attempted counts sum to log length",
        sum(entry.get("attempted") or 0 for entry in operation_results.values()) == len(log_lines),
        {"sum": sum(entry.get("attempted") or 0 for entry in operation_results.values()),
         "log_lines": len(log_lines)},
    )
    checks.check("generation log is non-empty", len(log_lines) > 0, len(log_lines))

    # ── Artifact ───────────────────────────────────────────────────────────────
    lines, terminator = read_jsonl_lines(data_path)
    rows = [json.loads(line) for line in lines]
    sha256_before = sha256_file(data_path)
    bytes_before = data_path.stat().st_size

    checks.check(
        "report source/synthetic counts match artifact row count",
        (report.get("source_count") or 0) + (report.get("synthetic_count") or 0) == len(rows),
        {"report": (report.get("source_count") or 0) + (report.get("synthetic_count") or 0),
         "rows": len(rows)},
    )

    if report.get("output"):
        recorded = Path(report["output"])
        if not recorded.is_absolute():
            recorded = Path(args.cwd_of_recorded_run).resolve() / recorded
        checks.check(
            "report.output resolves to this artifact",
            recorded.resolve() == data_path.resolve(),
            {"report_output": str(recorded.resolve()), "data": str(data_path.resolve())},
        )

    # ── Build the join index from ACCEPTED composite entries only ──────────────
    # Filtering on status is load-bearing: composite entries whose hypothesis was a
    # duplicate keep applied_steps but have status rewritten to "duplicate_rejected",
    # and they never produce an output row. Indexing on applied_steps presence alone
    # would over-count and trip the bijection assertion.
    index: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for entry in log_lines:
        operation = entry.get("operation")
        if operation in composite_operations and entry.get("status") == "accepted":
            key = (int(entry["pair_id"]), operation)
            if key in index:
                raise BackfillError(f"duplicate accepted composite log entry for {key}")
            index[key] = entry

    composite_rows = [r for r in rows if r.get("mr_id") in composite_operations]

    # ── Enrich ─────────────────────────────────────────────────────────────────
    new_lines: List[bytes] = []
    after_rows: List[Dict[str, Any]] = []
    changed = 0
    unmatched_rows: List[Any] = []
    mismatched: List[Any] = []
    component_distribution: "collections.Counter[Tuple[str, ...]]" = collections.Counter()

    for line, row in zip(lines, rows):
        mr_id = row.get("mr_id")
        has_field = COMPONENT_MRS_FIELD in row

        if mr_id not in composite_operations:
            if has_field:
                raise BackfillError(
                    f"non-composite row (mr_id={mr_id!r}, pair_id={row.get('pair_id')!r}) "
                    f"carries {COMPONENT_MRS_FIELD}; refusing to continue"
                )
            new_lines.append(line)
            after_rows.append(row)
            continue

        key = (int(row["pair_id"]), mr_id)
        entry = index.pop(key, None)
        if entry is None:
            unmatched_rows.append(key)
            new_lines.append(line)
            after_rows.append(row)
            continue

        try:
            components = component_mrs_from_log(
                mr_id, entry,
                composite_operations=composite_operations,
                operation_specs=operation_specs,
            )
        except ValueError as exc:
            # Surface as a BackfillError so the CLI reports "ABORTED: ..." instead of a
            # traceback, and so nothing is written.
            raise BackfillError(f"pair_id={key[0]} mr_id={mr_id}: {exc}") from exc
        assert components is not None  # composite_operations membership is checked above
        component_distribution[tuple(components)] += 1

        if has_field and list(row[COMPONENT_MRS_FIELD] or []) == list(components):
            # Already correct: keep the original bytes so a re-run is a true no-op.
            new_lines.append(line)
            after_rows.append(row)
            continue

        if has_field:
            mismatched.append({"key": list(key), "existing": row[COMPONENT_MRS_FIELD],
                               "from_log": components})
            if not args.force:
                new_lines.append(line)
                after_rows.append(row)
                continue

        if has_field:
            # --force: repair rather than merely overwrite. Strip the field first so a row
            # carrying it in the wrong position is normalized back to canonical order.
            stripped = {k: v for k, v in row.items() if k != COMPONENT_MRS_FIELD}
            enriched = insert_after(stripped, "mr_id", COMPONENT_MRS_FIELD, list(components))
        else:
            enriched = insert_after(row, "mr_id", COMPONENT_MRS_FIELD, list(components))
        if tuple(enriched) != COMPOSITE_ROW_KEY_ORDER:
            raise BackfillError(
                f"unexpected key order for pair_id={key[0]} mr_id={mr_id}: {tuple(enriched)}"
            )
        new_lines.append(json.dumps(enriched, ensure_ascii=False).encode("utf-8"))
        after_rows.append(enriched)
        changed += 1

    accepted_composite_entries = sum(
        1 for entry in log_lines
        if entry.get("operation") in composite_operations and entry.get("status") == "accepted"
    )
    checks.check("every composite row matched a log entry", not unmatched_rows, unmatched_rows)
    checks.check("every accepted composite log entry was consumed", not index, sorted(index))
    checks.check(
        "accepted composite log entry count equals composite row count",
        len(composite_rows) == accepted_composite_entries,
        {"rows": len(composite_rows), "accepted_entries": accepted_composite_entries},
    )
    checks.check(
        "existing component_mrs never disagreed with the log",
        not mismatched or args.force,
        mismatched,
    )
    checks.check(
        "every composite row carries the field after backfill",
        all(COMPONENT_MRS_FIELD in r for r in after_rows
            if r.get("mr_id") in composite_operations),
        {"composite_rows": len(composite_rows), "enriched_now": changed},
    )

    # ── Mutation safety ────────────────────────────────────────────────────────
    after_raw = terminator.join(new_lines) + terminator
    sha256_after = hashlib.sha256(after_raw).hexdigest()

    checks.check("line count unchanged", len(new_lines) == len(lines),
                 {"before": len(lines), "after": len(new_lines)})

    byte_drift = [i for i, (old, new) in enumerate(zip(lines, new_lines))
                  if old != new and rows[i].get("mr_id") not in composite_operations]
    checks.check("non-composite lines are byte-identical", not byte_drift, byte_drift[:20])

    changed_idx = [i for i, (old, new) in enumerate(zip(lines, new_lines)) if old != new]
    composite_idx = [i for i, r in enumerate(rows) if r.get("mr_id") in composite_operations]
    checks.check(
        "only composite lines changed",
        set(changed_idx).issubset(set(composite_idx)) and len(changed_idx) == changed,
        {"changed": len(changed_idx), "composite_rows": len(composite_idx)},
    )

    checks.check(
        "label and mr_id distributions unchanged",
        collections.Counter((r.get("mr_id"), r.get("label")) for r in rows)
        == collections.Counter((r.get("mr_id"), r.get("label")) for r in after_rows),
    )
    signature_before = signature_of(rows)
    signature_after = signature_of(after_rows)
    checks.check(
        "downstream split-manifest signature unchanged",
        signature_before == signature_after,
        {"before": signature_before, "after": signature_after},
    )

    checks.enforce()

    # ── Write ──────────────────────────────────────────────────────────────────
    status = "check_only"
    backup_created = False
    backup_reused = False
    if not args.check:
        if changed == 0:
            # Nothing to write. An existing backup is the pre-backfill state and is simply
            # left alone; it is expected to differ from the already-enriched current data.
            status = "already_applied"
            backup_reused = backup_path.exists()
        else:
            if args.no_backup:
                print("!! --no-backup: no rollback copy will be written", file=sys.stderr)
            elif backup_path.exists():
                # We are about to overwrite the data, so the backup must hold exactly the
                # bytes being replaced.
                if sha256_file(backup_path) != sha256_before:
                    raise BackfillError(
                        f"{backup_path} exists but does not match the current data; refusing "
                        "to overwrite it — move it aside first"
                    )
                backup_reused = True
            else:
                shutil.copy2(data_path, backup_path)
                backup_created = True
                if sha256_file(backup_path) != sha256_before:
                    raise BackfillError("backup hash does not match the source; aborting")

            temporary = data_path.with_suffix(data_path.suffix + ".tmp")
            temporary.write_bytes(after_raw)
            temporary.replace(data_path)
            status = "applied"
            if sha256_file(data_path) != sha256_after:
                raise BackfillError("written file hash does not match the computed hash")

    # ── Report ─────────────────────────────────────────────────────────────────
    backfill_report = {
        "schema_version": 1,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": status,
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "data": {
            "path": str(data_path),
            "sha256_before": sha256_before,
            "sha256_after": sha256_after,
            "bytes_before": bytes_before,
            "bytes_after": len(after_raw),
            "row_count_before": len(lines),
            "row_count_after": len(new_lines),
            "changed_lines": len(changed_idx),
            "unchanged_lines": len(lines) - len(changed_idx),
        },
        "generation_log": {
            "path": str(log_path),
            "sha256": sha256_file(log_path),
            "line_count": len(log_lines),
        },
        "generator_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "prompt_version": report.get("prompt_version"),
            "config_fingerprint": fingerprint,
            "config_fingerprint_source": fingerprint_source,
            "generator_source": {
                "path": str(generator_source),
                "sha256": generator_source_sha256,
            },
        },
        "backup": {
            "path": str(backup_path),
            "sha256": sha256_before,
            "created": backup_created,
            "reused": backup_reused,
            "enabled": not args.no_backup and not args.check,
        },
        "join": {
            "composite_operations": sorted(composite_operations),
            "accepted_composite_log_entries": sum(
                1 for entry in log_lines
                if entry.get("operation") in composite_operations
                and entry.get("status") == "accepted"
            ),
            "composite_rows": len(composite_rows),
            "composite_rows_enriched_now": changed,
            "unmatched_rows": [list(k) for k in unmatched_rows],
            "unmatched_log_entries": [list(k) for k in sorted(index)],
            "bijection": not unmatched_rows and not index,
        },
        "component_distribution": {
            "+".join(k): v for k, v in sorted(component_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "component_length_counts": dict(sorted(
            collections.Counter(len(k) for k in component_distribution).items()
        )),
        "validation": dict(checks.results),
        "validation_details": checks.details,
        "signatures": {
            "signature_before": signature_before,
            "signature_after": signature_after,
            "signature_fields": list(SIGNATURE_FIELDS),
            "unchanged": signature_before == signature_after,
        },
        "notes": [
            "component_mrs is written only on composite rows; atomic and source rows keep "
            "their original bytes.",
            "Signature uses the downstream whitelist and excludes component_mrs, so split "
            "manifests and conversion reports that reference this artifact stay valid.",
        ],
        "legacy_run_command_recovered": {
            "command_line": LEGACY_RUN_COMMAND_RECOVERED,
            "source": "codex session transcript /home/ubuntu/.codex/sessions/2026/09/09/",
            "confidence": "unofficial_recovery",
            "note": "reconstructed from a third-party log; not emitted by the run itself",
        },
    }

    if not args.check:
        backfill_report_path.parent.mkdir(parents=True, exist_ok=True)
        backfill_report_path.write_text(
            json.dumps(backfill_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return backfill_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the component_mrs provenance field into a generated artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 backfill_component_mrs.py --check\n"
            "  python3 backfill_component_mrs.py\n"
        ),
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--generation-log", default=None,
                        help="default: <data>.generation_log.jsonl")
    parser.add_argument("--report", default=None, help="default: <data>.report.json")
    parser.add_argument("--backup", default=None,
                        help="default: <data stem>.pre_component_mrs_<YYYYMMDD>.json")
    parser.add_argument("--backfill-report", default=None,
                        help="default: <data stem>.component_mrs_backfill.json")
    parser.add_argument("--generator-source", default=str(DEFAULT_GENERATOR_SOURCE))
    parser.add_argument("--cwd-of-recorded-run", default=str(DEFAULT_GENERATOR_SOURCE.parent),
                        help="cwd against which a relative report.output path is resolved "
                             "(default: the generator's directory)")
    parser.add_argument("--check", action="store_true",
                        help="validate and report only; write nothing")
    parser.add_argument("--no-backup", action="store_true",
                        help="skip the rollback copy (discouraged)")
    parser.add_argument("--force", action="store_true",
                        help="rewrite rows whose existing component_mrs disagrees with the log")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except BackfillError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(
        {
            "status": result["status"],
            "data": result["data"],
            "join": result["join"],
            "component_distribution": result["component_distribution"],
            "validation_passed": all(result["validation"].values()),
        },
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
