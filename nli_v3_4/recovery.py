"""Explicit, lossless development migration for the JSON-envelope repair.

Original runs remain untouched. Cached judgments are decoded with the same label,
grammar and MR acceptance rules. No external scorer or shortcut prediction is read.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from . import core
from .calls import CALL_PROTOCOL
from .data import (DATASETS, canonical_source, digest, export_followup, export_source,
                   load_dataset, normalize, read_jsonl, sha256, write_json, write_jsonl)
from .protocol import PROTOCOL_VERSION
from .workflow import _code_hashes, source_rows


class NeedsInference(RuntimeError):
    pass


class OfflineClient:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        raise NeedsInference("this original request has unused inference stages")


def migrate(source_root, manifest_path, target_root):
    source_root, target_root = Path(source_root), Path(target_root)
    if target_root.exists():
        raise FileExistsError("recovery output already exists; originals must not be overwritten")
    target_root.mkdir(parents=True)
    summary = {"operation": "development_protocol_migration", "protocol": PROTOCOL_VERSION,
               "external_models_used": False, "generation_budget_increased": False, "datasets": {}}
    client = OfflineClient()
    for dataset in ("train", *DATASETS):
        old = source_root / "v3_4" / f"{dataset}.jsonl"
        _, info, sources = source_rows(manifest_path, dataset, "development")
        source_by_id = {r["pair_id"]: r for r in sources}
        metadata = json.loads(old.with_suffix(".run.json").read_text())
        if (metadata["stage"] != "development" or metadata["version"] != "v3.4"
                or metadata["manifest_sha256"] != sha256(manifest_path)
                or metadata["source_sha256"] != digest(sources)
                or metadata["contract_sha256"] != core.fingerprint()):
            raise ValueError("migration requires identical sources and unchanged generation prompts")
        logs = read_jsonl(old.with_suffix(".generation_log.jsonl"))
        if len({r["request_id"] for r in logs}) != len(logs):
            raise ValueError("duplicate original journal request")
        call_directory = old.with_suffix(".calls")
        if call_directory.exists():
            completed_ids = {entry["request_id"] for entry in logs
                             if not entry.get("pending_continuation")}
            unfinished = {p.name for p in call_directory.iterdir() if p.is_dir()} - completed_ids
            if unfinished:
                raise ValueError("resume unfinished durable requests with their original code before migration")
        accepted = [deepcopy(e["row"]) for e in logs if e.get("row")]
        counts = Counter(r["label"] for r in accepted)
        wanted = {int(k): v for k, v in metadata["label_quotas"].items()}
        seen = {(normalize(s["premise"]), normalize(s["hypothesis"])) for s in sources}
        seen.update((normalize(r["premise"]), normalize(r["hypothesis"])) for r in accepted)
        recovered, pending = 0, 0
        for entry in logs:
            source = source_by_id[entry["pair_id"]]
            operation = entry["operation"]
            entry["execution_code_hashes"] = metadata["code_hashes"]
            if entry.get("row"):
                text, _ = core.generate_one(client, source, operation,
                                            metadata["seed"] + entry["refill_round"], entry["trace"])
                if text != entry["row"]["hypothesis"]:
                    raise ValueError("cannot independently revalidate a preserved accepted row")
                entry["protocol_revalidated"] = True
                continue
            try:
                text, repaired = core.generate_one(client, source, operation,
                    metadata["seed"] + entry["refill_round"], entry["trace"])
            except NeedsInference:
                entry["pending_continuation"] = True
                pending += 1
                continue
            if text is None:
                continue
            entry["protocol_replay_trace"] = repaired
            key = (normalize(source["premise"]), normalize(text))
            label = core.LABEL_IDS[core.target_label(source, operation)]
            if counts[label] >= wanted[label]:
                entry["recovery_discard_reason"] = "label_quota_filled"
                continue
            if key in seen:
                entry["recovery_discard_reason"] = "duplicate_or_unchanged"
                continue
            steps = repaired.get("applied_steps")
            row = export_followup(source, operation, text, label,
                core.OPERATION_SPECS[operation]["mr_type"], steps, info["schema"])
            entry["row"] = row
            entry["row_origin"] = "cached_judgment_protocol_replay"
            accepted.append(row); seen.add(key); counts[label] += 1; recovered += 1
        grouped = defaultdict(list)
        for row in accepted:
            grouped[row["pair_id"]].append(row)
        rows = []
        for source in sources:
            rows.append(export_source(source, info["schema"])); rows.extend(grouped[source["pair_id"]])
        if info["schema"] == "test":
            for index, row in enumerate(rows):
                row["idx"] = index
        new = target_root / "v3_4" / f"{dataset}.jsonl"
        migrated = {**metadata, "code_hashes": _code_hashes(), "response_protocol": PROTOCOL_VERSION,
                    "call_protocol": CALL_PROTOCOL}
        write_json(new.with_suffix(".run.json"), migrated)
        write_jsonl(new.with_suffix(".generation_log.jsonl"), logs)
        write_jsonl(new, rows)
        complete = all(counts[l] == wanted[l] for l in range(3))
        write_json(new.with_suffix(".report.json"), {**migrated,
            "state": "generation_complete" if complete else "recovered_partial", "generation_complete": complete,
            "overall_complete": False, "sources": len(sources), "followups": len(accepted),
            "requests": len(logs), "label_counts": dict(counts),
            "label_deficits": {l: wanted[l]-counts[l] for l in range(3)},
            "mr_counts": dict(Counter(r["mr_id"] for r in accepted)),
            "quality_status": "awaiting_human_review", "output_sha256": sha256(new),
            "journal_sha256": sha256(new.with_suffix(".generation_log.jsonl"))})
        evidence = {"original_path": str(old.resolve()), "original_output_sha256": sha256(old),
            "original_journal_sha256": sha256(old.with_suffix(".generation_log.jsonl")),
            "original_metadata_sha256": sha256(old.with_suffix(".run.json")),
            "preserved_rows": sum(bool(e.get("row")) for e in read_jsonl(old.with_suffix(".generation_log.jsonl"))),
            "recovered_rows": recovered, "pending_continuations": pending,
            "prompts_unchanged": True, "inference_calls_during_migration": 0}
        write_json(new.with_suffix(".migration.json"), evidence)
        summary["datasets"][dataset] = evidence
    write_json(target_root / "migration.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.source_root, args.manifest, args.target_root)
