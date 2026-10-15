"""Prepare/run a bounded development quality diagnostic; never production or refill.

Preparation and validation make no API calls. Running requires a separately agreed
budget: it does not draw new slots from the exhausted round-3 experiment.
"""
from collections import Counter, defaultdict
import argparse
import fcntl
import json
from pathlib import Path

from nli_v3_4 import core, workflow
from nli_v3_4.calls import DurableRequestClient, _save
from nli_v3_4.data import (canonical_source, digest, export_followup, export_source,
                         load_dataset, read_jsonl, sha256, source_uid, write_jsonl)
from nli_v3_4.protocol import has_api_failure

ORIGINS = ("train", "snli", "mnlim", "mnlimm", "sick")
OPERATIONS = ("voice_switch", "synonym_replacement", "adding_contradiction",
              "uninformative", "composite_inv")
KIND = "development_quality_repair_micro_pilot_v1"


def code_hashes():
    return {**workflow._code_hashes(), Path(__file__).name: sha256(__file__)}


def frozen_sources(manifest_path):
    path = Path(manifest_path)
    manifest = workflow._validate_manifest(path)
    sources = {}
    for origin in ORIGINS:
        info = manifest["datasets"][origin]
        if sha256(info["input"]) != info["sha256"]:
            raise ValueError("frozen full input changed")
        _, rows = load_dataset(path.parent / "development_sources" / f"{origin}.jsonl")
        if (sorted(r["pair_id"] for r in rows) != info["development_pair_ids"]
                or set(map(source_uid, rows)) != set(info["development_source_uids"])):
            raise ValueError("frozen development sources changed")
        sources[origin] = [canonical_source(row) for row in rows]
    return manifest, sources


def prepare(manifest_path, previous_review_key, output, seed=42):
    """Two requests/MR/origin, with source groups excluded after prior human exposure."""
    output = Path(output)
    if seed != 42:
        raise ValueError("pilot seed is fixed at 42")
    if output.exists():
        raise FileExistsError(output)
    _, sources = frozen_sources(manifest_path)
    excluded = defaultdict(set)
    for key in read_jsonl(previous_review_key):
        excluded[key["dataset"]].add(key["pair_id"])
    requests = []
    for origin in ORIGINS:
        used = set()
        for operation in OPERATIONS:
            eligible = [s for s in sources[origin] if s["pair_id"] not in excluded[origin] | used
                        and core.eligible(s, operation)]
            eligible.sort(key=lambda s: digest([s["hypothesis"], operation, seed]))
            if len(eligible) < 2:
                raise ValueError(f"not enough unreviewed eligible sources: {origin}/{operation}")
            for source in eligible[:2]:
                used.add(source["pair_id"])
                requests.append({"request_id": digest([KIND, origin, source_uid(source), operation, seed]),
                    "dataset": origin, "pair_id": source["pair_id"], "source_uid": source_uid(source),
                    "operation": operation, "seed": seed,
                    "max_calls": 9 if operation in core.COMPOSITE_STAGE_PLANS else 4})
    plan = {"kind": KIND, "stage": "development", "scope": "quality_diagnostic_not_final_gate",
            "run_output": str((output.parent / "execution").resolve()),
            "semantic_prompt_round": 3, "contract_sha256": core.fingerprint(), "code_hashes": code_hashes(),
            "manifest": str(Path(manifest_path).resolve()), "manifest_sha256": sha256(manifest_path),
            "previous_review_key": str(Path(previous_review_key).resolve()),
            "previous_review_key_sha256": sha256(previous_review_key),
            "sampling": "two_per_MR_origin_unique_unreviewed_sources_hypothesis_hash_seed42_v1",
            "requests": requests, "max_requests": len(requests),
            "max_model_calls": sum(r["max_calls"] for r in requests), "refill_rounds": 0,
            "external_nli_in_pipeline": False, "production_allowed": False}
    plan["plan_fingerprint"] = digest(plan)
    _save(output, plan)
    return plan


def validate(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    claimed = plan.pop("plan_fingerprint")
    if digest(plan) != claimed:
        raise ValueError("pilot plan changed")
    plan["plan_fingerprint"] = claimed
    if (plan["kind"] != KIND or plan["stage"] != "development"
            or plan["production_allowed"] is not False or plan["external_nli_in_pipeline"] is not False
            or plan["semantic_prompt_round"] != 3 or plan["refill_rounds"] != 0
            or plan["contract_sha256"] != core.fingerprint() or plan["code_hashes"] != code_hashes()):
        raise ValueError("pilot scope, contract or execution code changed")
    if sha256(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    if sha256(plan["previous_review_key"]) != plan["previous_review_key_sha256"]:
        raise ValueError("previous review exposure changed")
    _, sources = frozen_sources(plan["manifest"])
    by_id = {origin: {s["pair_id"]: s for s in rows} for origin, rows in sources.items()}
    counts, used = Counter(), set()
    excluded = {(r["dataset"], r["pair_id"]) for r in read_jsonl(plan["previous_review_key"])}
    for request in plan["requests"]:
        origin, pair_id, operation = request["dataset"], request["pair_id"], request["operation"]
        source = by_id[origin][pair_id]
        key = origin, pair_id
        if (key in used or key in excluded or operation not in OPERATIONS
                or request["seed"] != 42 or source_uid(source) != request["source_uid"]
                or not core.eligible(source, operation)
                or request["max_calls"] != (9 if operation in core.COMPOSITE_STAGE_PLANS else 4)
                or request["request_id"] != digest([KIND, origin, source_uid(source), operation, 42])):
            raise ValueError("invalid/review-exposed pilot request")
        used.add(key); counts[origin, operation] += 1
    if (counts != Counter({(d, op): 2 for d in ORIGINS for op in OPERATIONS})
            or len(plan["requests"]) != 50
            or plan["max_requests"] != 50 or plan["max_model_calls"] != 250
            or sum(r["max_calls"] for r in plan["requests"]) != 250):
        raise ValueError("pilot budget or strata changed")
    return plan, sources


class BoundedRequestClient(DurableRequestClient):
    def __init__(self, client, directory, max_calls):
        super().__init__(client, directory)
        self.max_calls = max_calls

    def create(self, **kwargs):
        if self.sequence >= self.max_calls:
            raise RuntimeError("quality pilot request budget exhausted; no dispatch")
        return super().create(**kwargs)


def run(plan_path, output, *, resume=False, base_url="http://127.0.0.1:1234/v1", timeout=120):
    plan, sources = validate(plan_path)  # Fail before client construction/output writes.
    output = Path(output)
    if output.resolve() != Path(plan["run_output"]):
        raise ValueError("pilot budget is bound to its single planned output directory")
    if not resume and output.exists():
        raise FileExistsError(output)
    if resume and not (output / "run.json").exists():
        raise FileNotFoundError("resume requires original run metadata")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"kind": KIND, "stage": "development", "plan_sha256": sha256(plan_path),
                  "contract_sha256": core.fingerprint(), "code_hashes": code_hashes(),
                  "base_url": base_url, "timeout": timeout, "max_requests": 50,
                  "max_model_calls": 250, "production_allowed": False}
        meta, journal = output / "run.json", output / "entries.json"
        if resume:
            if json.loads(meta.read_text()) != config:
                raise ValueError("pilot resume configuration changed")
            entries = json.loads(journal.read_text())
        else:
            _save(meta, config); _save(journal, [])
            entries = []
        requests = {r["request_id"]: r for r in plan["requests"]}
        if len({e["request_id"] for e in entries}) != len(entries):
            raise ValueError("duplicate pilot journal request")
        for entry in entries:
            if entry["request_id"] not in requests or entry["request"] != requests[entry["request_id"]]:
                raise ValueError("journal differs from frozen pilot request")
        processed = {entry["request_id"] for entry in entries}
        by_id = {d: {s["pair_id"]: s for s in rows} for d, rows in sources.items()}
        def checkpoint(state):
            for origin in ORIGINS:
                schema = "train" if origin == "train" else "test"
                grouped = defaultdict(list)
                for entry in entries:
                    if entry["request"]["dataset"] == origin and entry.get("row"):
                        grouped[entry["request"]["pair_id"]].append(entry["row"])
                rows = []
                for source in sources[origin]:
                    rows.append(export_source(source, schema)); rows.extend(grouped[source["pair_id"]])
                if schema == "test":
                    for i, row in enumerate(rows): row["idx"] = i
                path = output / "v3_4" / f"{origin}.jsonl"
                write_jsonl(path, rows)
                _save(path.with_suffix(".run.json"), {**config, "dataset": origin,
                    "manifest_sha256": plan["manifest_sha256"], "source_sha256": digest(sources[origin])})
            calls = list((output / "calls").glob("*/*.json"))
            if len(calls) > 250:
                raise RuntimeError("pilot call ledger exceeds declared budget")
            _save(output / "status.json", {**config, "state": state, "requests_completed": len(entries),
                "outputs": sum(e.get("row") is not None for e in entries), "call_slots_consumed": len(calls),
                "requests_complete": len(entries) == 50, "quality_passed": False,
                "full_goal_achieved": False, "production_allowed": False})
        checkpoint("prepared")
        if len(processed) == 50:
            checkpoint("requests_complete"); return
        from openai import OpenAI
        client = OpenAI(api_key="local", base_url=base_url, timeout=timeout, max_retries=0)
        errors, state = 0, "interrupted"
        try:
            for request in plan["requests"]:
                if request["request_id"] in processed: continue
                source = by_id[request["dataset"]][request["pair_id"]]
                bounded = BoundedRequestClient(client, output / "calls" / request["request_id"], request["max_calls"])
                text, trace = core.generate_one(bounded, source, request["operation"], request["seed"])
                label = core.LABEL_IDS[core.target_label(source, request["operation"])]
                row = export_followup(source, request["operation"], text, label,
                    core.OPERATION_SPECS[request["operation"]]["mr_type"], trace.get("applied_steps"),
                    "train" if request["dataset"] == "train" else "test") if text else None
                entries.append({"request_id": request["request_id"], "request": request, "trace": trace, "row": row})
                _save(journal, entries); processed.add(request["request_id"])
                checkpoint("running")
                print(f"quality pilot requests={len(entries)}/50 outputs={sum(e.get('row') is not None for e in entries)}", flush=True)
                errors = errors + 1 if has_api_failure(trace) else 0
                if errors >= 3:
                    state = "service_error"
                    raise RuntimeError("three consecutive API failures; preserve pilot and stop")
            state = "requests_complete"
        finally:
            checkpoint(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--previous-review-key", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("--plan", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.manifest, args.previous_review_key, args.output)
        print(json.dumps({"requests": plan["max_requests"], "max_model_calls": plan["max_model_calls"], "api_calls": 0}))
    elif args.command == "validate":
        plan, _ = validate(args.plan)
        print(json.dumps({"valid": True, "requests": plan["max_requests"], "max_model_calls": plan["max_model_calls"], "api_calls": 0}))
    else:
        run(args.plan, args.output, resume=args.resume, base_url=args.base_url)


if __name__ == "__main__":
    main()
