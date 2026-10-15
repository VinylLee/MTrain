"""Bounded post-review development pilot; prepare/validate/audit make no API calls."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path

import run_nli_quality_repair_pilot as previous
from nli_v3_4 import core
from nli_v3_4.calls import _save
from nli_v3_4.data import (canonical_source, digest, export_followup, export_source, is_source,
                         load_dataset, read_jsonl, sha256, source_uid, write_jsonl)
from nli_v3_4.protocol import has_api_failure
from nli_v3_4.refinement import RefinementEngine, REVISION, call_cap, token_cap

ENGINE = RefinementEngine()
KIND = "post_review_typed_detail_single_pair_development_pilot_v1"


def code_hashes():
    return {**previous.code_hashes(), "nli_v3_4/redesign.py": sha256("nli_v3_4/redesign.py"),
            "nli_v3_4/refinement.py": sha256("nli_v3_4/refinement.py"), Path(__file__).name: sha256(__file__)}


def prepare(manifest_path, review_keys, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    _, sources = previous.frozen_sources(manifest_path)
    excluded = {(r["dataset"], r["pair_id"]) for key in review_keys for r in read_jsonl(key)}
    requests = []
    for origin in previous.ORIGINS:
        used = set()
        for op in previous.OPERATIONS:
            candidates = [s for s in sources[origin] if (origin, s["pair_id"]) not in excluded
                and s["pair_id"] not in used and ENGINE.eligible(s, op)]
            candidates.sort(key=lambda s: digest([s["hypothesis"], op, 42, REVISION]))
            if not candidates: raise ValueError(f"no unreviewed source for {origin}/{op}")
            s = candidates[0]; used.add(s["pair_id"])
            requests.append({"request_id": digest([REVISION, origin, source_uid(s), op, 42]),
                "dataset": origin, "pair_id": s["pair_id"], "source_uid": source_uid(s),
                "operation": op, "seed": 42, "max_calls": call_cap(op)})
    plan = {"kind": KIND, "stage": "development", "engine": "refinement", "revision": REVISION,
        "scope": "post_review_development_not_part_of_original_three_round_experiment",
        "manifest": str(Path(manifest_path).resolve()), "manifest_sha256": sha256(manifest_path),
        "previous_review_keys": {str(Path(k).resolve()): sha256(k) for k in review_keys},
        "contract": ENGINE.contract(), "contract_sha256": ENGINE.fingerprint(), "code_hashes": code_hashes(),
        "requests": requests, "max_requests": 25, "max_model_calls": 295, "max_output_tokens": 144640,
        "refill_rounds": 0, "production_allowed": False, "external_nli_in_generation": False,
        "run_output": str((output.parent / "execution").resolve())}
    plan["plan_fingerprint"] = digest(plan); _save(output, plan)
    return plan


def validate(path):
    plan = json.loads(Path(path).read_text()); claimed = plan.pop("plan_fingerprint")
    if digest(plan) != claimed: raise ValueError("refinement plan changed")
    plan["plan_fingerprint"] = claimed
    if (plan["kind"] != KIND or plan["stage"] != "development" or plan["engine"] != "refinement"
            or plan["revision"] != REVISION or plan["contract"] != ENGINE.contract()
            or plan["contract_sha256"] != ENGINE.fingerprint() or plan["code_hashes"] != code_hashes()
            or plan["max_requests"] != 25 or plan["max_model_calls"] != 295 or plan["max_output_tokens"] != 144640
            or plan["production_allowed"] is not False or plan["external_nli_in_generation"] is not False
            or plan["refill_rounds"] != 0): raise ValueError("refinement scope/contract/budget changed")
    if sha256(plan["manifest"]) != plan["manifest_sha256"]: raise ValueError("manifest changed")
    excluded = set()
    for key, expected in plan["previous_review_keys"].items():
        if sha256(key) != expected: raise ValueError("review exposure changed")
        excluded.update((r["dataset"], r["pair_id"]) for r in read_jsonl(key))
    _, sources = previous.frozen_sources(plan["manifest"])
    seen = set(); strata = Counter()
    for r in plan["requests"]:
        s = next(s for s in sources[r["dataset"]] if s["pair_id"] == r["pair_id"])
        key = (r["dataset"], r["pair_id"]); op = r["operation"]
        if (key in seen or key in excluded or op not in previous.OPERATIONS or r["seed"] != 42
                or r["source_uid"] != source_uid(s) or not ENGINE.eligible(s, op)
                or r["max_calls"] != call_cap(op)
                or r["request_id"] != digest([REVISION, r["dataset"], source_uid(s), op, 42])):
            raise ValueError("invalid refinement request")
        seen.add(key); strata[r["dataset"], op] += 1
    if (strata != Counter({(d, op): 1 for d in previous.ORIGINS for op in previous.OPERATIONS})
            or len(plan["requests"]) != 25 or sum(r["max_calls"] for r in plan["requests"]) != 295
            or sum(token_cap(r["operation"]) for r in plan["requests"]) != 144640):
        raise ValueError("refinement request strata changed")
    return plan, sources


def run(path, resume=False, base_url="http://127.0.0.1:1234/v1"):
    plan, sources = validate(path); out = Path(plan["run_output"])
    if out.exists() and not resume: raise FileExistsError(out)
    if resume and not (out/"run.json").exists(): raise FileNotFoundError("resume metadata missing")
    out.mkdir(parents=True, exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"kind": KIND, "stage": "development", "engine": "refinement",
            "plan_sha256": sha256(path), "contract_sha256": ENGINE.fingerprint(), "code_hashes": code_hashes(),
            "base_url": base_url, "manifest_sha256": plan["manifest_sha256"], "max_requests": 25,
            "max_model_calls": 295, "max_output_tokens": 144640, "production_allowed": False}
        if resume:
            if json.loads((out/"run.json").read_text()) != config: raise ValueError("resume identity changed")
            entries = json.loads((out/"entries.json").read_text())
        else:
            entries = []; _save(out/"run.json", config); _save(out/"entries.json", entries)
        requests = {r["request_id"]: r for r in plan["requests"]}
        if len({e["request_id"] for e in entries}) != len(entries): raise ValueError("duplicate request")
        for e in entries:
            if e["request"] != requests.get(e["request_id"]): raise ValueError("saved request changed")
            if e.get("row"):
                r = e["request"]; s = next(s for s in sources[r["dataset"]] if s["pair_id"] == r["pair_id"])
                if ENGINE.replay_one(s, r["operation"], e["trace"], 42) != e["row"]["hypothesis"]:
                    raise ValueError("saved row did not replay")
        completed = {e["request_id"] for e in entries}
        def checkpoint(state):
            for origin, originals in sources.items():
                grouped = defaultdict(list)
                for e in entries:
                    if e.get("row") and e["request"]["dataset"] == origin:
                        grouped[e["request"]["pair_id"]].append(e["row"])
                rows = []
                for s in originals:
                    rows.append(export_source(s, "train" if origin == "train" else "test"))
                    rows.extend(grouped[s["pair_id"]])
                if origin != "train":
                    for i, row in enumerate(rows): row["idx"] = i
                file = out/"v3_4"/f"{origin}.jsonl"; write_jsonl(file, rows)
                _save(file.with_suffix(".run.json"), {**config, "dataset": origin, "source_sha256": digest(originals)})
            slots = list((out/"calls").glob("*/*.json"))
            if len(slots) > 295: raise RuntimeError("refinement call ceiling exceeded")
            _save(out/"status.json", {**config, "state": state, "requests_completed": len(entries),
                "requests_complete": len(entries) == 25, "call_slots_consumed": len(slots),
                "outputs": sum(e.get("row") is not None for e in entries), "human_review_complete": False,
                "quality_passed": False, "goal_complete": False})
        checkpoint("prepared")
        if len(entries) == 25: checkpoint("requests_complete"); return
        from openai import OpenAI
        api = OpenAI(base_url=base_url, api_key="local", timeout=120, max_retries=0)
        state = "interrupted"; errors = 0
        try:
            for r in plan["requests"]:
                if r["request_id"] in completed: continue
                s = next(s for s in sources[r["dataset"]] if s["pair_id"] == r["pair_id"])
                bounded = previous.BoundedRequestClient(api, out/"calls"/r["request_id"], r["max_calls"])
                text, trace = ENGINE.generate_one(bounded, s, r["operation"], r["seed"])
                row = export_followup(s, r["operation"], text, core.LABEL_IDS[core.target_label(s, r["operation"])],
                    core.OPERATION_SPECS[r["operation"]]["mr_type"], trace.get("applied_steps"),
                    "train" if r["dataset"] == "train" else "test") if text else None
                entries.append({"request_id": r["request_id"], "request": r, "trace": trace, "row": row})
                _save(out/"entries.json", entries); completed.add(r["request_id"]); checkpoint("running")
                print(f"refinement requests={len(entries)}/25 outputs={sum(bool(e['row']) for e in entries)}", flush=True)
                errors = errors + 1 if has_api_failure(trace) else 0
                if errors >= 3: state = "service_error"; raise RuntimeError("three consecutive API failures; stop")
            state = "requests_complete"
        finally: checkpoint(state)


def audit(path):
    plan, sources = validate(path); out = Path(plan["run_output"])
    entries = json.loads((out/"entries.json").read_text())
    status = json.loads((out/"status.json").read_text())
    if status["state"] != "requests_complete" or [e["request"] for e in entries] != plan["requests"]:
        raise ValueError("not terminal or request list changed")
    slots = list((out/"calls").glob("*/*.json")); before = {str(f): sha256(f) for f in slots}
    if len(slots) > 295: raise ValueError("call ceiling exceeded")
    # Every trace includes exact dispatch hashes; replay completed requests using read-only durable files.
    from nli_v3_4.calls import CALL_PROTOCOL
    class Cache:
        def __init__(self, directory): self.directory = directory; self.sequence = 0; self.chat = self.completions = self
        def create(self, **kwargs):
            from types import SimpleNamespace
            saved = json.loads((self.directory/f"{self.sequence:04d}.json").read_text())
            if (saved["protocol"] != CALL_PROTOCOL or saved["sequence"] != self.sequence
                    or saved["request_sha256"] != digest(kwargs) or saved["state"] != "completed"):
                raise RuntimeError("invalid/unavailable cached dispatch")
            self.sequence += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=saved["content"]))])
    total = 0
    for e in entries:
        r = e["request"]; s = next(s for s in sources[r["dataset"]] if s["pair_id"] == r["pair_id"])
        client = Cache(out/"calls"/r["request_id"])
        text, trace = ENGINE.generate_one(client, s, r["operation"], r["seed"])
        if (json.loads(json.dumps(trace)) != e["trace"] or bool(text) != bool(e["row"])
                or client.sequence > r["max_calls"] or client.sequence != len(list(client.directory.glob("*.json")))):
            raise ValueError("raw request replay differs")
        total += client.sequence
        if text and (ENGINE.replay_one(s, r["operation"], e["trace"], r["seed"]) != e["row"]["hypothesis"]):
            raise ValueError("accepted row did not replay")
    if total != len(slots) or any(sha256(f) != h for f, h in before.items()): raise ValueError("call ledger changed")
    hashes = {}; counts = {}
    for d, originals in sources.items():
        file = out/"v3_4"/f"{d}.jsonl"; rows, actual = load_dataset(file)
        if list(map(canonical_source, actual)) != originals: raise ValueError("original sources/golds changed")
        strip = lambda r: digest({k: v for k, v in r.items() if k != "idx"})
        if Counter(strip(r) for r in rows if not is_source(r)) != Counter(strip(e["row"]) for e in entries if e["request"]["dataset"] == d and e["row"]):
            raise ValueError("output differs from accepted journal")
        if d != "train" and [r["idx"] for r in rows] != list(range(len(rows))): raise ValueError("test idx changed")
        hashes[d] = sha256(file); counts[d] = {"sources": len(actual), "followups": len(rows)-len(actual)}
    result = {"stage": "refinement_terminal_integrity", "requests": 25, "calls": len(slots),
        "accepted": sum(bool(e["row"]) for e in entries), "by_origin": counts,
        "source_groups": sum(map(len, sources.values())), "all_raw_requests_replayed": True,
        "output_sha256": hashes, "entries_sha256": sha256(out/"entries.json"), "audit_api_calls": 0,
        "quality_passed": False, "production_allowed": False, "goal_complete": False}
    _save(out/"TERMINAL_AUDIT.json", result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--previous-review-key", type=Path, action="append", required=True); q.add_argument("--output", type=Path, required=True)
    for cmd in ("validate", "run", "audit"):
        q = sub.add_parser(cmd); q.add_argument("--plan", type=Path, required=True)
        if cmd == "run": q.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "prepare": prepare(a.manifest, a.previous_review_key, a.output); print('{"prepared":true,"api_calls":0}')
    elif a.command == "validate": validate(a.plan); print('{"valid":true,"api_calls":0}')
    elif a.command == "audit": print(json.dumps(audit(a.plan), ensure_ascii=False))
    else: run(a.plan, a.resume)


if __name__ == "__main__": main()
