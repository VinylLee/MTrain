"""Local manifests, balanced scheduling and resumable journals for both export formats."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import random
import time

from . import core
from .calls import CALL_PROTOCOL, DurableRequestClient, validate_budget_reservation
from .schemas import GENERATION_FORMAT
from .protocol import PROTOCOL_VERSION, has_api_failure
from .data import (DATASETS, LABELS, audit, canonical_source, digest, export_followup,
                   export_source, is_source, load_dataset, normalize, quotas, read_jsonl,
                   sha256, source_uid, write_json, write_jsonl)


def prepare(train, test_root, output, per_label=40, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = {"train": Path(train), **{d: Path(test_root) / f"{d}.jsonl" for d in DATASETS}}
    manifest = {"stage": "development", "seed": seed, "per_label": per_label,
                "production_external_nli": False, "remote_push": False, "datasets": {}}
    for name, path in paths.items():
        rows, originals = load_dataset(path)
        buckets = {label: [r for r in originals if r["label"] == label] for label in range(3)}
        selected = []
        for label, bucket in buckets.items():
            if len(bucket) < per_label:
                raise ValueError(f"not enough label {label} sources in {name}")
            selected.extend(random.Random(seed + label).sample(bucket, per_label))
        selected.sort(key=lambda r: r["pair_id"])
        ids = {r["pair_id"] for r in selected}
        selected_old = [r for r in rows if r["pair_id"] in ids]
        sources = [canonical_source(r) for r in originals]
        write_jsonl(output / "sources" / f"{name}.jsonl", sources)
        write_jsonl(output / "development_sources" / f"{name}.jsonl", [canonical_source(r) for r in selected])
        write_jsonl(output / "development_old" / f"{name}.jsonl", selected_old)
        write_jsonl(output / "baseline_full" / f"{name}.jsonl", rows)
        manifest["datasets"][name] = {"input": str(path.resolve()), "sha256": sha256(path),
            "schema": "train" if name == "train" else "test",
            "source_count": len(originals), "target_followups": len(rows) - len(originals),
            "development_pair_ids": sorted(ids), "development_source_uids": [source_uid(r) for r in selected],
            "development_original_followup_labels": dict(Counter(r["label"] for r in selected_old if not is_source(r))),
            "audit": audit(rows)}
    manifest["manifest_fingerprint"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "contract_initial.json", core.contract())
    return manifest


def _validate_manifest(path):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("manifest_fingerprint")
    if digest(manifest) != expected:
        raise ValueError("development manifest was modified")
    manifest["manifest_fingerprint"] = expected
    return manifest


def source_rows(manifest_path, dataset, stage):
    manifest = _validate_manifest(manifest_path)
    info = manifest["datasets"][dataset]
    if sha256(info["input"]) != info["sha256"]:
        raise ValueError("frozen input changed")
    rows, sources = load_dataset(info["input"])
    if stage == "development":
        ids = set(info["development_pair_ids"])
        sources = [s for s in sources if s["pair_id"] in ids]
    return manifest, info, [canonical_source(r) for r in sources]


def original_quotas(counts, total):
    counts = {int(k): v for k, v in counts.items()}
    if not sum(counts.values()):
        raise ValueError("empty old followup cohort")
    targets = {label: total * counts.get(label, 0) / sum(counts.values()) for label in range(3)}
    result = {label: int(n) for label, n in targets.items()}
    remainder = total - sum(result.values())
    order = sorted(result, key=lambda l: (-(targets[l] - result[l]), l))
    for label in order[:remainder]:
        result[label] += 1
    return result


def scheduling_queues(sources, seed, refill_round, engine):
    # Scheduling depends on labels and MR applicability, never on train/test role.
    ordered = list(sources)
    random.Random(seed + refill_round).shuffle(ordered)
    result = {label: {} for label in range(3)}
    for operation in engine.OPERATION_SPECS:
        for source in ordered:
            if engine.eligible(source, operation):
                label = engine.LABEL_IDS[engine.target_label(source, operation)]
                result[label].setdefault(operation, deque()).append(source)
    return result


def _errors(value):
    if isinstance(value, dict):
        return ([value["error"]] if "error" in value else []) + sum((_errors(v) for v in value.values()), [])
    if isinstance(value, list):
        return sum((_errors(v) for v in value), [])
    return []


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {p: sha256(root / p) for p in ("nli_v3_4/core.py", "nli_v3_4/data.py", "nli_v3_4/workflow.py", "nli_v3_4/protocol.py", "nli_v3_4/recovery.py", "nli_v3_4/invariants.py",
                "nli_v3_4/processes.py", "nli_v3_4/calls.py", "nli_v3_4/schemas.py", "nli_v3_4/realization.py", "run_nli_v3_4.py", "run_nli_v3_4_development.py",
                "augment_snli_all_label_mrs.py", "provenance_schema.py")}


def run_generation(manifest_path, dataset, output, *, stage="development", version="v3.4",
                   target=None, quota_policy="balanced", seed=42, base_url="http://127.0.0.1:1234/v1",
                   timeout=120, max_refill_rounds=4, limit_requests=None, resume=False, freeze_path=None):
    from openai import OpenAI
    import augment_snli_all_label_mrs as legacy
    output = Path(output)
    if version not in {"v3.3", "v3.4"} or stage not in {"development", "production"}:
        raise ValueError("invalid stage/version")
    if stage == "production":
        if version != "v3.4" or quota_policy != "balanced" or not freeze_path:
            raise ValueError("production requires frozen balanced v3.4")
        frozen = json.loads(Path(freeze_path).read_text())
        if frozen.get("contract_sha256") != core.fingerprint() or frozen.get("code_hashes") != _code_hashes():
            raise ValueError("production contract/code differs from frozen development version")
        if not frozen.get("quality_passed") or not frozen.get("development_shortcut_passed"):
            raise ValueError("development gates have not passed")
        if frozen.get("manifest_sha256") != sha256(manifest_path):
            raise ValueError("production source cohort differs from development manifest")
    manifest, info, sources = source_rows(manifest_path, dataset, stage)
    target = (180 if stage == "development" else info["target_followups"]) if target is None else target
    if target < 3 or (stage == "production" and target != info["target_followups"]):
        raise ValueError("invalid or changed production target")
    if max_refill_rounds != 4:
        raise ValueError("four refill rounds are fixed for comparable runs")
    wanted = quotas(target) if quota_policy == "balanced" else original_quotas(info["development_original_followup_labels"], target)
    engine = core if version == "v3.4" else legacy
    # Keep legacy ablations at their old allocation so the contribution of this
    # declared distribution change can be reported separately from wording.
    inv_minimum = {l:(wanted[l] if l==0 else int(math.ceil(wanted[l]*core.MINIMUM_INV_FRACTION)))
                   for l in range(3)} if version=="v3.4" else {l:0 for l in range(3)}
    run_config = {"version": version, "stage": stage, "dataset": dataset, "seed": seed,
        "manifest_sha256": sha256(manifest_path), "source_sha256": digest(sources),
        "contract_sha256": core.fingerprint() if version == "v3.4" else legacy.config_fingerprint(
            legacy.PROMPT_VERSION, legacy.OPERATION_SPECS, legacy.COMPOSITE_STAGE_PLANS),
        "code_hashes": _code_hashes(), "target_followups": target, "quota_policy": quota_policy,
        "label_quotas": wanted, "candidate_count": 3, "rounds": 2,
        "max_refill_rounds": 4, "model": core.ALLOWED_MODEL,
        "scheduling_policy": "balance_attempts_within_target_label_and_MR_v2",
        "minimum_invariant_counts":inv_minimum,
        "base_url": base_url, "timeout": timeout, "external_nli_in_pipeline": False,
        "response_protocol": PROTOCOL_VERSION, "call_protocol": CALL_PROTOCOL,
        "generation_response_format": GENERATION_FORMAT if version == "v3.4" else "legacy_unconstrained"}
    meta = output.with_suffix(".run.json")
    journal = output.with_suffix(".generation_log.jsonl")
    report = output.with_suffix(".report.json")
    if resume:
        if not meta.exists() or not journal.exists():
            raise FileNotFoundError("resume needs metadata and journal")
        stored = json.loads(meta.read_text())
        if digest(stored) != digest(run_config):
            raise ValueError("resume configuration changed")
        logs = read_jsonl(journal)
    else:
        if any(p.exists() for p in (output, meta, journal, report)):
            raise FileExistsError("run already exists; use --resume with identical settings")
        write_json(meta, run_config)
        journal.touch()
        logs = []
    accepted = [log["row"] for log in logs if log.get("row")]
    counts = Counter(r["label"] for r in accepted)
    mr_counts = Counter(r["mr_id"] for r in accepted)
    inv_counts = Counter(r["label"] for r in accepted if engine.OPERATION_SPECS[r["mr_id"]]["family"]=="inv")
    processed = {log["request_id"] for log in logs if not log.get("pending_continuation")}
    log_indices = {log["request_id"]: i for i, log in enumerate(logs)}
    pending_ids = {log["request_id"] for log in logs if log.get("pending_continuation")}
    seen = {(normalize(s["premise"]), normalize(s["hypothesis"])) for s in sources}
    seen.update((normalize(r["premise"]), normalize(r["hypothesis"])) for r in accepted)
    source_by_id = {s["pair_id"]: s for s in sources}
    if len(log_indices) != len(logs) or any(counts[l] > wanted[l] for l in range(3)):
        raise ValueError("duplicate journal request or exceeded quota")
    if any(counts[l]-inv_counts[l]>wanted[l]-inv_minimum[l] for l in range(3)):
        raise ValueError("journal exceeds the declared non-invariant allocation")
    for entry in logs:
        source = source_by_id.get(entry.get("pair_id"))
        op = entry.get("operation")
        if source is None or op not in engine.OPERATION_SPECS:
            raise ValueError("unknown journal source/operation")
        if entry["request_id"] != digest([source_uid(source), op, entry["refill_round"], seed, version]):
            raise ValueError("journal request identity mismatch")
        if "budget_reservation" in entry:
            validate_budget_reservation(entry, source, seed, version)
        row = entry.get("row")
        if row and (row["premise"] != source["premise"] or row["label"] !=
                    engine.LABEL_IDS[engine.target_label(source, op)] or row["mr_id"] != op):
            raise ValueError("journal row differs from requested MR/source")
    schema = info["schema"]

    def checkpoint(state):
        grouped = defaultdict(list)
        for r in accepted:
            grouped[r["pair_id"]].append(r)
        rows = []
        for source in sources:
            rows.append(export_source(source, schema))
            rows.extend(grouped[source["pair_id"]])
        if schema == "test":
            for idx, row in enumerate(rows):
                row["idx"] = idx
        write_jsonl(output, rows)
        quotas_met = all(counts[l] == wanted[l] and inv_counts[l]>=inv_minimum[l] for l in range(3))
        call_counts = Counter()
        def count_calls(trace):
            if isinstance(trace, dict):
                call_counts.update(e["role"] for e in trace.get("model_call_events", []))
                for key, value in trace.items():
                    if key not in {"model_call_events", "prior_trace"}:
                        count_calls(value)
            elif isinstance(trace, list):
                for value in trace: count_calls(value)
        for entry in logs: count_calls(entry["trace"])
        source_label_family = Counter(f"{engine.OPERATION_SPECS[r['mr_id']]['family']}/source_{source_by_id[r['pair_id']]['label']}/target_{r['label']}" for r in accepted)
        write_json(report, {**run_config, "state": state, "generation_complete": quotas_met,
            "overall_complete": False, "quality_status": "awaiting_human_review",
            "sources": len(sources), "followups": len(accepted), "requests": len(logs),
            "pending_continuations": sum(bool(e.get("pending_continuation")) for e in logs),
            "label_counts": dict(counts), "label_deficits": {l: wanted[l]-counts[l] for l in range(3)},
            "invariant_label_counts":dict(inv_counts),"invariant_deficits":{l:max(0,inv_minimum[l]-inv_counts[l]) for l in range(3)},
            "mr_counts": dict(mr_counts), "output_sha256": sha256(output),
            "model_call_counts":dict(call_counts), "source_label_family_counts":dict(source_label_family),
            "journal_sha256": sha256(journal), "external_nli_in_pipeline": False})

    checkpoint("prepared" if not logs else "resuming")
    if all(counts[l] == wanted[l] for l in range(3)):
        checkpoint("generation_complete")
        return json.loads(report.read_text())
    client = OpenAI(api_key="local", base_url=base_url, timeout=timeout, max_retries=0)
    # A listing alone is insufficient. Fail before costly batches on unavailable service.
    try:
        legacy._completion(client, core.ALLOWED_MODEL, "Reply with OK only.", "OK", 0.0, 8)
    except Exception:
        checkpoint("service_unavailable")
        raise
    consecutive_errors = 0
    new_requests = 0
    # An INV MR is applicable to all source labels. Attempts spent filling the
    # harder entailment quota must not suppress N/C INV requests in other queues.
    attempts_by_label_mr = Counter((log["target_label"],log["operation"]) for log in logs)
    accepted_by_label_mr = Counter((r["label"],r["mr_id"]) for r in accepted)
    attempts_by_source = Counter(log["pair_id"] for log in logs)
    stop = False
    final_state = "partial"
    try:
        with journal.open("a") as handle:
            for refill_round in range(max_refill_rounds):
                queues = scheduling_queues(sources, seed, refill_round, engine)
                while True:
                    def permitted_ops(l):
                        return [op for op,q in queues[l].items() if q and
                            (engine.OPERATION_SPECS[op]["family"]=="inv" or counts[l]-inv_counts[l]<wanted[l]-inv_minimum[l])]
                    labels = [l for l in range(3) if counts[l] < wanted[l] and permitted_ops(l)]
                    if not labels:
                        break
                    label = min(labels, key=lambda l: (counts[l]/max(1,wanted[l]), l))
                    def is_pending(s, op):
                        return bool(pending_ids) and digest([source_uid(s), op, refill_round, seed, version]) in pending_ids
                    operations=permitted_ops(label)
                    invariant_ops=[op for op in operations if engine.OPERATION_SPECS[op]["family"]=="inv"]
                    if inv_counts[label]<inv_minimum[label] and invariant_ops:operations=invariant_ops
                    operation = min(operations,
                                    key=lambda op: (not any(is_pending(s, op) for s in queues[label][op]),
                                                    attempts_by_label_mr[(label,op)], accepted_by_label_mr[(label,op)], op))
                    queue = queues[label][operation]
                    source = min(queue, key=lambda s: (not is_pending(s, operation), attempts_by_source[s["pair_id"]]))
                    queue.remove(source)
                    request_id = digest([source_uid(source), operation, refill_round, seed, version])
                    if request_id in processed:
                        continue
                    if limit_requests is not None and new_requests >= limit_requests:
                        stop = True
                        break
                    prior_entry = logs[log_indices[request_id]] if request_id in log_indices else None
                    prior_trace = prior_entry["trace"] if prior_entry and prior_entry.get("pending_continuation") else None
                    # A continuation starts at its saved trace; a crashed fresh
                    # request restarts at call zero and replays its durable responses.
                    origin = digest(prior_trace) if prior_trace else "fresh"
                    call_directory = output.with_suffix(".calls") / request_id / origin
                    request_client = DurableRequestClient(client, call_directory)
                    if version == "v3.4":
                        text, trace = (core.generate_one(request_client, source, operation, seed + refill_round, prior_trace)
                                       if prior_trace else core.generate_one(request_client, source, operation, seed + refill_round))
                    else:
                        text, trace = core.generate_legacy_bounded(request_client, source, operation, seed + refill_round, prior_trace)
                    entry = {"request_id": request_id, "pair_id": source["pair_id"],
                        "source_uid": source_uid(source), "operation": operation, "refill_round": refill_round,
                        "target_label": label, "trace": trace, "row": None,
                        "call_ledger": str(call_directory.resolve()),
                        "call_replay": {"new_calls": request_client.new_calls,
                                        "replayed_calls": request_client.replayed_calls,
                                        "uncertain_calls": request_client.uncertain_calls}}
                    if text:
                        key = (normalize(source["premise"]), normalize(text))
                        if key not in seen:
                            steps = trace.get("applied_steps")
                            if operation in core.COMPOSITE_STAGE_PLANS and (
                                not isinstance(steps, list) or tuple(steps) not in core.COMPOSITE_STAGE_PLANS[operation]):
                                raise ValueError("missing/invalid composite order")
                            row = export_followup(source, operation, text, label,
                                engine.OPERATION_SPECS[operation]["mr_type"], steps, schema)
                            entry["row"] = row
                            accepted.append(row); seen.add(key); counts[label] += 1; mr_counts[operation] += 1
                            accepted_by_label_mr[(label,operation)] += 1
                            if engine.OPERATION_SPECS[operation]["family"]=="inv":inv_counts[label]+=1
                        else:
                            entry["discard_reason"] = "duplicate_or_unchanged"
                    if prior_entry:
                        entry["prior_trace"] = prior_entry["trace"]
                        entry["prior_execution_code_hashes"] = prior_entry.get("execution_code_hashes")
                        logs[log_indices[request_id]] = entry
                        handle.close()
                        write_jsonl(journal, logs)
                        handle = journal.open("a")
                    else:
                        handle.write(json.dumps(entry, ensure_ascii=False)+"\n"); handle.flush()
                        log_indices[request_id] = len(logs); logs.append(entry)
                        attempts_by_label_mr[(label,operation)] += 1
                        attempts_by_source[source["pair_id"]] += 1
                    processed.add(request_id); new_requests += 1
                    pending_ids.discard(request_id)
                    consecutive_errors = consecutive_errors + 1 if text is None and has_api_failure(trace) else 0
                    if new_requests % 10 == 0:
                        checkpoint("running")
                        print(dataset, version, "requests", len(processed), "accepted", dict(counts), flush=True)
                    if consecutive_errors >= 3:
                        final_state = "service_error"
                        raise RuntimeError("three consecutive request errors; preserve journal and stop")
                if stop or all(counts[l] == wanted[l] for l in range(3)):
                    break
    finally:
        if 'handle' in locals():
            handle.close()
        checkpoint("generation_complete" if all(counts[l] == wanted[l] for l in range(3)) else final_state)
    return json.loads(report.read_text())


def freeze(manifest_path, quality_path, shortcut_path, output, round_number):
    if not 1 <= round_number <= 3:
        raise ValueError("at most three development rounds")
    quality = json.loads(Path(quality_path).read_text())
    shortcut = json.loads(Path(shortcut_path).read_text())
    manifest = _validate_manifest(manifest_path)
    if manifest.get("per_label") != 40 or any(len(v["development_pair_ids"]) != 120
                                             for v in manifest["datasets"].values()):
        raise ValueError("freeze requires the approved 600-source development cohort")
    if set(quality.get("datasets", {})) != {"train", *DATASETS}:
        raise ValueError("human evidence must cover all five data sources")
    if quality.get("stage") != "development" or quality.get("contract_sha256") != core.fingerprint():
        raise ValueError("quality evidence is not for this development contract")
    if not quality.get("human_review_complete") or not quality.get("quality_passed"):
        raise ValueError("real human quality review has not passed")
    if not shortcut.get("all_datasets_pass") or set(shortcut.get("evaluations", {})) != set(DATASETS):
        raise ValueError("development shortcut gate has not passed on all four sources")
    from analysis.hypothesis_only_logreg import CONFIG
    if shortcut.get("config") != CONFIG or not shortcut.get("inputs_unchanged"):
        raise ValueError("shortcut evidence uses a different diagnostic or changed inputs")
    # Bind evidence to this exact generated cohort, not a similarly named result.
    for dataset, evidence in quality["datasets"].items():
        path = evidence["v3_4_path"]
        if (shortcut["input_sha256"].get(str(Path(path).resolve())) != sha256(path)
                or evidence.get("v3_4_sha256") != sha256(path)):
            raise ValueError("quality and shortcut evidence refer to different files")
        report = json.loads(Path(path).with_suffix(".report.json").read_text())
        if not report.get("generation_complete") or report.get("stage") != "development":
            raise ValueError("cannot freeze an incomplete development generation")
        if (report.get("code_hashes") != _code_hashes() or report.get("contract_sha256") != core.fingerprint()
                or report.get("target_followups") != 180 or report.get("quota_policy") != "balanced"
                or report.get("manifest_sha256") != sha256(manifest_path)):
            raise ValueError("development files have different code/configuration/cohort identities")
    if Path(output).exists():
        raise FileExistsError("freeze already exists")
    write_json(output, {"stage": "frozen", "round": round_number,
        "contract_sha256": core.fingerprint(), "contract": core.contract(), "code_hashes": _code_hashes(),
        "manifest_sha256": sha256(manifest_path), "quality_sha256": sha256(quality_path),
        "shortcut_sha256": sha256(shortcut_path), "quality_passed": True,
        "development_shortcut_passed": True, "external_nli_in_production": False,
        "remote_push": False})
