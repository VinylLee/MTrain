"""OFFLINE DEVELOPMENT ONLY: inspect prompt errors, never filter generation output.

Accepts only journalled development runs belonging to the frozen 600-source manifest.
Writes probabilities/flags separately; it has no accepted/rejected export or relabel API.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nli_v3_4.data import canonical_source, digest, is_source, load_dataset, sha256, write_json, write_jsonl

MODELS = {
    "deberta": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"),
    "roberta": ("FacebookAI/roberta-large-mnli", "2a8f12d27941090092df78e4ba6f0928eb5eac98"),
}


def development_rows(manifest_path, dataset, generated_path):
    manifest = json.loads(Path(manifest_path).read_text())
    metadata = json.loads(Path(generated_path).with_suffix(".run.json").read_text())
    if metadata.get("stage") != "development" or metadata.get("dataset") != dataset:
        raise ValueError("NLI diagnostics are restricted to development runs")
    if metadata.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("development manifest mismatch")
    rows, sources = load_dataset(generated_path)
    if digest([canonical_source(r) for r in sources]) != metadata.get("source_sha256"):
        raise ValueError("generated sources differ from the development run")
    ids = set(manifest["datasets"][dataset]["development_pair_ids"])
    if any(r["pair_id"] not in ids for r in rows):
        raise ValueError("nondevelopment source in input; do not inspect confirmation data")
    original = {r["pair_id"]: r for r in sources}
    for r in rows:
        if is_source(r):
            continue
        yield r, original[r["pair_id"]]


def addition_contexts(dataset, generated_path, journal_path, entries, protocol_python=None, manifest_path=None):
    """Bind optional round-4 fact diagnostics to accepted raw traces, never select rows."""
    if protocol_python is not None:
        # The fixed Torch environment need not install the generation API SDK.
        # Revalidate raw journals in the explicitly chosen protocol interpreter;
        # this child mode imports no Torch and never constructs an API client.
        result=subprocess.check_output([str(protocol_python),str(Path(__file__).resolve()),
            "--manifest",str(manifest_path),"--dataset",dataset,"--generated",str(generated_path),
            "--addition-journal",str(journal_path),"--journal-contexts-only"],text=True)
        value=json.loads(result)
        if value["generated_sha256"]!=sha256(generated_path) or value["journal_sha256"]!=sha256(journal_path):
            raise ValueError("protocol child inspected different inputs")
        return value["contexts"]
    from nli_v3_4.redesign import Round4Engine, ADDITIONS
    metadata=json.loads(Path(generated_path).with_suffix(".run.json").read_text())
    engine=Round4Engine()
    if metadata.get("engine")=="refinement":
        from nli_v3_4.refinement import RefinementEngine
        engine=RefinementEngine()
    elif metadata.get("engine")=="refinement_v2":
        from nli_v3_4.refinement_v2 import DirectTypedEngine
        engine=DirectTypedEngine()
    elif metadata.get("engine")=="refinement_v2_article":
        from nli_v3_4.article_guard import ArticleCheckedEngine
        engine=ArticleCheckedEngine()
    if (metadata.get("stage")!="development" or metadata.get("engine") not in {"round4","refinement","refinement_v2","refinement_v2_article"}
            or metadata.get("contract_sha256")!=engine.fingerprint()):
        raise ValueError("addition diagnostics require a bound round-4 development run")
    journal=json.loads(Path(journal_path).read_text());contexts={}
    by_key={digest({k:v for k,v in row.items() if k!="idx"}):(row,source) for row,source in entries}
    for item in journal:
        request=item.get("request",{});row=item.get("row")
        if request.get("dataset")!=dataset or not row or request.get("operation") not in ADDITIONS:continue
        key=digest({k:v for k,v in row.items() if k!="idx"})
        if key not in by_key or key in contexts:raise ValueError("addition journal/output mismatch")
        actual,source=by_key[key];trace=item["trace"]
        if engine.replay_one(canonical_source(source),request["operation"],trace,request["seed"])!=actual["hypothesis"]:
            raise ValueError("cannot replay addition trace")
        selected=[p for a in trace["attempts"] if a.get("status")=="accepted"
                  for p in a["proposals"] if p["hypothesis"]==actual["hypothesis"]]
        if len(selected)!=1:raise ValueError("ambiguous added clause in trace")
        contexts[key]=selected[0]["added_clause"]
    for row,_ in entries:
        if row["mr_id"] in ADDITIONS and digest({k:v for k,v in row.items() if k!="idx"}) not in contexts:
            raise ValueError("missing addition trace; do not invent fact diagnostics")
    return contexts


def pick_device(requested):
    import torch
    if requested == "cpu" or not torch.cuda.is_available():
        return "cpu"
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    free = [(int(i), int(mem)) for line in result.splitlines() for i, mem in [line.split(",")]]
    eligible = [(i, mem) for i, mem in free if mem >= 8192]
    if not eligible:
        return "cpu"
    return f"cuda:{max(eligible, key=lambda item: item[1])[0]}"


def score_tasks(model, tokenizer, tasks, device, batch_size):
    import torch
    scores = []
    mapping = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if set(mapping.values()) != {"entailment", "neutral", "contradiction"}:
        raise ValueError(f"unsupported label mapping {mapping}")
    limit = min(int(getattr(model.config, "max_position_embeddings", 512)),
                int(tokenizer.model_max_length), 512)
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        valid, valid_indices = [], []
        partial = [None] * len(batch)
        for index, (_, left, right) in enumerate(batch):
            encoded = tokenizer(left, right, truncation=False)
            if len(encoded["input_ids"]) > limit:
                partial[index] = {"status": "too_long", "tokens": len(encoded["input_ids"])}
            else:
                valid.append(encoded); valid_indices.append(index)
        if valid:
            encoded = tokenizer.pad(valid, return_tensors="pt").to(device)
            with torch.inference_mode():
                probabilities = model(**encoded).logits.float().softmax(-1).cpu().tolist()
            for index, probs in zip(valid_indices, probabilities):
                p = {mapping[k]: value for k, value in enumerate(probs)}
                partial[index] = {"status": "scored", "probabilities": p,
                                  "prediction": max(p, key=p.get)}
        scores.extend(partial)
    return scores


def run(manifest_path, dataset, generated_path, output, cache, requested_device="auto", batch_size=8, addition_journal=None, protocol_python=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(2)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = sha256(generated_path)
    journal_before=sha256(addition_journal) if addition_journal else None
    entries = list(development_rows(manifest_path, dataset, generated_path))
    contexts=addition_contexts(dataset,generated_path,addition_journal,entries,protocol_python,manifest_path) if addition_journal else {}
    tasks = []
    rows = []
    layouts=[]
    for index, (row, source) in enumerate(entries):
        record = {"row_number": index, "pair_id": row["pair_id"], "mr_id": row["mr_id"],
            "expected_label": ("entailment", "neutral", "contradiction")[row["label"]],
            "premise": row["premise"], "hypothesis": row["hypothesis"],
            "source_hypothesis": source["hypothesis"], "scores": {}}
        tasks.append((index, row["premise"], row["hypothesis"]))
        layout=["pair"]
        if row["mr_type"] == "inv":
            tasks.extend([(index, source["hypothesis"], row["hypothesis"]),
                          (index, row["hypothesis"], source["hypothesis"])])
            layout.extend(("inv_forward","inv_reverse"))
        key=digest({k:v for k,v in row.items() if k!="idx"})
        if key in contexts:
            added=contexts[key];record.update(added_clause=added,source_gold_label=("entailment","neutral","contradiction")[source["label"]])
            tasks.extend([(index,row["premise"],source["hypothesis"]),
                          (index,row["premise"],added),(index,source["hypothesis"],added)])
            layout.extend(("source_pair","premise_to_added","source_to_added"))
        rows.append(record)
        layouts.append(layout)
    device = pick_device(requested_device)
    models_meta = {}
    for name, (model_id, revision) in MODELS.items():
        snapshot = Path(cache) / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True,
                    use_safetensors=True).to(device).eval()
        results = score_tasks(model, tokenizer, tasks, device, batch_size)
        cursor = 0
        for index, (row, _) in enumerate(entries):
            n = len(layouts[index])
            rows[index]["scores"][name] = dict(zip(layouts[index], results[cursor:cursor+n]))
            cursor += n
        models_meta[name] = {"model_id": model_id, "revision": revision, "device": device,
                            "id2label": model.config.id2label,
                            "weights_sha256": sha256(snapshot/"model.safetensors"),
                            "config_sha256": sha256(snapshot/"config.json")}
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    for row in rows:
        flags = []
        predictions = []
        for name, tests in row["scores"].items():
            for test, score in tests.items():
                if score["status"] != "scored":
                    flags.append(f"{name}:{test}:too_long")
                    continue
                expected = (row["expected_label"] if test in {"pair","premise_to_added"} else
                            row["source_gold_label"] if test=="source_pair" else
                            "neutral" if test=="source_to_added" else "entailment")
                if score["probabilities"][expected] < 0.8:
                    flags.append(f"{name}:{test}:low_target_probability")
                if test == "pair":
                    predictions.append(score["prediction"])
        if len(set(predictions)) > 1:
            flags.append("pair:model_disagreement")
        row["design_diagnostic_flags"] = flags
    if sha256(generated_path) != before:
        raise RuntimeError("input changed during design diagnostic")
    if addition_journal and sha256(addition_journal)!=journal_before:
        raise RuntimeError("addition journal changed during design diagnostic")
    write_jsonl(output / "design_scores.jsonl", rows)
    write_json(output / "report.json", {"stage": "development_diagnostic_only", "dataset": dataset,
        "input": str(Path(generated_path).resolve()), "input_sha256": before,
        "generation_metadata_sha256": sha256(Path(generated_path).with_suffix(".run.json")),
        "manifest_sha256": sha256(manifest_path), "models": models_meta, "rows": len(rows),
        "flagged_rows": sum(bool(r["design_diagnostic_flags"]) for r in rows),
        "addition_journal_sha256":journal_before,
        "protocol_verification_interpreter":str(protocol_python) if protocol_python else None,
        "model_input_pairs":len(tasks)*len(MODELS),
        "input_unchanged": True, "used_to_filter_data": False, "production_dependency": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--cache", type=Path, default=Path.home()/".cache/huggingface/hub")
    p.add_argument("--device", choices=("auto", "cpu"), default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--addition-journal",type=Path,help="Optional round-4 entries.json; diagnose P/A and H/A without filtering")
    p.add_argument("--protocol-python",type=Path,help="Optional interpreter with generation protocol dependencies for read-only journal replay")
    p.add_argument("--journal-contexts-only",action="store_true",help="Verify journal and print contexts, without Torch or inference")
    a = p.parse_args()
    if a.journal_contexts_only:
        if not a.addition_journal:p.error("journal context inspection requires --addition-journal")
        entries=list(development_rows(a.manifest,a.dataset,a.generated))
        contexts=addition_contexts(a.dataset,a.generated,a.addition_journal,entries)
        print(json.dumps({"contexts":contexts,"generated_sha256":sha256(a.generated),"journal_sha256":sha256(a.addition_journal)}))
    else:
        if not a.output:p.error("inference requires --output")
        run(a.manifest, a.dataset, a.generated, a.output, a.cache, a.device, a.batch_size,a.addition_journal,a.protocol_python)


if __name__ == "__main__":
    main()
