"""Frozen hypothesis-only diagnostic: train features on the actual training file only."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from nli_v3_4.data import LABELS, DATASETS, audit, is_source, load_dataset, sha256, write_json, write_jsonl


CONFIG = {"ngram_range": [1, 2], "min_df": 2, "max_features": 30000,
          "sublinear_tf": True, "C": 1.0, "solver": "lbfgs", "class_weight": None,
          "max_iter": 3000, "random_state": 42, "tol": 0.0001}


def fit_probe(rows):
    # The ONLY feature input is hypothesis; metadata is used after prediction for reports.
    for iterations in (3000, 6000, 12000):
        probe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                max_features=30000, sublinear_tf=True)),
            ("lr", LogisticRegression(C=1, solver="lbfgs", class_weight=None,
                max_iter=iterations, tol=0.0001, random_state=42))])
        with warnings.catch_warnings(record=True) as seen, threadpool_limits(limits=2):
            warnings.simplefilter("always", ConvergenceWarning)
            probe.fit([r["hypothesis"] for r in rows], [r["label"] for r in rows])
        if not any(issubclass(w.category, ConvergenceWarning) for w in seen):
            return probe, iterations
    raise RuntimeError("logistic regression did not converge; do not publish an underfit result")


def bootstrap(rows, correct, repeats=1500):
    groups = {}
    for row, value in zip(rows, correct):
        counts = groups.setdefault(row["pair_id"], [0, 0])
        counts[0] += int(value)
        counts[1] += 1
    counts = np.asarray(list(groups.values()), dtype=np.int64)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(repeats):
        sampled = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        values.append(float(sampled[0] / sampled[1]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def evaluate(probe, rows, training_majority, exclude_ids=()):
    follows = [r for r in rows if not is_source(r) and r["pair_id"] not in exclude_ids]
    if not follows:
        raise ValueError("no followups in evaluation cohort")
    y = np.asarray([r["label"] for r in follows])
    pred = probe.predict([r["hypothesis"] for r in follows])
    counts = Counter(y.tolist())
    score = float(accuracy_score(y, pred))
    return {"followups": len(follows), "source_groups": len({r["pair_id"] for r in follows}),
            "accuracy": score, "passes_60_percent": score <= 0.6,
            "ci95_source_group_bootstrap": bootstrap(follows, pred == y),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "confusion_matrix_E_N_C": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "followup_labels": dict(counts), "test_majority_baseline": max(counts.values()) / len(y),
            "train_majority_baseline": float(np.mean(y == training_majority)),
            "per_mr": {mr: {"n": sum(r["mr_id"] == mr for r in follows),
                "accuracy": float(np.mean([a == b for r, a, b in zip(follows, y, pred) if r["mr_id"] == mr]))}
                for mr in sorted({r["mr_id"] for r in follows})}}, follows, pred


def run(train, tests, output, development_manifest=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic run: {output}")
    training, _ = load_dataset(train)
    initial = {str(p.resolve()): sha256(p) for p in [Path(train), *tests.values()]}
    probe, iterations = fit_probe(training)
    majority = Counter(r["label"] for r in training).most_common(1)[0][0]
    result = {"stage": "diagnostic_only", "config": CONFIG, "actual_max_iter": iterations,
              "sklearn_version": sklearn.__version__, "train": str(Path(train).resolve()),
              "input_sha256": initial, "training_audit": audit(training), "evaluations": {}}
    exclusions = {}
    if development_manifest:
        import json
        manifest = json.loads(Path(development_manifest).read_text())
        exclusions = {d: set(v["development_pair_ids"]) for d, v in manifest["datasets"].items()}
    predictions = []
    for name, path in tests.items():
        rows, _ = load_dataset(path)
        metrics, follows, pred = evaluate(probe, rows, majority)
        result["evaluations"][name] = {"full": metrics, "audit": audit(rows)}
        if name in exclusions:
            result["evaluations"][name]["nondevelopment"] = evaluate(probe, rows, majority, exclusions[name])[0]
        predictions.extend({"dataset": name, "pair_id": r["pair_id"], "row_index": r.get("idx"),
                            "mr_id": r["mr_id"], "label": r["label"], "prediction": int(p)}
                           for r, p in zip(follows, pred))
        print(name, f"followup accuracy={metrics['accuracy']:.4f}", flush=True)
    result["all_datasets_pass"] = all(v["full"]["passes_60_percent"] for v in result["evaluations"].values())
    result["inputs_unchanged"] = all(sha256(p) == h for p, h in initial.items())
    if not result["inputs_unchanged"]:
        raise RuntimeError("input changed during diagnostic")
    top = {}
    words = probe.named_steps["tfidf"].get_feature_names_out()
    for label, coef in zip(probe.named_steps["lr"].classes_, probe.named_steps["lr"].coef_):
        top[LABELS[int(label)]] = [{"term": words[i], "weight": float(coef[i])} for i in np.argsort(coef)[-40:][::-1]]
    result["top_positive_terms"] = top
    write_json(output / "metrics.json", result)
    write_jsonl(output / "predictions.jsonl", predictions)
    lines = ["# Hypothesis-only TF-IDF＋逻辑回归诊断", "", "完整训练文件拟合；输入仅 hypothesis，词表和 IDF 不接触测试数据。", "",
             "|测试来源|增强样本|Accuracy|95% CI|多数标签基线|≤60%|", "|---|---:|---:|---|---:|---|"]
    for name, v in result["evaluations"].items():
        m=v["full"]; lo,hi=m["ci95_source_group_bootstrap"]
        lines.append(f"|{name}|{m['followups']}|{m['accuracy']:.2%}|{lo:.2%}–{hi:.2%}|{m['test_majority_baseline']:.2%}|{'通过' if m['passes_60_percent'] else '未通过'}|")
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT_zh.md").write_text("\n".join(lines)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path)
    args = parser.parse_args()
    run(args.train, {d: args.test_root / f"{d}.jsonl" for d in DATASETS}, args.output, args.development_manifest)


if __name__ == "__main__":
    main()
