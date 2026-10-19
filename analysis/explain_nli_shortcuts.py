"""Replay frozen audit splits to explain probes and bootstrap the surface probe."""

from collections import Counter
from pathlib import Path
import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import audit_nli_shortcuts as audit


def write_explanations_report(metrics, result, output):
    def pct(x):
        return f"{100*x:.2f}%"
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                         ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])
    agg = metrics["aggregate"]
    lines = ["# 补充诊断：表层探针、长度与剩余词线索", "", "日期：2026-10-12。复用主报告的冻结源组划分。重放三 seed 表层探针及 seed42 raw/masked 词探针，所有四集的准确率均与主报告逐 seed 指标完全一致。", "",
        "## 表层探针的不确定性与类别平衡", "",
        table(["数据", "表层 accuracy", "源组 bootstrap 95% CI", "balanced accuracy", "多数类基线", "高于基线（百分点）"], [
            [d, pct(agg[d]["surface_only"]["followup"]["accuracy"]["mean"]),
             "–".join(pct(v) for v in result["surface_bootstrap"][d]["ci95"]),
             pct(agg[d]["surface_only"]["followup"]["balanced_accuracy"]["mean"]), pct(agg[d]["majority"]["followup"]["accuracy"]["mean"]),
             f"{100*(agg[d]['surface_only']['followup']['accuracy']['mean']-agg[d]['majority']['followup']['accuracy']['mean']):.2f}"] for d in audit.DATASETS]), "",
        "该探针只有 8 个词面布尔特征和 3 个长度/标点计数。高于多数类基线的幅度为当前数据的表层标签可预测性证据，不能解释成正式 LLM 中有同样比例的预测因果上依赖捷径。", "",
        "## follow-up 长度与目标标签", "",
        table(["数据", "entailment 平均/中位词数", "neutral 平均/中位词数", "contradiction 平均/中位词数"], [
            [d, *[f"{result['length_by_followup_label'][d][label]['mean']:.2f} / {result['length_by_followup_label'][d][label]['median']:.1f}" for label in audit.LABELS]]
            for d in ("train", *audit.DATASETS)]), "",
        "训练增强中 E 平均约 5.67 词，N 约 13.02 词，C 约 8.25 词。增补整段 clause 的 MR 与局部 INV 编辑的长度差异，也可被单句分类器利用。以后平衡长度需要在真实语义与 MR 合同允许范围内实现，不能删掉决定标签的细节或机械截断。", "",
        "## 词探针的高权重特征（seed42）", "",
        table(["探针", "标签", "最高的 8 个正权重 unigram/bigram"], [
            [name, label, ", ".join("`"+r["feature"]+"`" for r in data[:8])]
            for name, labels in result["word_seed42_top_features"].items() for label, data in labels.items()]), "",
        "raw_word 的 N 特征主要是 and/if，C 特征主要是 no/not/nobody/but，与预定义线索检测一致。masked_word 中 does/do 等特征也可能来自删除否定后的语法残留；不能把它们全部认定为原始生成器中新发现的捷径。线性系数反映相关性，完整权重及训练行标签计数见 explanatory_probes.json。", "",
        "## 结构检查中的待复核样本", "",
        "本轮规范化 P/H 标签冲突在训练和四集均为 0；但有两条 follow-up 的规范化 hypothesis 与其源相同：", "",
        table(["数据", "pair_id", "MR", "source hypothesis", "follow-up hypothesis"], [
            [d, r["pair_id"], r["mr_id"], r["source_hypothesis"], r["followup_hypothesis"]]
            for d, rows in result["unchanged_followup_examples"].items() for r in rows]), "",
        "这两条应人工复核操作有效性；本轮未删除或改写它们。标签自洽、审核元数据与原始/增强文本的操作有效性需要分别判断。", "",
        "补充计算未读取 premise 作为单句探针输入，未改动任何输入文件。", ""]
    (output / "EXPLANATIONS_zh.md").write_text("\n".join(lines))


def main():
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=audit.ROOT / "analysis/shortcut_audit_20261012")
    output = parser.parse_args().output.resolve()
    metrics = json.loads((output / "metrics.json").read_text())
    splits = json.loads((output / "splits.json").read_text())
    train = audit.read_rows(audit.TRAIN, training=True)
    tests = {d: audit.read_rows(audit.ROOT / metrics["inputs"][d]["path"]) for d in audit.DATASETS}
    result = {"script_sha256": audit.sha256(__file__), "frozen_inputs": metrics["inputs"],
              "surface_feature_names": [*audit.CUES, "hypothesis_word_count", "hypothesis_character_count", "hypothesis_comma_count"],
              "surface_bootstrap": {}, "surface_seed42_coefficients": {}, "word_seed42_top_features": {},
              "length_by_followup_label": {}, "unchanged_followup_examples": {}, "replay_matches_primary_metrics": True}
    predictions = {d: [] for d in audit.DATASETS}
    for seed in metrics["seeds"]:
        fitting = set(splits[str(seed)]["training_pair_ids"])
        rows = [r for r in train if r["pair_id"] in fitting]
        model = audit.model_for("surface_only", seed)
        model.fit(audit.surfaces(rows), [r["label"] for r in rows])
        for dataset, examples in tests.items():
            pred = model.predict(audit.surfaces(examples))
            predictions[dataset].append(pred)
            expected = next(run for run in metrics["runs"] if run["seed"] == seed)["evaluations"][dataset]["surface_only"]["followup"]["accuracy"]
            follow = [i for i, r in enumerate(examples) if not r["source"]]
            actual = np.mean(pred[follow] == np.asarray([examples[i]["label"] for i in follow]))
            assert abs(actual - expected) < 1e-12, (seed, dataset)
        if seed == 42:
            classifier = model.named_steps["clf"]
            for label, coefficients in zip(classifier.classes_, classifier.coef_):
                result["surface_seed42_coefficients"][audit.LABELS[label]] = {
                    "intercept": float(classifier.intercept_[list(classifier.classes_).index(label)]),
                    "standardized_feature_weights": dict(zip(result["surface_feature_names"], coefficients.tolist()))}
    for dataset, rows in tests.items():
        result["surface_bootstrap"][dataset] = audit.bootstrap_groups(rows, predictions[dataset])
    fitting = set(splits["42"]["training_pair_ids"])
    rows = [r for r in train if r["pair_id"] in fitting]
    for name in ("raw_word", "masked_word"):
        model = audit.model_for(name, 42)
        text = audit.model_input(rows, name)
        model.fit(text, [r["label"] for r in rows])
        vocabulary = model.named_steps["features"].get_feature_names_out()
        analyzer = model.named_steps["features"].build_analyzer()
        counts = {feature: Counter() for feature in vocabulary}
        for row, value in zip(rows, text):
            for feature in set(analyzer(value)):
                if feature in counts:
                    counts[feature][audit.LABELS[row["label"]]] += 1
        result["word_seed42_top_features"][name] = {}
        for label, coefficients in zip(model.named_steps["clf"].classes_, model.named_steps["clf"].coef_):
            top = np.argsort(coefficients)[-12:][::-1]
            result["word_seed42_top_features"][name][audit.LABELS[label]] = [
                {"feature": vocabulary[i], "weight": float(coefficients[i]), "training_row_label_counts": dict(counts[vocabulary[i]])} for i in top]
        for dataset, examples in tests.items():
            pred = model.predict(audit.model_input(examples, name))
            follow = [i for i, r in enumerate(examples) if not r["source"]]
            expected = metrics["runs"][0]["evaluations"][dataset][name]["followup"]["accuracy"]
            assert abs(np.mean(pred[follow] == np.asarray([examples[i]["label"] for i in follow])) - expected) < 1e-12
    for dataset, rows in {"train": train, **tests}.items():
        result["length_by_followup_label"][dataset] = {}
        for label in range(3):
            lengths = [len(audit.normalize(r["hypothesis"]).split()) for r in rows if not r["source"] and r["label"] == label]
            result["length_by_followup_label"][dataset][audit.LABELS[label]] = {
                "n": len(lengths), "mean": float(np.mean(lengths)), "median": float(np.median(lengths))}
        sources = {r["pair_id"]: r for r in rows if r["source"]}
        result["unchanged_followup_examples"][dataset] = [
            {"pair_id": r["pair_id"], "mr_id": r["mr_id"], "source_hypothesis": sources[r["pair_id"]]["hypothesis"],
             "followup_hypothesis": r["hypothesis"], "label": audit.LABELS[r["label"]]}
            for r in rows if not r["source"] and audit.normalize(r["hypothesis"]) == audit.normalize(sources[r["pair_id"]]["hypothesis"])]
    result["inputs_unchanged"] = all(audit.sha256(audit.ROOT / v["path"]) == v["sha256"] for v in metrics["inputs"].values())
    assert result["inputs_unchanged"]
    result["elapsed_seconds"] = time.time() - started
    audit.save_json(output / "explanatory_probes.json", result)
    write_explanations_report(metrics, result, output)
    print("EXPLANATORY_PROBES_COMPLETE", result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
