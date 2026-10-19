"""Read-only, CPU-only diagnostics for NLI augmentation shortcuts.

Writes a reproducible report and metrics; never rewrites input datasets.
Probe models are diagnostic classifiers, not the project's NLI training runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import time
import unicodedata

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json"
DATASETS = ("snli", "mnlim", "mnlimm", "sick")
LABELS = ("entailment", "neutral", "contradiction")
NAMES = {
    "raw_word": "Hypothesis word TF-IDF",
    "raw_word_char": "Hypothesis word + char TF-IDF",
    "masked_word": "Cue-masked hypothesis word TF-IDF",
    "surface_only": "Surface cues + length",
    "original_only_word": "Original-only hypothesis word TF-IDF",
    "premise_only_word": "Premise word TF-IDF",
    "majority": "Training majority",
    "mr_only": "MR metadata majority (diagnostic upper bound)",
}
NEG = re.compile(r"\b(?:no|not|never|none|nobody|nothing|neither|without|cannot)\b|\b\w+n['’]t\b", re.I)
CUES = {
    "only_if": re.compile(r"\bonly\s+if\b", re.I),
    "comma_and": re.compile(r",\s*and\b", re.I),
    "comma_but": re.compile(r",\s*but\b", re.I),
    "any_but": re.compile(r"\bbut\b", re.I),
    "explicit_negation": NEG,
    "starts_negative": re.compile(r"^\s*(?:no\b|nobody\b|nothing\b|none\b)", re.I),
    "any_if": re.compile(r"\bif\b", re.I),
    "modal_uncertainty": re.compile(r"\b(?:may|might|maybe|perhaps|possibly)\b", re.I),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def read_rows(path, training=False):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            row = {k: value[k] for k in ("premise", "hypothesis", "label", "pair_id", "mr_id", "mr_type")}
            row["source"] = not value["is_syn"] if training else value["is_source"]
            assert row["label"] in (0, 1, 2)
            assert row["source"] == (row["mr_id"] == "none")
            assert row["premise"].strip() and row["hypothesis"].strip()
            row["component_mrs"] = value.get("component_mrs")
            rows.append(row)
    source = {r["pair_id"]: r for r in rows if r["source"]}
    assert len(source) == sum(r["source"] for r in rows)
    assert all(r["pair_id"] in source for r in rows)
    return rows


def ratio(a, b):
    return float(a / b) if b else None


def summarize_rows(rows):
    original = {r["pair_id"]: r for r in rows if r["source"]}
    followups = [r for r in rows if not r["source"]]
    exposed = {r["pair_id"] for r in followups}
    per_mr = {}
    for mr in sorted({r["mr_id"] for r in followups}):
        subset = [r for r in followups if r["mr_id"] == mr]
        counts = Counter(r["label"] for r in subset)
        per_mr[mr] = {"n": len(subset), "labels": {LABELS[k]: counts[k] for k in range(3)},
                      "dominant_label": LABELS[counts.most_common(1)[0][0]],
                      "dominant_label_rate": max(counts.values()) / len(subset)}
    triples = defaultdict(set)
    for r in rows:
        triples[(normalize(r["premise"]), normalize(r["hypothesis"]))].add(r["label"])
    return {
        "rows": len(rows), "originals": len(original), "followups": len(followups),
        "original_labels": dict(Counter(LABELS[r["label"]] for r in original.values())),
        "followup_labels": dict(Counter(LABELS[r["label"]] for r in followups)),
        "augmented_source_groups": len(exposed),
        "augmented_source_entailment_rate": ratio(sum(original[p]["label"] == 0 for p in exposed), len(exposed)),
        "followup_weighted_source_entailment_rate": ratio(sum(original[r["pair_id"]]["label"] == 0 for r in followups), len(followups)),
        "source_entailment_rate": ratio(sum(r["label"] == 0 for r in original.values()), len(original)),
        "unchanged_followups": sum(normalize(r["hypothesis"]) == normalize(original[r["pair_id"]]["hypothesis"]) for r in followups),
        "normalized_pair_label_conflicts": sum(len(labels) > 1 for labels in triples.values()),
        "per_mr": per_mr,
    }


def cue_report(rows):
    output = {}
    for region in ("all", "source", "followup"):
        subset = [r for r in rows if region == "all" or r["source"] == (region == "source")]
        output[region] = {}
        for name, pattern in CUES.items():
            hits = [r for r in subset if pattern.search(r["hypothesis"])]
            counts = Counter(r["label"] for r in hits)
            output[region][name] = {
                "n": len(hits), "coverage": ratio(len(hits), len(subset)),
                "labels": {LABELS[k]: counts[k] for k in range(3)},
                "dominant_label": LABELS[counts.most_common(1)[0][0]] if counts else None,
                "purity": ratio(max(counts.values()) if counts else 0, len(hits)),
            }
        for name, include_negation in (("connector_rule", False), ("connector_and_negation_rule", True)):
            predictions = []
            for r in subset:
                h = r["hypothesis"]
                pred = None
                if CUES["only_if"].search(h) or CUES["comma_and"].search(h):
                    pred = 1
                elif CUES["comma_but"].search(h) or (include_negation and (CUES["any_but"].search(h) or NEG.search(h))):
                    pred = 2
                if pred is not None:
                    predictions.append((r["label"], pred))
            output[region][name] = {
                "covered": len(predictions), "total": len(subset),
                "coverage": ratio(len(predictions), len(subset)),
                "accuracy_on_covered": ratio(sum(y == p for y, p in predictions), len(predictions)),
                "correct_fraction_of_all_rows": ratio(sum(y == p for y, p in predictions), len(subset)),
            }
    return output


def overlap_report(train, tests):
    train_pairs = {(r["premise"], r["hypothesis"]) for r in train}
    norm_pairs = {(normalize(p), normalize(h)) for p, h in train_pairs}
    train_p = {p for p, h in norm_pairs}
    train_h = {h for p, h in norm_pairs}
    # Exact token-set Jaccard candidate retrieval: a match at threshold t must
    # share at least one of the query's n-ceil(t*n)+1 rarest tokens. No candidate
    # limit or random subsampling is used; this is not a semantic similarity test.
    by_p = defaultdict(list)
    for i, row in enumerate(train):
        by_p[normalize(row["premise"])].append(i)
    premise_keys = list(by_p)
    psets = [set(p.split()) for p in premise_keys]
    hsets = [set(normalize(r["hypothesis"]).split()) for r in train]
    postings = defaultdict(set)
    for i, tokens in enumerate(psets):
        for token in tokens:
            postings[token].add(i)
    result = {}
    for dataset, rows in tests.items():
        cached = {}
        hits = {"0.8": [], "0.9": []}
        counts = Counter()
        for test_i, row in enumerate(rows):
            p, h = normalize(row["premise"]), normalize(row["hypothesis"])
            counts["exact_pair"] += (row["premise"], row["hypothesis"]) in train_pairs
            counts["normalized_pair"] += (p, h) in norm_pairs
            counts["swapped_normalized_pair"] += (h, p) in norm_pairs
            counts["normalized_premise_only"] += p in train_p
            counts["normalized_hypothesis_only"] += h in train_h
            if p not in cached:
                q = set(p.split())
                k = len(q) - math.ceil(0.8 * len(q)) + 1
                rare = sorted(q, key=lambda token: (len(postings.get(token, ())), token))[:k]
                candidates = set().union(*(postings.get(token, set()) for token in rare))
                matches = []
                for candidate in candidates:
                    tokens = psets[candidate]
                    if not 0.8 * len(q) <= len(tokens) <= len(q) / 0.8:
                        continue
                    similarity = len(q & tokens) / len(q | tokens)
                    if similarity >= 0.8:
                        matches.append((candidate, similarity))
                cached[p] = matches
            qh = set(h.split())
            best = 0.0
            best_i = None
            best_sims = None
            for candidate, psim in cached[p]:
                for train_i in by_p[premise_keys[candidate]]:
                    th = hsets[train_i]
                    hsim = len(qh & th) / len(qh | th)
                    score = min(psim, hsim)
                    if score > best:
                        best, best_i, best_sims = score, train_i, (psim, hsim)
            for threshold in (0.8, 0.9):
                if best >= threshold:
                    hits[str(threshold)].append({"test_row": test_i, "train_row": best_i,
                                                "premise_jaccard": best_sims[0], "hypothesis_jaccard": best_sims[1]})
        result[dataset] = {"rows": len(rows), **{key: int(counts[key]) for key in (
            "exact_pair", "normalized_pair", "swapped_normalized_pair", "normalized_premise_only", "normalized_hypothesis_only")},
            "joint_token_jaccard": {t: {"matching_test_rows": len(v), "examples": v[:10]} for t, v in hits.items()}}
        print("OVERLAP", dataset, json.dumps(result[dataset]), flush=True)
    return result


def masked(text):
    text = CUES["only_if"].sub(" ", text)
    text = CUES["comma_and"].sub(" ", text)
    text = CUES["comma_but"].sub(" ", text)
    text = CUES["any_but"].sub(" ", text)
    text = NEG.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def surfaces(rows):
    return np.asarray([[float(bool(p.search(r["hypothesis"]))) for p in CUES.values()] + [
        len(re.findall(r"\w+", r["hypothesis"])), len(r["hypothesis"]), r["hypothesis"].count(",")
    ] for r in rows], dtype=float)


def metrics(rows, pred):
    y = np.asarray([r["label"] for r in rows])
    output = {}
    for region in ("all", "source", "followup"):
        selection = np.asarray([region == "all" or r["source"] == (region == "source") for r in rows])
        output[region] = {"n": int(selection.sum()), "accuracy": float(accuracy_score(y[selection], pred[selection])),
                          "balanced_accuracy": float(balanced_accuracy_score(y[selection], pred[selection])),
                          "macro_f1": float(f1_score(y[selection], pred[selection], labels=[0, 1, 2], average="macro", zero_division=0)),
                          "confusion_matrix": confusion_matrix(y[selection], pred[selection], labels=[0, 1, 2]).tolist()}
    source_index = {r["pair_id"]: i for i, r in enumerate(rows) if r["source"]}
    follow_indices = np.asarray([i for i, r in enumerate(rows) if not r["source"]], dtype=int)
    source_indices = np.asarray([source_index[rows[i]["pair_id"]] for i in follow_indices], dtype=int)
    output["paired"] = {
        "edges": len(follow_indices),
        "source_accuracy_weighted_by_followups": float(np.mean(pred[source_indices] == y[source_indices])),
        "joint_edge_accuracy": float(np.mean((pred[source_indices] == y[source_indices]) & (pred[follow_indices] == y[follow_indices]))),
    }
    output["per_mr"] = {}
    for mr in sorted({r["mr_id"] for r in rows if not r["source"]}):
        indices = np.asarray([i for i, r in enumerate(rows) if r["mr_id"] == mr])
        output["per_mr"][mr] = {"n": len(indices), "accuracy": float(np.mean(pred[indices] == y[indices]))}
    return output


def word_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)


def model_for(name, seed):
    if name == "surface_only":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=seed))])
    features = word_vectorizer()
    if name == "raw_word_char":
        features = FeatureUnion([("word", features), ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000, sublinear_tf=True))])
    return Pipeline([("features", features), ("clf", LinearSVC(C=1.0, dual=True, max_iter=10000, random_state=seed))])


def model_input(rows, name):
    if name == "surface_only":
        return surfaces(rows)
    if name == "premise_only_word":
        return [r["premise"] for r in rows]
    if name == "masked_word":
        return [masked(r["hypothesis"]) for r in rows]
    return [r["hypothesis"] for r in rows]


def bootstrap_groups(rows, predictions, repetitions=1500):
    indices = [i for i, r in enumerate(rows) if not r["source"]]
    keys = sorted({rows[i]["pair_id"] for i in indices})
    key_index = {key: i for i, key in enumerate(keys)}
    group = np.asarray([key_index[rows[i]["pair_id"]] for i in indices])
    y = np.asarray([rows[i]["label"] for i in indices])
    correctness = np.mean([pred[indices] == y for pred in predictions], axis=0)
    correct = np.bincount(group, weights=correctness, minlength=len(keys))
    totals = np.bincount(group, minlength=len(keys))
    rng = np.random.default_rng(4242)
    estimates = []
    for start in range(0, repetitions, 100):
        sampled = rng.integers(0, len(keys), size=(min(100, repetitions-start), len(keys)))
        estimates.extend((correct[sampled].sum(axis=1) / totals[sampled].sum(axis=1)).tolist())
    return {"unit": "source pair_id within dataset", "replicates": repetitions,
            "statistic": "mean of three seed accuracies, micro-weighted followups",
            "groups": len(keys), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist()}


def aggregated(runs):
    output = {}
    for dataset in (*DATASETS, "train_holdout"):
        output[dataset] = {}
        for name in NAMES:
            samples = [run["evaluations"][dataset][name] for run in runs]
            item = {}
            for region in ("all", "source", "followup"):
                item[region] = {"n": samples[0][region]["n"] if dataset != "train_holdout" else [s[region]["n"] for s in samples]}
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    values = [s[region][metric] for s in samples]
                    item[region][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "seeds": values}
            item["paired"] = {key: float(np.mean([s["paired"][key] for s in samples])) for key in (
                "source_accuracy_weighted_by_followups", "joint_edge_accuracy")}
            item["per_mr"] = {mr: {"n": samples[0]["per_mr"][mr]["n"],
                "accuracy_mean": float(np.mean([s["per_mr"][mr]["accuracy"] for s in samples]))}
                for mr in samples[0]["per_mr"] if all(mr in s["per_mr"] for s in samples)}
            output[dataset][name] = item
    return output


def write_report(result, output):
    def pct(value):
        return f"{100 * value:.2f}%" if value is not None else "—"
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                         ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])
    agg, stats, cues = result["aggregate"], result["cohorts"], result["cues"]
    peak = [agg[d]["raw_word_char"]["followup"]["accuracy"]["mean"] for d in DATASETS]
    lines = ["# 增强训练集与 v1.1 测试集捷径诊断报告", "", "日期：2026-10-12。所有数字均为本轮重新计算，不沿用历史准确率。", "",
        f"**主要发现：hypothesis-only 的词与字符 TF-IDF 分类器，在四个测试集 follow-up 上达到 {pct(min(peak))}–{pct(max(peak))}。** 分类器完全没有读取 premise，也没有读取 MR、label、component_mrs 或 ensemble 审核信息作为输入。", "",
        "这说明当前数据存在明显的单句标签线索。文本重合检查与模板捷径是两类证据，不能将高单句准确率直接称为训练测试泄漏。", "",
        "## 1. 数据范围与可复现性", "",
        table(["数据", "原始样本", "follow-up", "合计"], [[name, s["originals"], s["followups"], s["rows"]] for name, s in stats.items()]), "",
        "训练输入是完整 v3.3 的 5,340 行数据；测试输入是本机导入的四个 v1.1 JSONL。各训练划分由原始 pair_id 分组，源行与全部 follow-up 共同进入同一侧。不同数据集的 pair_id 命名空间分别处理。", "",
        "每个 seed 使用 80% 原始 pair groups 拟合，20% 留作训练域诊断验证；按原始标签分层。seed=42/43/44。四个正式测试集只用于评估，未用于选择特征、模型超参数或阈值。", "",
        "TF-IDF 的词表与 IDF 仅在每个 seed 的训练部分拟合。词特征为 unigram/bigram，min_df=2，max_features=30,000；字符特征为 char_wb 3–5 gram，max_features=60,000。LinearSVC C=1；表层探针使用标准化的 11 个数值特征与 LogisticRegression C=1。", "",
        "这是轻量 CPU 诊断分类器拟合，不是 Gemma/LoRA 训练，也未调用 LLM 或 GPU。原有输入、标签与提示词均未修改。", "",
        "输入路径与 SHA-256：", "",
        table(["数据", "路径", "SHA-256"], [[k, f"`{v['path']}`", f"`{v['sha256']}`"] for k, v in result["inputs"].items()]), "",
        "## 2. 文本重合与近重复", "",
        table(["测试集", "精确 P/H", "规范化 P/H", "P/H 对调", "双侧 Jaccard ≥0.9", "双侧 Jaccard ≥0.8", "仅 H 规范化重合"],
              [[d, o["exact_pair"], o["normalized_pair"], o["swapped_normalized_pair"], o["joint_token_jaccard"]["0.9"]["matching_test_rows"], o["joint_token_jaccard"]["0.8"]["matching_test_rows"], o["normalized_hypothesis_only"]] for d, o in result["overlap"].items()]), "",
        "规范化使用 Unicode NFKC、大小写折叠和词 token。近重复要求同一个训练行与测试行的 premise、hypothesis 各自 token-set Jaccard 都达到阈值；使用完整候选检索，无随机抽样或候选数量截断。它是词面重复筛查，不是穷尽语义污染证明。仅 H 重合可能来自常见短句，不能单独认定完整样本泄漏。", "",
        "## 3. 固定连接词的标签绑定", "",
        table(["数据", "线索", "follow-up 命中数", "覆盖率", "占优标签", "标签纯度"],
              [[d, c, cues[d]["followup"][c]["n"], pct(cues[d]["followup"][c]["coverage"]), cues[d]["followup"][c]["dominant_label"], pct(cues[d]["followup"][c]["purity"])]
               for d in ("train", *DATASETS) for c in ("only_if", "comma_and", "comma_but", "explicit_negation")]), "",
        "生成器 `augment_snli_all_label_mrs.py` 的 build_generation_prompt 强制固定前缀和连接词：uninformative 使用 `, and`，adding_contradiction 使用 `, but`，conditional_clause 使用 `only if`。生成测试适配器直接导入该模块。当前测试文本也呈现同样关联；由于导入目录缺少完整生成日志，本报告不声称已认证所有测试行的历史生成命令。", "",
        "预定义规则：先将 only-if / comma-and 判为 neutral，再将 comma-but 判为 contradiction；扩展规则在此前基础上将普通 but / 显式否定判为 contradiction。其余行弃权，规则没有依据测试结果调整。", "",
        table(["数据", "三连接词覆盖", "覆盖部分准确率", "加否定覆盖", "覆盖部分准确率", "扩展规则正确行 / 全部 follow-up"],
              [[d, pct(cues[d]["followup"]["connector_rule"]["coverage"]), pct(cues[d]["followup"]["connector_rule"]["accuracy_on_covered"]),
                pct(cues[d]["followup"]["connector_and_negation_rule"]["coverage"]), pct(cues[d]["followup"]["connector_and_negation_rule"]["accuracy_on_covered"]), pct(cues[d]["followup"]["connector_and_negation_rule"]["correct_fraction_of_all_rows"])] for d in ("train", *DATASETS)]), "",
        "覆盖部分准确率是条件准确率，不能写成整个测试集准确率。未覆盖行不计为预测正确。否定本身可以是合法语义操作；风险来自它与目标标签的单侧绑定，而不是出现否定词就必然有问题。", "",
        "## 4. 单句探针与消融", "",
        table(["探针：follow-up accuracy（三 seed 均值）", *DATASETS],
              [[NAMES[name], *[pct(agg[d][name]["followup"]["accuracy"]["mean"]) for d in DATASETS]] for name in NAMES]), "",
        table(["测试集", "词+字符 H accuracy ± seed SD", "按 pair-group bootstrap 95% CI", "balanced accuracy", "macro-F1"],
              [[d, f"{pct(agg[d]['raw_word_char']['followup']['accuracy']['mean'])} ± {pct(agg[d]['raw_word_char']['followup']['accuracy']['std'])}",
                f"{pct(result['bootstrap'][d]['ci95'][0])}–{pct(result['bootstrap'][d]['ci95'][1])}", pct(agg[d]["raw_word_char"]["followup"]["balanced_accuracy"]["mean"]), f"{agg[d]['raw_word_char']['followup']['macro_f1']['mean']:.4f}"] for d in DATASETS]), "",
        "bootstrap 按有 follow-up 的源 pair groups 整组抽样，1,500 次，统计量为三 seed follow-up micro accuracy 的均值。seed SD 描述分组拟合随机性，bootstrap 区间描述当前测试源组抽样不确定性，均不覆盖生成器或语义标签错误的不确定性。", "",
        "原始样本训练探针使用与增强探针相同的 80% 源组，但仅拟合源行，因此样本数更少；它用于衡量增强暴露是否带来额外单句标签可预测性，不能当作严格预算匹配的增强效果比较。", "",
        "MR metadata majority 只根据训练部分的 MR ID 预测占优标签，是结构诊断上界；普通 NLI 测试未必提供 MR ID，不能把这个值称为真实测试标签泄漏。", "",
        "遮蔽探针删除 only-if、comma-and、comma-but、but 和显式否定。raw_word 与 masked_word 的特征和模型参数相同，可直接比较。这个处理会破坏部分语义与语法，仅用于诊断，不生成新 NLI 数据，也不能将准确率下降当作安全消除捷径的证据。", "",
        table(["测试集", "原始词 H", "遮蔽词 H", "下降（百分点）", "仅表层特征"],
              [[d, pct(agg[d]["raw_word"]["followup"]["accuracy"]["mean"]), pct(agg[d]["masked_word"]["followup"]["accuracy"]["mean"]),
                f"{100*(agg[d]['raw_word']['followup']['accuracy']['mean']-agg[d]['masked_word']['followup']['accuracy']['mean']):.2f}", pct(agg[d]["surface_only"]["followup"]["accuracy"]["mean"])] for d in DATASETS]), "",
        "## 5. 原始测试与配对联合正确率", "",
        table(["测试集", "H 词+字符：全部原始行", "follow-up", "对应源行（按 follow-up 加权）", "source-follow-up 联合正确率"],
              [[d, pct(agg[d]["raw_word_char"]["source"]["accuracy"]["mean"]), pct(agg[d]["raw_word_char"]["followup"]["accuracy"]["mean"]),
                pct(agg[d]["raw_word_char"]["paired"]["source_accuracy_weighted_by_followups"]), pct(agg[d]["raw_word_char"]["paired"]["joint_edge_accuracy"])] for d in DATASETS]), "",
        "联合指标以每条 source-follow-up 边为单位，只有两端预测均正确才计为正确；一个源有多个 follow-up 会产生多条边。本指标不是所有 follow-up 必须同时正确的整组指标，也不是项目正式 MR/MSR 指标的重计算。", "",
        "对于 neutral MR，若 MSR 只检验 follow-up 是否为 neutral，固定连接词可让 MSR 很高而无需判断源样本。应并列报告原始 NLI、逐标签 balanced accuracy、逐 MR 指标和配对联合正确率，避免单一 follow-up/MSR 掩盖这种差异。", "",
        "## 6. 源样本选择与 MR 配比", "",
        table(["数据", "原始源 E 占比", "有 follow-up 的源组数", "这些源组 E 占比", "按 follow-up 加权的源 E 占比"],
              [[d, pct(s["source_entailment_rate"]), s["augmented_source_groups"], pct(s["augmented_source_entailment_rate"]), pct(s["followup_weighted_source_entailment_rate"])] for d, s in stats.items()]), "",
        "flip/neutral 操作在当前合同下多作用于 entailment 源，INV 在接受过滤后也可能偏向 E。这属于实际选择偏差；不能用跨 relation-family 或改标签的方法人为抹平。", "",
        table(["MR", "训练 follow-up 数", "训练占优标签/纯度", *[f"{d} H词+字符 accuracy (n)" for d in DATASETS]],
              [[mr, info["n"], f"{info['dominant_label']} / {pct(info['dominant_label_rate'])}",
                *[f"{pct(agg[d]['raw_word_char']['per_mr'][mr]['accuracy_mean'])} ({agg[d]['raw_word_char']['per_mr'][mr]['n']})" for d in DATASETS]] for mr, info in stats["train"]["per_mr"].items()]), "",
        "稀少 MR 的准确率波动很大，不宜用 2–13 条测试/训练样本作稳健结论。逐 MR 高准确率也可能来自固定目标标签与模板，而非具体变换理解。", "",
        "## 7. 降低捷径的建议（本轮未实施）", "",
        "1. **优先处理固定模板与标签绑定。** 单纯找不同主题或删除近重复不会处理 only-if/comma-and/comma-but 的绑定。对允许表达变体的 MR，增加结构与连接形式多样性，并保留独立操作有效性与 NLI 标签审核。对于当前定义明确要求 only-if 的 conditional_clause，改变该格式需要单独评审并升级合同版本，不能偷偷改成其他 MR 或只删词。", "",
        "2. **保持测试独立性。** 已用这四集诊断过生成问题，后续若据此调整，应将它们作为诊断基准，另冻结未参与调优的源样本/模板组合用于确认评估。训练和测试的实现形式可以区分，但 MR 的操作、relation-family 和组合顺序必须保持明确一致。", "",
        "3. **考虑合法反例与源标签覆盖。** 同一表面线索在不同上下文中应能对应多个标签；这些反例必须具有真实语义依据。可补充独立注明来源的 ordinary-NLI 控制样本，或提高既有 INV 中 C/N 源的有效覆盖。不能强行给固定 E→N/E→C 的 MR 分配其他标签，也不能靠无条件提高拒绝率获得漂亮数字。", "",
        "4. **诊断与质量同步设门槛。** 比较相同源组、训练预算和标签分布下的 hypothesis-only raw/masked、表层探针、原始测试、逐标签和联合指标；同时报告语义审核、MR fidelity、各 MR 接受率与覆盖。hypothesis-only 下降而语义/覆盖恶化不能算成功。", "",
        "5. **保留当前版本。** 新生成数据使用独立目录、新合同版本与完整哈希/日志；保留 v3.3 训练和 v1.1 测试作为固定基准。先做小规模代表性试验，审阅之后再考虑全量重生成。", "",
        "目前证据可支持“单句表面线索很强、与生成合同一致”。它不能单独证明正式 Gemma 模型具体依赖了哪条捷径，也不能证明全部标签在语义上正确。", "",
        "## 8. 输出与复现", "", "- `metrics.json`：数据统计、线索纯度、重合、逐 seed 与逐 MR 指标、混淆矩阵及 bootstrap。",
        "- `splits.json`：三 seed 的训练/验证源组，供复现并核对分组隔离。",
        "- `shortcut_accuracy.png` / `.pdf`：诊断概览图。",
        "- `audit_nli_shortcuts.py`：完整计算脚本位于上一级目录。", "",
        "```bash", "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\",
        "  /home/ubuntu/anaconda3/bin/python analysis/audit_nli_shortcuts.py \\",
        "  --output analysis/shortcut_audit_new_run", "```", "",
        f"环境：Python {result['environment']['python']}；scikit-learn {result['environment']['sklearn']}；NumPy {result['environment']['numpy']}。",
        f"输入哈希复核：{result['inputs_unchanged']}。耗时：{result['elapsed_seconds']:.1f} 秒。", ""]
    (output / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = result["aggregate"]
    x = np.arange(len(DATASETS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for name, label in (("raw_word_char", "Hypothesis: word + char"), ("surface_only", "Surface cues + length"),
                        ("masked_word", "Hypothesis: cues removed"), ("original_only_word", "Hypothesis: original-only train"),
                        ("majority", "Train majority")):
        y = [100*agg[d][name]["followup"]["accuracy"]["mean"] for d in DATASETS]
        axes[0].plot(x, y, marker="o", label=label)
    axes[0].set_title("Follow-up accuracy without premise reasoning")
    axes[0].legend(fontsize=8, loc="lower left")
    for i, (region, label) in enumerate((("source", "Original source"), ("followup", "Follow-up"), ("joint", "Joint source + follow-up"))):
        y = [100*(agg[d]["raw_word_char"]["paired"]["joint_edge_accuracy"] if region == "joint" else
                  agg[d]["raw_word_char"][region]["accuracy"]["mean"]) for d in DATASETS]
        axes[1].bar(x+(i-1)*0.24, y, width=0.24, label=label)
    axes[1].set_title("Hypothesis-only word + char probe")
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_xticks(x, ["SNLI", "MNLI-m", "MNLI-mm", "SICK"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "shortcut_accuracy.png", dpi=180)
    fig.savefig(output / "shortcut_accuracy.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/shortcut_audit_20261012")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("choose a new output directory; existing audit is preserved")
    started = time.time()
    paths = {"train": TRAIN, **{d: ROOT / "mr_test_data_merged_v1.1" / f"{d}.jsonl" for d in DATASETS}}
    inputs = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in paths.items()}
    train = read_rows(TRAIN, training=True)
    tests = {d: read_rows(paths[d]) for d in DATASETS}
    cohorts = {name: summarize_rows(rows) for name, rows in {"train": train, **tests}.items()}
    result = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": inputs, "cohorts": cohorts, "cues": {name: cue_report(rows) for name, rows in {"train": train, **tests}.items()},
              "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": sha256(__file__), "seeds": [42, 43, 44], "model_definitions": NAMES, "runs": []}
    print("COHORTS", json.dumps({name: {k: s[k] for k in ('rows','originals','followups')} for name, s in cohorts.items()}), flush=True)
    result["overlap"] = overlap_report(train, tests)
    save_json(output / "metrics.json", result)
    original = [r for r in train if r["source"]]
    pair_ids = [r["pair_id"] for r in original]
    source_labels = [r["label"] for r in original]
    predictions = {d: [] for d in DATASETS}
    splits = {}
    for seed in result["seeds"]:
        fitting, heldout = train_test_split(pair_ids, test_size=0.2, stratify=source_labels, random_state=seed)
        fitting_set, heldout_set = set(fitting), set(heldout)
        assert not fitting_set & heldout_set
        fitting_rows = [r for r in train if r["pair_id"] in fitting_set]
        heldout_rows = [r for r in train if r["pair_id"] in heldout_set]
        evaluations = {**tests, "train_holdout": heldout_rows}
        run = {"seed": seed, "fit_source_groups": len(fitting), "fit_rows": len(fitting_rows),
               "holdout_source_groups": len(heldout), "evaluations": {d: {} for d in evaluations}}
        splits[str(seed)] = {"training_pair_ids": sorted(fitting), "holdout_pair_ids": sorted(heldout)}
        majority = Counter(r["label"] for r in fitting_rows).most_common(1)[0][0]
        label_by_mr = {mr: Counter(r["label"] for r in fitting_rows if r["mr_id"] == mr).most_common(1)[0][0]
                       for mr in {r["mr_id"] for r in fitting_rows}}
        for name in NAMES:
            if name not in ("majority", "mr_only"):
                rows_for_fit = [r for r in fitting_rows if r["source"]] if name == "original_only_word" else fitting_rows
                model = model_for(name, seed)
                model.fit(model_input(rows_for_fit, name), np.asarray([r["label"] for r in rows_for_fit]))
            for dataset, rows in evaluations.items():
                if name == "majority":
                    pred = np.full(len(rows), majority, dtype=int)
                elif name == "mr_only":
                    pred = np.asarray([label_by_mr.get(r["mr_id"], majority) for r in rows])
                else:
                    pred = model.predict(model_input(rows, name))
                run["evaluations"][dataset][name] = metrics(rows, pred)
                if name == "raw_word_char" and dataset in DATASETS:
                    predictions[dataset].append(pred)
            print("PROBE", seed, name, {d: round(run['evaluations'][d][name]['followup']['accuracy'], 5) for d in DATASETS}, flush=True)
        result["runs"].append(run)
        save_json(output / "metrics.json", result)
        save_json(output / "splits.json", splits)
    result["aggregate"] = aggregated(result["runs"])
    result["bootstrap"] = {d: bootstrap_groups(tests[d], predictions[d]) for d in DATASETS}
    result["inputs_unchanged"] = all(sha256(paths[d]) == inputs[d]["sha256"] for d in paths)
    assert result["inputs_unchanged"], "an input changed during the audit"
    result["elapsed_seconds"] = time.time() - started
    result["status"] = "complete"
    save_json(output / "metrics.json", result)
    write_report(result, output)
    plot_summary(result, output)
    print("AUDIT_COMPLETE", output, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
