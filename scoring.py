"""
scoring.py
----------
本模块提供以下功能：
1. `nli_entailment_score`：计算前提句和假设句之间的 NLI entailment 分数。
2. `compute_score`：结合语义相似度和 NLI entailment 分数，计算综合得分。
3. `select_top_half`：从候选句子中选择得分最高的一半。

依赖：
- transformers
- sentence_transformers
- torch
- numpy

示例用法见 main 函数。
"""

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import List, Dict

# 初始化模型
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI")
nli_model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/roberta-base-MNLI").to(device)
nli_model.eval()


@torch.no_grad()
def nli_entailment_score(premise: str, hypothesis: str) -> float:
    """
    计算前提句和假设句之间的 NLI entailment 分数。

    参数：
    - premise (str): 前提句。
    - hypothesis (str): 假设句。

    返回：
    - float: entailment 分数（entailment + neutral 的概率和）。
    """
    inputs = tokenizer(premise,
                       hypothesis,
                       return_tensors="pt",
                       truncation=True).to(device)
    logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    return probs[0] + probs[1]  # entailment + neutral


def compute_score(sample: Dict[str, str]) -> float:
    """
    计算样本的综合得分，结合语义相似度和 NLI entailment 分数。

    参数：
    - sample (Dict[str, str]): 包含 'premise' 和 'hypothesis' 的字典。

    返回：
    - float: 综合得分，范围 [0, 1]。
    """
    # 计算语义相似度
    emb1 = sbert_model.encode(sample["premise"], convert_to_tensor=True)
    emb2 = sbert_model.encode(sample["hypothesis"], convert_to_tensor=True)
    sim_score = util.cos_sim(emb1, emb2).item()

    # 计算 NLI entailment 分数
    entail_score = nli_entailment_score(sample["premise"],
                                        sample["hypothesis"])

    # 综合得分
    final_score = (sim_score + entail_score) / 2
    return final_score


def select_top_half(data: List[Dict[str, object]]) -> List[Dict[str, str]]:
    """
    从候选句子中选择得分最高的一半。

    参数：
    - data (List[Dict[str, object]]): 包含 'original' 和 'candidates' 的数据列表。
      每个 'candidates' 是一个包含 'premise' 和 'hypothesis' 的字典列表。

    返回：
    - List[Dict[str, str]]: 得分最高的一半候选句子。
    """
    filtered = []
    for entry in data:
        original = entry["original"]
        candidates = entry["candidates"]

        # 计算每个候选句子的得分
        scores = [compute_score(c) for c in candidates]
        best = candidates[np.argmax(scores)]
        best["score"] = max(scores)
        filtered.append(best)

    # 计算得分中位数，选择得分高于中位数的句子
    threshold = np.median([x["score"] for x in filtered])
    top_half = [x for x in filtered if x["score"] >= threshold]

    # 移除临时的 'score' 字段
    for x in top_half:
        x.pop("score", None)
    return top_half


if __name__ == "__main__":
    # 测试代码
    print("=== 测试 scoring.py ===")

    # 测试 nli_entailment_score
    premise = "A man is playing a guitar."
    hypothesis = "A person is playing music."
    entailment_score = nli_entailment_score(premise, hypothesis)
    print(f"NLI entailment score: {entailment_score:.4f}")

    # 测试 compute_score
    sample = {
        "premise": "A man is playing a guitar.",
        "hypothesis": "A person is playing music."
    }
    score = compute_score(sample)
    print(f"Computed score: {score:.4f}")

    # 测试 select_top_half
    data = [{
        "original":
        "A man is playing a guitar.",
        "candidates": [{
            "premise": "A man is playing a guitar.",
            "hypothesis": "A person is playing music."
        }, {
            "premise": "A man is playing a guitar.",
            "hypothesis": "A dog is barking."
        }, {
            "premise": "A man is playing a guitar.",
            "hypothesis": "A woman is cooking."
        }]
    }]
    top_half = select_top_half(data)
    print("Top half candidates:")
    for item in top_half:
        print(item)
