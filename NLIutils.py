"""
conda activate LibreTranslate3.9
libretranslate

"""
import json
import os
import random
from typing import Optional
import numpy as np

import requests
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def translate_libre(source: str, direction: str):
    """
    使用LibreTranslate服务翻译单个文本
    
    Args:
        source: 输入文本（字符串）
        direction: 翻译方向，格式为"源语言2目标语言"，如"en2zh"、"zh2en"等
        
    Returns:
        tuple: (翻译后的文本, 备选翻译列表)
        - 第一个元素是主要翻译结果
        - 第二个元素是备选翻译列表（如果有的话）
    """
    # 设置LibreTranslate API的URL
    url = "http://127.0.0.1:5000/translate"

    # 解析翻译方向，例如"en2zh"拆分为源语言"en"和目标语言"zh"
    source_lang, target_lang = direction.split('2')

    # 准备请求参数
    payload = {
        "q": source,  # 要翻译的文本
        "source": source_lang,  # 源语言代码
        "target": target_lang,  # 目标语言代码
        "format": "text",  # 文本格式
        "alternatives": 3,  # 请求3个备选翻译
        "api_key": ""  # API密钥，这里为空
    }

    # 设置请求头
    headers = {"Content-Type": "application/json"}

    # 发送POST请求到翻译API
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # 解析JSON响应
    result = json.loads(response.text)

    # 提取主要翻译结果
    main_translation = result["translatedText"]

    # 提取备选翻译（如果存在）
    alternative_translations = result.get("alternatives", [])

    # 返回主要翻译和备选翻译
    return main_translation, alternative_translations


def back_translate(text, directions=("en2zh", "zh2en")):
    """
    将文本通过指定的翻译方向进行多次回译
    
    Args:
        text: 输入文本字符串
        directions: 翻译方向的元组，例如 ("en2zh", "zh2en", "en2fr", ...)
        
    Returns:
        包含所有回译结果的列表
    """
    if not isinstance(text, str):
        raise ValueError("输入文本必须是字符串")

    # 起始只有一个原文本
    translations = [text]

    for direction in directions:
        next_translations = []

        for item in translations:
            # 获取主要翻译和备选翻译
            main_translation, alternatives = translate_libre(item, direction)

            # 收集所有翻译结果
            next_translations.append(main_translation)
            next_translations.extend(alternatives)
            # 去除重复的翻译结果
            next_translations = list(dict.fromkeys(next_translations))
        # 更新翻译集合
        translations = next_translations

    return translations


def compute_dissimilarity(sentence1, sentence2):
    """
    计算两个句子的不相似度
    
    Args:
        sentence1: 第一个句子
        sentence2: 第二个句子
        
    Returns:
        不相似度分数，范围为 0-1，越小表示越相似
    """
    # Jaccard 相似度: 计算两个集合的交集和并集的比率
    # 将句子拆分为单词集合
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())

    # 计算交集和并集
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # 避免除以零
    if union == 0:
        return 0

    return 1 - intersection / union


def compute_semantic_similarity(sentence1, sentence2):
    """使用预训练模型计算语义相似度（需安装sentence-transformers）"""

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    return float(util.pytorch_cos_sim(embedding1, embedding2)[0][0])


def augment_back_translate(
        original: str,
        candidates: list[str],
        alpha: float = 0.4,
        similarity_threshold: float = 0.7,
        semantic_similarity_threshold: float = 0.5) -> Optional[str]:
    """
    从 candidates 中根据加权随机选择一个回译结果。
    score = dissimilarity * α + semantic_similarity * (1-α)

    Args:
        original: 原始句子
        candidates: 回译结果列表
        alpha: 不相似度权重 (0~1)，语义相似度权重为 1-α

    Returns:
        最终选择的句子
    """
    scores = []
    new_candidates = []
    for cand in candidates:
        d = compute_dissimilarity(original, cand)  # Jaccard-based 不相似度
        s = compute_semantic_similarity(original, cand)  # 语义相似度
        # 滤除不相似度过大的句子和语义相似度过小的句子
        if d > similarity_threshold or s < semantic_similarity_threshold:
            continue
        # 如果句子过长，降低权重
        if len(cand) > len(original) * 1.5:
            d *= 0.3
            s *= 0.3
        scores.append(alpha * d + (1 - alpha) * s)
        new_candidates.append(cand)
    candidates = new_candidates
    if not scores or len(scores) == 0:
        return None
    ws = np.array(scores, dtype=float)
    # 保证所有权重非负
    ws = np.clip(ws, a_min=0, a_max=None)
    # 避免和为 0
    if ws.sum() == 0:
        ws = np.ones_like(ws)
    ws = ws / ws.sum()

    # 随机采样一个句子
    return random.choices(candidates, weights=ws.tolist(), k=1)[0]


if __name__ == "__main__":
    """
    测试回译功能以及计算相似度的主函数
    """
    print("=" * 70)
    print("回译测试与相似度计算")
    print("=" * 70)

    # 测试用例列表
    test_cases = [{
        "text": "He is doing his job.",
        "label": "习语表达",
        "direction": ("en2zh", "zh2en")
    }] * 10

    # 对每个测试用例进行回译并计算相似度
    for case in test_cases:
        text = case["text"]
        label = case["label"]
        directions = case["direction"]

        print(f"\n测试用例 ({label}):")
        print(f"原文: {text}")

        # 获取回译结果
        print(
            f"回译结果: {augment_back_translate(text, back_translate(text, directions=directions))}"
        )
