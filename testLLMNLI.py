"""
llama-3.2-3b-instruct:
    Accuracy: 0.3896
    Precision: 0.5063
    Recall: 0.3896
    F1: 0.3341

llama-3-8b-instruct:
    Accuracy: 0.5114
    Precision: 0.5924
    Recall: 0.5114
    F1: 0.4753
"""

import torch
from LMstudio import HOST, MODEL, PORT, OpenAIChat
from NLIdataset import NLIDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from mylogging import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


def evaluate_nli_model(model,
                       test_dataset,
                       dataset_name,
                       batch_size=8,
                       max_workers=4):
    """
    测试大模型的 NLI 能力，并计算 accuracy, precision, recall 和 f1。
    自动适配二分类（如 RTE）与三分类（如 SNLI、SICK、MNLI/MNLI-MM）数据集。

    参数：
    - model: OpenAIChat 模型实例
    - test_dataset: 测试数据集
    - dataset_name: 数据集名称，用于判断分类任务类型
    - batch_size: 每次测试的样本数量
    - max_workers: 并发线程数

    返回：
    - metrics: 包含 accuracy, precision, recall 和 f1 的字典
    """
    logger.info("开始测试大模型的 NLI 能力...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    dataset_name_lower = dataset_name.lower()
    is_binary_task = dataset_name_lower == "rte"
    num_classes = 2 if is_binary_task else 3
    logger.info(
        f"任务类型: {'二分类' if is_binary_task else '三分类'} (dataset={dataset_name_lower})")

    # 推断标签空间，兼容不同标签编号
    label_to_id = {}
    if hasattr(test_dataset, "features") and "label" in test_dataset.features:
        label_feature = test_dataset.features["label"]
        if hasattr(label_feature, "names") and label_feature.names:
            label_to_id = {
                name.lower(): idx
                for idx, name in enumerate(label_feature.names)
            }

    label_ids_present = sorted({
        int(item) for item in test_dataset["label"] if item is not None
    })
    if not label_ids_present:
        label_ids_present = list(range(num_classes))

    if not label_to_id and label_ids_present:
        label_to_id["entailment"] = label_ids_present[0]
        if is_binary_task and len(label_ids_present) >= 2:
            label_to_id["not_entailment"] = label_ids_present[1]
        elif len(label_ids_present) >= 3:
            label_to_id["neutral"] = label_ids_present[1]
            label_to_id["contradiction"] = label_ids_present[2]

    entail_id = label_to_id.get(
        "entailment",
        label_ids_present[0] if label_ids_present else 0,
    )
    neutral_id = label_to_id.get("neutral")
    contradiction_id = label_to_id.get("contradiction")
    non_entail_id = label_to_id.get("not_entailment")

    if is_binary_task:
        if non_entail_id is None and len(label_ids_present) > 1:
            non_entail_id = next(
                (lid for lid in label_ids_present if lid != entail_id), entail_id)
        if non_entail_id is None or non_entail_id == entail_id:
            non_entail_id = next(
                (lid for lid in reversed(label_ids_present) if lid != entail_id),
                entail_id,
            )
        fallback_non_entail_id = non_entail_id
    else:
        if neutral_id is None and len(label_ids_present) >= 2:
            neutral_id = label_ids_present[1]
        if contradiction_id is None and len(label_ids_present) >= 3:
            contradiction_id = label_ids_present[2]
        fallback_non_entail_id = (contradiction_id if contradiction_id is not None
                                   else (label_ids_present[-1]
                                         if label_ids_present else -1))

    def contains_any(text, keywords):
        return any(keyword in text for keyword in keywords)

    # 收集预测值和真实标签
    all_preds = []
    all_labels = []

    def process_sample(premise, hypothesis, label):
        """
        处理单个样本，调用模型并返回预测值和真实标签。
        """
        input_text = (
            f"Premise: {premise}\nHypothesis: {hypothesis}\n"
            "Is the hypothesis entailed by the premise? "
            "Answer with 'entailment', 'neutral', or 'contradiction'.")
        response = model.chat(input_text)
        logger.info(f"模型回复: {response}")

        normalized = response.lower()

        negative_keywords = (
            "not entail", "non-entail", "non entail", "not_entail",
            "does not entail", "contradiction", "contradicts", "false",
            "disagrees", "no", "refute", "refutes")
        neutral_keywords = (
            "neutral", "undetermined", "cannot determine", "unknown")
        entail_keywords = ("entailment", "entailed", "entails", "yes")

        if contains_any(normalized, negative_keywords):
            if num_classes == 2:
                pred = fallback_non_entail_id
            else:
                pred = (contradiction_id if contradiction_id is not None else
                        fallback_non_entail_id)
        elif num_classes == 2 and contains_any(normalized, neutral_keywords):
            pred = fallback_non_entail_id
        elif num_classes == 3 and contains_any(normalized, neutral_keywords):
            pred = neutral_id if neutral_id is not None else fallback_non_entail_id
        elif contains_any(normalized, entail_keywords):
            pred = entail_id
        else:
            logger.warning(f"无法解析模型回复: {response}")
            pred = fallback_non_entail_id if num_classes == 2 else -1

        return pred, label

    # 使用 ThreadPoolExecutor 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]
            premises = batch["premise"]
            hypotheses = batch["hypothesis"]
            labels = batch["label"]

            for premise, hypothesis, label in zip(premises, hypotheses,
                                                  labels):
                futures.append(
                    executor.submit(process_sample, premise, hypothesis,
                                    label))

        for future in as_completed(futures):
            pred, label = future.result()
            all_preds.append(pred)
            all_labels.append(label)

    # 过滤掉无法解析的预测值
    valid_indices = [i for i, pred in enumerate(all_preds) if pred != -1]
    valid_preds = [all_preds[i] for i in valid_indices]
    valid_labels = [all_labels[i] for i in valid_indices]

    if not valid_preds:
        logger.warning("未获得有效预测结果，返回空指标。")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # 计算指标
    accuracy = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    logger.info(f"测试完成，结果如下：")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    return metrics


if __name__ == "__main__":
    # 初始化 OpenAIChat 模型
    bot = OpenAIChat(
        model="Mistral-7B-instruct-v0.2",
        host=HOST,
        port=PORT,
        stateless=True,  # 无状态模式
    )

    # 加载 SNLI 数据集
    dataset_handler = NLIDataset(dataset_name="rte",
                                 val_ratio=0.1,
                                 labeled_ratio=0.005,
                                 unlabeled_ratio=0.1 - 0.005,
                                 use_original_val=True)
    test_dataset = dataset_handler.get_test_dataset()
    

    # 测试模型
    metrics = evaluate_nli_model(bot,
                                 test_dataset,
                                 dataset_handler.dataset_name,
                                 batch_size=8,
                                 max_workers=4)

    # 打印结果
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
