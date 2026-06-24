# test.py
"""
本模块包含NLI模型的测试与推断方法。
可根据需要进行准确率、F1等指标的计算。
train_test_model是用于测试本脚本的模型名称，需要从train.py中生成，可以删除。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from NLImodel import NLIModel
from NLIdataset import NLIDataset
from mylogging import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def test_nli(model,
             tokenizer,
             test_dataset,
             batch_size: int = 16,
             label_smoothing: float = 0.0,
             label_map: dict = None):
    """
    测试NLI模型的函数
    -----------------
    参数:
    - model: 已训练好的NLI模型
    - tokenizer: 分词器
    - test_dataset: 测试数据集
    - batch_size (int): batch大小

    返回:
    - 测试集的多种指标（损失、准确率、精确率、召回率、F1分数）
    - dict: "loss", "accuracy", "precision", "recall", "f1"
    """

    # 获取模型当前所在的设备
    device = next(model.parameters()).device
    logger.info(f"模型当前在设备: {device}")

    def collate_fn(batch):
        premises = [item["premise"] for item in batch]
        hypotheses = [item["hypothesis"] for item in batch]
        labels = [item["label"] for item in batch]

        encodings = tokenizer(premises,
                              hypotheses,
                              truncation=True,
                              padding=True,
                              return_tensors="pt")
        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        return encodings

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    model.eval()
    preds = []
    gold_labels = []
    val_loss_sum = 0.0
    n_items = 0

    logger.info("开始测试模型...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            labels = batch["labels"].to(device)
            batch = {
                k: v.to(device)
                for k, v in batch.items() if k != "labels"
            }
            batch["labels"] = labels  # 将labels也移到设备上用于计算loss

            outputs = model(**batch)
            logits = outputs.logits
            loss = F.cross_entropy(logits,
                                   labels,
                                   reduction="sum",
                                   label_smoothing=label_smoothing)

            val_loss_sum += loss.item()
            n_items += labels.size(0)

            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            # 应用标签映射（如RTE二分类：将contradiction(2)映射为not_entailment(1)）
            if label_map:
                pred = [label_map.get(p, p) for p in pred]
            preds.extend(pred)
            gold_labels.extend(labels.cpu().numpy())

    # 计算多种指标
    accuracy = accuracy_score(gold_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels, preds, average="weighted")
    avg_loss = (val_loss_sum / n_items) if n_items > 0 else float("inf")

    logger.info(f"测试完成，结果如下：")
    logger.info(f"平均损失 (Loss): {avg_loss:.4f}")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1 分数 (F1 Score): {f1:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    logger.info("开始测试流程")

    # 初始化数据集
    dataset_handler = NLIDataset(dataset_name="snli",
                                 labeled_ratio=0.02,
                                 unlabeled_ratio=0.1 - 0.005,
                                 use_original_val=True)

    test_dataset = dataset_handler.get_test_dataset()
    logger.info(f"测试集加载完成，样本数: {len(test_dataset)}")

    # 加载模型
    # model_path = "/home/ubuntu/MTrain/MTrainNLI/nli_saved_model/snli/bert-base-uncased/1.0_False_mistral-7b-instruct-v0.2_0.8939_0.8937"
    model_path = "/home/ubuntu/MTrain/experiments/snli_lr0.0037_phi-4_augmented_data_三合一_0.8.json_20251105_153237/augment_training_best"
    logger.info(f"准备从以下路径加载模型: {model_path}")

    # 直接使用 load_model 类方法加载模型，并将其结果赋值给 model 变量
    model = NLIModel.load_model(load_dir=model_path)
    tokenizer = model.tokenizer
    logger.info("模型加载完成，准备开始测试")

    # 测试模型
    metrics = test_nli(model.model, tokenizer, test_dataset, batch_size=8)

    logger.info("测试流程结束，结果如下：")
    logger.info(metrics)
