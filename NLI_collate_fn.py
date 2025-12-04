import torch
from torch.utils.data import DataLoader, ConcatDataset
from NLImodel import NLIModel
from datasets import Dataset


def get_collate_fn(model: NLIModel, type: str = "labeled"):
    """
    获取数据加载器的collate函数
    
    参数:
    - model: NLI模型实例
    - type: 需要的collate函数类型, 如"labeled"或"augmented_unlabel"或"filter"
    
    返回:
    - function: 数据加载器的collate函数
    """
    tokenizer = model.tokenizer

    # 增强数据和无标签数据的collate函数
    def collate_fn_augmented_unlabel(batch):
        premises = [item["premise"] for item in batch]
        hypotheses = [item["hypothesis"] for item in batch]
        labels = [item["label"] for item in batch]
        is_syn = [item.get("is_syn", 0) for item in batch
                  ]  # 默认为0 (False)，确保来自labeled_dataset的样本被视为原始样本
        pair_id = [item.get("pair_id", -1) for item in batch]
        mr_type = [item.get("mr_type", "inv") for item in batch]  # 默认为"inv"
        encodings = tokenizer(premises,
                              hypotheses,
                              truncation=True,
                              padding=True,
                              return_tensors="pt")
        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        encodings["is_syn"] = torch.tensor(is_syn, dtype=torch.bool)
        encodings["pair_id"] = pair_id  # 保留为 Python list（不进 GPU）
        encodings["mr_type"] = mr_type
        return encodings

    # 有标签数据的collate函数
    def collate_fn_labeled(batch):
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

    # 过滤数据的collate函数
    def collate_fn_fliter(batch):
        premises = [item["premise"] for item in batch]
        hypotheses = [item["hypothesis"] for item in batch]
        labels = [item["label"] for item in batch]
        is_syn = [item.get("is_syn", 0) for item in batch
                  ]  # 默认为0 (False)，确保来自labeled_dataset的样本被视为原始样本
        pair_id = [item.get("pair_id", -1) for item in batch]
        mr_type = [item.get("mr_type", "inv") for item in batch]  # 默认为"inv"
        encodings = tokenizer(premises,
                              hypotheses,
                              truncation=True,
                              padding=True,
                              return_tensors="pt")
        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        encodings["is_syn"] = torch.tensor(is_syn, dtype=torch.bool)
        encodings["pair_id"] = pair_id  # 保留为 Python list（不进 GPU）
        encodings["mr_type"] = mr_type
        return encodings, batch

    # 返回相应的collate函数
    if type == "labeled":
        return collate_fn_labeled
    elif type == "augmented_unlabel":
        return collate_fn_augmented_unlabel
    elif type == "filter":
        return collate_fn_fliter
    else:
        return None


def create_dataloaders(model: NLIModel,
                       labeled_dataset: Dataset,
                       augmented_dataset: Dataset = None,
                       unlabeled_dataset: Dataset = None,
                       validation_dataset: Dataset = None,
                       batch_size: int = 8):
    """
    创建各类数据的DataLoader
    
    参数:
    - model: NLI模型实例
    - labeled_dataset: 有标签数据集
    - augmented_dataset: 增强数据集（可选）
    - unlabeled_dataset: 无标签数据集（可选）
    - validation_dataset: 验证数据集（可选）
    - batch_size: 批处理大小
    
    返回:
    - dict: 包含各种数据加载器的字典
    """
    result = {}
    collate_fn_labeled = get_collate_fn(model, type="labeled")
    collate_fn_augmented_unlabel = get_collate_fn(model,
                                                  type="augmented_unlabel")
    # 创建有标签数据加载器
    result["labeled"] = DataLoader(labeled_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn_labeled)

    # 创建验证数据加载器（如果有）
    if validation_dataset is not None:
        result["validation"] = DataLoader(validation_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn_labeled)

    # 创建增强数据加载器（如果有）
    if augmented_dataset is not None:
        # 将 labeled_dataset 和 augmented_dataset 合并并排序
        # labeled_data = list(labeled_dataset)
        # augmented_data = list(augmented_dataset)

        # # 合并所有数据
        # all_data = labeled_data + augmented_data

        # # 按照 pair_id 排序，然后按照 is_syn 排序（0在前，1在后）
        # all_data.sort(key=lambda x: (x.get("pair_id", -1), x.get("is_syn", 0)))

        # # 创建新的 Dataset
        # combined_dataset = Dataset.from_list(all_data)
        result["augmented"] = DataLoader(
            augmented_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_augmented_unlabel)

    # 创建无标签数据加载器（如果有）
    if len(unlabeled_dataset) > 0:
        result["unlabeled"] = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_augmented_unlabel)

    return result
