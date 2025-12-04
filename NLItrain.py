# train.py
"""
本模块包含NLI模型的训练相关的方法。
train_test_model是用于测试本脚本的模型名称，可以删除。
"""

import os
from typing import Optional
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW
from tqdm import tqdm
from NLImodel import NLIModel
from mylogging import logger
import torch.nn.functional as F
from datasets import Dataset

# ============================== 数据加载器 ==============================

from NLI_collate_fn import get_collate_fn, create_dataloaders

# ============================== 训练函数 ==============================

from NLI_train_one_epoch import train_one_epoch_RDrop, train_one_epoch_safe_MR_multiview
from NLI_train_one_epoch import train_one_epoch_dual_consistency

# ============================== 训练主函数 ==============================

from NLI_train_stage import train_with_labeled_data, train_with_consistency, train_with_augment, train_with_unlabeled_data

# ============================== 数据筛选函数 ==============================


def filter_dataset_by_confidence(model: NLIModel, dataset_to_filter: Dataset,
                                 device, confidence_threshold: float,
                                 batch_size: int):
    """
    根据模型预测的置信度筛选数据集。

    参数:
    - model: NLIModel实例。
    - dataset_to_filter: 待筛选的Hugging Face数据集 (包含原始样本字典)。
    - device: 计算设备 (e.g., "cuda" or "cpu")。
    - confidence_threshold: 置信度阈值。高于此阈值的样本将被保留。
    - batch_size: 用于模型推理的批处理大小。

    返回:
    - Dataset: 筛选后的Hugging Face数据集。如果所有样本都被滤除，则返回空数据集。
    """
    # 将模型设置为评估模式。这会禁用诸如Dropout之类的层，确保预测结果的确定性。
    model_instance = model.model
    model_instance.eval()

    # 初始化一个空列表，用于存储通过置信度筛选的样本。
    filtered_samples: list[dict] = []

    # 获取用于数据加载器的collate函数。
    # `is_labeled=False` 通常用于增强数据或无标签数据，
    # `collate_fn_augmented_unlabel` 会被使用，它能处理包含 'premise', 'hypothesis', 'label' (可以是占位符) 等字段的样本。
    # 创建一个DataLoader用于批量加载待筛选的数据。
    # `shuffle=False` 因为筛选时样本顺序无关紧要。
    temp_loader = DataLoader(
        dataset_to_filter,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(model, type="filter"),
    )

    # 使用 `torch.no_grad()` 上下文管理器进行推理。
    # 在此块中，PyTorch不会计算或存储梯度，从而减少内存使用并加速计算。
    with torch.no_grad():
        # 使用tqdm显示处理进度条，遍历DataLoader生成的每个批次。
        # 每个迭代步骤产生 `encodings` (模型输入) 和 `original_items_batch` (原始样本列表)。
        for encodings, original_items_batch in tqdm(
                temp_loader, desc=f"Filtering dataset by confidence"):

            # 从 `encodings` 字典中提取必要的输入张量，并将它们移动到指定的计算设备。
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            # `token_type_ids` (或称 segment IDs) 用于某些模型（如BERT）区分句子对中的不同句子。
            # 使用 `.get()` 以安全地处理此字段可能不存在的情况（例如对于RoBERTa模型）。
            token_type_ids = encodings.get("token_type_ids", None)

            # 构建模型输入字典
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids.to(
                    device)  # 也移动到设备

            # 将输入数据传递给模型进行前向传播，获取模型输出。
            outputs = model_instance(**model_inputs)

            # 从模型输出中提取 `logits`。Logits是模型对每个类别的原始、未标准化的预测分数。
            logits = outputs.logits
            # 使用Softmax函数将logits转换为概率分布。每一行的概率总和为1。
            probabilities = F.softmax(logits, dim=-1)
            # `torch.max` 沿着最后一个维度（类别维度）找到最大概率及其对应的索引。
            # `confidences` 张量包含每个样本的最大预测概率（即置信度）。
            # `predicted_labels` 张量包含每个样本预测的类别索引。
            confidences, predicted_labels = torch.max(probabilities, dim=-1)

            # 遍历当前批次中的每一个原始样本。
            for i in range(len(original_items_batch)):
                # 获取当前样本的模型预测置信度。`.item()` 将单元素张量转换为Python标量。
                item_confidence = confidences[i].item()

                # 检查该样本的置信度是否达到或超过预设的阈值,并且标签与预测的标签一致。
                if item_confidence >= confidence_threshold and \
                        original_items_batch[i]["label"] == predicted_labels[i].item():
                    # 如果满足条件，创建原始样本字典的一个副本。
                    # 复制是为了避免修改原始数据集（`dataset_to_filter`）中的数据，
                    # 特别是当我们要向副本添加伪标签时。
                    sample_copy = original_items_batch[i].copy()
                    # 将通过筛选（且可能已添加伪标签）的样本副本添加到 `filtered_samples` 列表中。
                    filtered_samples.append(sample_copy)

    # 筛选过程结束后，检查 `filtered_samples` 列表是否为空。
    if not filtered_samples:
        # 如果列表为空（即没有样本满足置信度阈值），记录一条警告信息。
        logger.warning(f"过滤数据集为空。阈值：{confidence_threshold}。这可能会影响后续的训练。")
        # 返回一个空的Hugging Face Dataset对象，以保持函数返回类型的一致性。
        return Dataset.from_list([])

    # 记录筛选操作的结果，包括原始数据集大小和筛选后的数据集大小。
    logger.info(
        f"原始数据集大小：{len(dataset_to_filter)}，过滤后的数据集大小为{len(filtered_samples)}（置信阈值：{confidence_threshold}）"
    )
    # 使用 `filtered_samples` 列表（其中每个元素都是一个符合条件的样本字典）
    # 创建一个新的Hugging Face Dataset对象，并将其返回。
    return Dataset.from_list(filtered_samples)


def filter_dataset(model: NLIModel,
                   dataset_to_filter: Dataset,
                   device,
                   batch_size: int,
                   _collate_fn_augmented,
                   another_dataset: Dataset = None,
                   confidence_threshold: float = 0.85) -> Optional[DataLoader]:
    """
    筛选增强数据集，保留置信度高于阈值的样本。
    参数:
    - model: NLIModel实例。
    - dataset_to_filter: 待筛选的增强数据集。
    - device: 计算设备 (e.g., "cuda" or "cpu")。
    - batch_size: 用于模型推理的批处理大小。
    - _collate_fn_augmented: 用于增强数据集的collate函数。
    返回:
    - Dataset: 筛选后的增强数据集。如果所有样本都被滤除，则返回空数据集。
    """
    model.model.eval()
    filtered_dataset = filter_dataset_by_confidence(
        model,
        dataset_to_filter,
        device,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size)
    if len(filtered_dataset) > 0:
        logger.info(f"增强数据集筛选完毕。筛选后样本数: {len(filtered_dataset)}. 重建增强数据加载器...")
        # 使用筛选后的增强数据和原始的 labeled_dataset 重建 ConcatDataset
        if another_dataset is None:
            another_dataset = Dataset.from_list([])
        combined_augmented_dataset_after_filtering: ConcatDataset = ConcatDataset(
            [another_dataset, filtered_dataset])
        combined_dataset = DataLoader(
            combined_augmented_dataset_after_filtering,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn_augmented)
    else:
        logger.warning("增强数据集在筛选后为空。后续增强数据训练阶段可能被跳过或仅使用有标签数据。")
        combined_dataset = None  # 设置为None，以便后续训练阶段正确跳过
    return combined_dataset


# ============================== 训练主函数 ==============================


def train_nli(
        model: NLIModel,
        labeled_dataset: Dataset,
        augmented_dataset: Dataset = None,
        unlabeled_dataset: Dataset = None,
        validation_dataset: Dataset = None,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        epochs: int = 25,
        patience: int = 12,
        alpha_init: float = 0.1,  # CE 权重 α
        gamma: float = 0.5,  # R-Drop KL 权重
        # 双重一致性训练参数 (用于增强数据阶段)
    dc_alpha: float = 0.3,
        dc_lam_inv: float = 1.0,
        dc_lam_flip: float = 0.5,
        # 无标签数据训练参数 (使用双重一致性对比逻辑)
        ul_alpha: float = 0.3,
        ul_lam_inv: float = 1.0,
        ul_lam_flip: float = 0.5,
        ul_lam_cont: float = 0.4,
        ul_temperature: float = 0.1,
        use_data_augmentation=False,
        ckpt_name: str = "nli_model",
        only_unlabeled: bool = False):
    """
    训练NLI模型的主函数 (半监督流程)
    
    参数:
    - model: NLIModel对象
    - labeled_dataset: 有标签训练数据
    - augmented_dataset: 增强数据集（可选）
    - unlabeled_dataset: 无标签训练数据（可选）
    - validation_dataset: 验证数据集（可选）
    - learning_rate: 学习率
    - batch_size: batch大小
    - epochs: 每阶段训练轮数
    - patience: 早停机制的容忍轮数
    - alpha_init: CE损失初始权重
    - gamma: R-Drop KL散度权重
    - dc_alpha: 双重一致性训练阶段的CE损失权重
    - dc_lam_inv: 双重一致性训练阶段的标签保持KL损失权重
    - dc_lam_flip: 双重一致性训练阶段的标签翻转KL损失权重
    - ul_alpha: 无标签数据训练阶段的CE损失权重
    - ul_lam_inv: 无标签数据训练阶段的标签保持KL损失权重
    - ul_lam_flip: 无标签数据训练阶段的标签翻转KL损失权重
    - ul_lam_cont: 无标签数据训练阶段的对比损失权重
    - ul_temperature: 无标签数据训练阶段的对比损失温度参数
    - ckpt_name: 模型检查点名称
    
    返回:
    - model_instance: 训练后的模型实例
    """
    device = model.device
    model_instance = model.model

    # 记录数据集信息
    logger.info(f"使用设备：{device}")
    logger.info(f"模型初始化完成，使用预训练模型: {model.model_name}")
    logger.info(f"有标签数据集大小: {len(labeled_dataset)}")
    if unlabeled_dataset is not None:
        logger.info(f"无标签数据集大小: {len(unlabeled_dataset)}")
    if validation_dataset is not None:
        logger.info(f"验证数据集大小: {len(validation_dataset)}")
    if augmented_dataset is not None:
        logger.info(f"增强数据集大小: {len(augmented_dataset)}")

    # 初始化优化器
    optimizer = AdamW(model_instance.parameters(),
                      lr=learning_rate,
                      weight_decay=0.01)
    logger.info(f"优化器AdamW已初始化，学习率: {learning_rate}")

    # 创建数据加载器
    loaders = create_dataloaders(model, labeled_dataset, augmented_dataset,
                                 unlabeled_dataset, validation_dataset,
                                 batch_size)

    # # ========================================== 阶段1: 有标签数据训练 =========================================
    if not only_unlabeled:
        model_instance, best_loss, current_alpha_labeled = train_with_labeled_data(
            model_instance, loaders, optimizer, device, epochs, patience,
            alpha_init, gamma)

        # # 保存模型状态
        # ckpt_base_dir = "/home/ubuntu/MTrain/MTrainNLI/nli_model_ckpt"
        # ckpt_base_name = f"labeled_{best_loss}_{ckpt_name}"
        # ckpt_path = os.path.join(ckpt_base_dir, ckpt_base_name)
        # try:
        #     model.save_model(ckpt_path)
        #     logger.info(f"已将模型保存到 {ckpt_path}")
        # except Exception as e:
        #     logger.error(f"保存模型失败: {e}")

        # # 调试阶段直接加载模型
        # ckpt_base = "/home/ubuntu/MTrain/MTrainNLI/nli_model_ckpt/labeled_0.6673151768185805_snli_lr0.0037_phi-4"
        # model = NLIModel.load_model(ckpt_base)
        # logger.info(f"阶段一模型加载完成: {ckpt_base}")
        # # 测试模型，得出基本参数以作参考
        # import NLItest
        # test_metric = NLItest.test_nli(model=model.model,
        #                                tokenizer=model.tokenizer,
        #                                test_dataset=validation_dataset,
        #                                batch_size=batch_size)
        # logger.info(f"加载的阶段一模型测试完成: \n{test_metric['accuracy']:.4f}")

        # 在阶段一后，筛选增强数据
        if use_data_augmentation and "augmented" in loaders and loaders[
                "augmented"] is not None:
            logger.info("阶段一后：开始基于置信度筛选增强数据集...")
            loaders["augmented"] = filter_dataset(
                model=model,
                dataset_to_filter=augmented_dataset,
                device=device,
                batch_size=batch_size,
                _collate_fn_augmented=loaders["augmented"].collate_fn,
                another_dataset=labeled_dataset)

    # ========================================== 阶段2: 双重一致性训练 =========================================
    # # 减少学习率
    # reduce_lr_factor = 0.001  # 减少学习率的因子
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] *= reduce_lr_factor
    #     logger.info(f"双重一致性训练阶段：学习率已调整为 {learning_rate * reduce_lr_factor}")

    if not only_unlabeled:
        model_instance, _, current_alpha_dc, current_lam_inv_dc, current_lam_flip_dc = train_with_augment(
            model_instance, loaders, optimizer, device, epochs, dc_alpha,
            dc_lam_inv, dc_lam_flip)
        logger.info(
            f"双重一致性训练阶段完成。最终参数: alpha={current_alpha_dc:.3f}, lam_inv={current_lam_inv_dc:.3f}, lam_flip={current_lam_flip_dc:.3f}"
        )

        # # # 在阶段二后，筛选无标签数据 (赋予伪标签)
        # if use_data_augmentation and unlabeled_dataset is not None and "unlabeled" in loaders and loaders[
        #         "unlabeled"] is not None:
        #     logger.info("阶段二后：开始基于置信度筛选无标签数据集...")
        #     loaders["unlabeled"] = filter_dataset(
        #         model=model,
        #         dataset_to_filter=unlabeled_dataset,
        #         device=device,
        #         batch_size=batch_size,
        #         confidence_threshold=0.9,
        #         _collate_fn_augmented=loaders["unlabeled"].collate_fn)

    # ========================================== 阶段3: 无标签数据训练 =========================================
    if use_data_augmentation:
        model_instance, _, _, _, _, _ = train_with_unlabeled_data(
            model_instance,
            loaders,
            optimizer,
            device,
            epochs,
            patience,
            alpha_ul=ul_alpha,
            lam_inv_ul=ul_lam_inv,
            lam_flip_ul=ul_lam_flip,
            lam_cont_ul=ul_lam_cont,
            temperature_ul=ul_temperature)

    logger.info("训练完成，返回模型实例")
    return model_instance


def train_nli_progressive(
        model: NLIModel,
        labeled_dataset: Dataset,
        augmented_dataset: Dataset = None,
        unlabeled_dataset: Dataset = None,
        validation_dataset: Dataset = None,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        epochs_per_stage: int = 5,
        num_iterations: int = 5,  # 重复三个阶段的次数
        patience: int = 5,
        alpha_init: float = 0.1,
        gamma: float = 0.5,
        dc_alpha: float = 0.3,
        dc_lam_inv: float = 1.0,
        dc_lam_flip: float = 0.5,
        ul_alpha: float = 0.3,
        ul_lam_inv: float = 1.0,
        ul_lam_flip: float = 0.5,
        ul_lam_cont: float = 0.4,
        ul_temperature: float = 0.1,
        augment_increment_ratio: float = 0.3,  # 每次增加增强数据的比例
        unlabel_increment_ratio: float = 0.3,  # 每次增加无标签数据的比例
        ckpt_name: str = "nli_model_progressive"):
    """
    渐进式训练NLI模型的主函数，通过逐步增加数据集来防止过拟合
    
    参数:
    - model: NLIModel对象
    - labeled_dataset: 有标签训练数据
    - augmented_dataset: 增强数据集（可选）
    - unlabeled_dataset: 无标签训练数据（可选）
    - validation_dataset: 验证数据集（可选）
    - learning_rate: 学习率
    - batch_size: batch大小
    - epochs_per_stage: 每个子阶段的训练轮数
    - num_iterations: 重复三个阶段的次数
    - patience: 早停机制的容忍轮数
    - augment_increment_ratio: 每次增加增强数据的比例
    - unlabel_increment_ratio: 每次增加无标签数据的比例
    - 其他参数: 与原train_nli函数相同
    
    返回:
    - model_instance: 训练后的模型实例
    """
    device = model.device
    model_instance = model.model

    # 记录初始信息
    logger.info(f"开始渐进式训练，共{num_iterations}轮迭代")
    logger.info(f"使用设备：{device}")
    logger.info(f"有标签数据集大小: {len(labeled_dataset)}")
    if augmented_dataset is not None:
        logger.info(f"增强数据集大小: {len(augmented_dataset)}")
    if unlabeled_dataset is not None:
        logger.info(f"无标签数据集大小: {len(unlabeled_dataset)}")
    if validation_dataset is not None:
        logger.info(f"验证数据集大小: {len(validation_dataset)}")

    # 初始化优化器
    optimizer = AdamW(model_instance.parameters(),
                      lr=learning_rate,
                      weight_decay=0.01)
    original_lr = learning_rate

    # 计算每次增加的数据量
    augment_total_size = len(augmented_dataset) if augmented_dataset else 0
    unlabel_total_size = len(unlabeled_dataset) if unlabeled_dataset else 0

    current_augment_size = 0
    current_unlabel_size = 0

    for iteration in range(num_iterations):
        logger.info(
            f"\n========== 开始第 {iteration + 1}/{num_iterations} 轮迭代 =========="
        )

        # ==================== 阶段1: 仅使用有标签数据 ====================
        logger.info(f"第{iteration + 1}轮 - 阶段1: 有标签数据训练")

        # 为每轮迭代重置学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = original_lr * (0.8**iteration)  # 每轮迭代降低学习率

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"当前学习率: {current_lr}")

        # 创建仅包含有标签数据的加载器
        loaders_stage1 = create_dataloaders(model, labeled_dataset, None, None,
                                            validation_dataset, batch_size)

        model_instance, _, _ = train_with_labeled_data(
            model_instance, loaders_stage1, optimizer, device,
            epochs_per_stage, patience, alpha_init, gamma)

        # ==================== 阶段2: 逐步增加增强数据 ====================
        if augmented_dataset is not None and augment_total_size > 0:
            logger.info(f"第{iteration + 1}轮 - 阶段2: 逐步增加增强数据训练")

            # 计算本轮应该增加的增强数据量
            increment_size = int(augment_total_size * augment_increment_ratio)
            current_augment_size = min(current_augment_size + increment_size,
                                       augment_total_size)

            logger.info(
                f"使用增强数据量: {current_augment_size}/{augment_total_size}")

            if current_augment_size > 0:
                # 创建当前大小的增强数据集
                # 使用 select 方法从 Dataset 中选择指定范围的数据
                current_augmented_dataset = augmented_dataset.select(
                    range(current_augment_size))

                # 先筛选增强数据
                filtered_augmented = filter_dataset_by_confidence(
                    model,
                    current_augmented_dataset,
                    device,
                    confidence_threshold=0.85,
                    batch_size=batch_size)

                if len(filtered_augmented) > 0:
                    # 创建包含有标签数据和筛选后增强数据的加载器
                    loaders_stage2 = create_dataloaders(
                        model, labeled_dataset, filtered_augmented, None,
                        validation_dataset, batch_size)

                    # 降低学习率进行增强数据训练
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5

                    model_instance, _, current_alpha_dc, current_lam_inv_dc, current_lam_flip_dc = train_with_augment(
                        model_instance, loaders_stage2, optimizer, device,
                        epochs_per_stage, dc_alpha, dc_lam_inv, dc_lam_flip)
                else:
                    logger.warning("增强数据筛选后为空，跳过阶段2")

        # ==================== 阶段3: 逐步增加无标签数据 ====================
        if unlabeled_dataset is not None and unlabel_total_size > 0:
            logger.info(f"第{iteration + 1}轮 - 阶段3: 逐步增加无标签数据训练")

            # 计算本轮应该增加的无标签数据量
            increment_size = int(unlabel_total_size * unlabel_increment_ratio)
            current_unlabel_size = min(current_unlabel_size + increment_size,
                                       unlabel_total_size)

            logger.info(
                f"使用无标签数据量: {current_unlabel_size}/{unlabel_total_size}")

            if current_unlabel_size > 0:
                # 创建当前大小的无标签数据集
                # 使用 select 方法从 Dataset 中选择指定范围的数据
                current_unlabeled_dataset = unlabeled_dataset.select(
                    range(current_unlabel_size))

                # 筛选无标签数据（赋予伪标签）
                filtered_unlabeled = filter_dataset_by_confidence(
                    model,
                    current_unlabeled_dataset,
                    device,
                    confidence_threshold=0.9,
                    batch_size=batch_size)

                if len(filtered_unlabeled) > 0:
                    # 准备当前阶段的增强数据（如果有的话）
                    current_aug_for_stage3 = None
                    if augmented_dataset is not None and current_augment_size > 0:
                        current_aug_for_stage3 = augmented_dataset.select(
                            range(current_augment_size))

                    # 创建完整的数据加载器
                    loaders_stage3 = create_dataloaders(
                        model, labeled_dataset, current_aug_for_stage3,
                        filtered_unlabeled, validation_dataset, batch_size)

                    # 进一步降低学习率进行无标签数据训练
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5

                    model_instance, _, _, _, _, _ = train_with_unlabeled_data(
                        model_instance, loaders_stage3, optimizer, device,
                        epochs_per_stage, patience, ul_alpha, ul_lam_inv,
                        ul_lam_flip, ul_lam_cont, ul_temperature)
                else:
                    logger.warning("无标签数据筛选后为空，跳过阶段3")

        logger.info(f"第{iteration + 1}轮迭代完成")

        # 在验证集上评估当前模型性能
        if validation_dataset is not None:
            try:
                import NLItest
                test_metric = NLItest.test_nli(model=model_instance,
                                               tokenizer=model.tokenizer,
                                               test_dataset=validation_dataset,
                                               batch_size=batch_size)
                logger.info(
                    f"第{iteration + 1}轮验证准确率: {test_metric['accuracy']:.4f}")
            except ImportError:
                logger.warning("无法导入NLItest模块，跳过验证")

    logger.info("渐进式训练完成，返回模型实例")
    return model_instance


def train_nli_unlabeled_labeled(
        model: NLIModel,
        labeled_dataset: Dataset,
        augmented_dataset: Dataset = None,
        unlabeled_dataset: Dataset = None,
        validation_dataset: Dataset = None,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        unlabeled_epochs: int = 15,  # 无标签数据预训练轮数
        labeled_epochs: int = 10,  # 有标签数据微调轮数
        augmented_epochs: int = 8,  # 增强数据训练轮数
        patience: int = 8,
        # 无标签数据预训练参数
        ul_pretrain_alpha: float = 0.2,
        ul_pretrain_lam_inv: float = 0.8,
        ul_pretrain_lam_flip: float = 0.4,
        ul_pretrain_lam_cont: float = 0.6,
        ul_pretrain_temperature: float = 0.15,
        # 有标签数据微调参数
        labeled_alpha: float = 0.5,
        labeled_gamma: float = 0.3,
        # 增强数据训练参数
        aug_alpha: float = 0.4,
        aug_lam_inv: float = 0.6,
        aug_lam_flip: float = 0.3,
        ckpt_name: str = "nli_model_ul_labeled"):
    """
    先用无标签数据训练模型学习规律，然后用有标签数据和增强数据进行微调的训练方法
    
    训练流程:
    1. 阶段1: 无标签数据预训练 - 让模型学习基本的语言理解和推理模式
    2. 阶段2: 有标签数据微调 - 使用准确标签提升模型准确性
    3. 阶段3: 增强数据优化 - 使用增强数据进一步提升泛化能力
    
    参数:
    - model: NLIModel对象
    - labeled_dataset: 有标签训练数据
    - augmented_dataset: 增强数据集（可选）
    - unlabeled_dataset: 无标签训练数据（可选）
    - validation_dataset: 验证数据集（可选）
    - learning_rate: 初始学习率
    - batch_size: batch大小
    - unlabeled_epochs: 无标签数据预训练轮数
    - labeled_epochs: 有标签数据微调轮数
    - augmented_epochs: 增强数据训练轮数
    - patience: 早停机制的容忍轮数
    - ul_pretrain_*: 无标签数据预训练阶段的各项参数
    - labeled_*: 有标签数据微调阶段的各项参数
    - aug_*: 增强数据训练阶段的各项参数
    - ckpt_name: 模型检查点名称
    
    返回:
    - model_instance: 训练后的模型实例
    """
    device = model.device
    model_instance = model.model

    # 记录数据集信息
    logger.info(f"========== 开始 先无标签后有标签 训练流程 ==========")
    logger.info(f"使用设备：{device}")
    logger.info(f"模型初始化完成，使用预训练模型: {model.model_name}")
    logger.info(f"有标签数据集大小: {len(labeled_dataset)}")
    if unlabeled_dataset is not None:
        logger.info(f"无标签数据集大小: {len(unlabeled_dataset)}")
    if augmented_dataset is not None:
        logger.info(f"增强数据集大小: {len(augmented_dataset)}")
    if validation_dataset is not None:
        logger.info(f"验证数据集大小: {len(validation_dataset)}")

    # 初始化优化器
    optimizer = AdamW(model_instance.parameters(),
                      lr=learning_rate,
                      weight_decay=0.01)
    logger.info(f"优化器AdamW已初始化，学习率: {learning_rate}")

    # ========================================== 阶段1: 无标签数据预训练 =========================================
    if unlabeled_dataset is not None and len(unlabeled_dataset) > 0:
        logger.info("========== 阶段1: 无标签数据预训练 ==========")
        logger.info("通过无标签数据让模型学习语言理解和推理的基本模式...")

        # 创建仅包含无标签数据的加载器
        loaders_stage1 = create_dataloaders(model, labeled_dataset, None,
                                            unlabeled_dataset,
                                            validation_dataset, batch_size)

        # 设置较低的学习率进行预训练
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * 0.5

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"无标签预训练阶段学习率: {current_lr}")

        # 使用无标签数据进行预训练
        model_instance, _, _, _, _, _ = train_with_unlabeled_data(
            model_instance,
            loaders_stage1,
            optimizer,
            device,
            unlabeled_epochs,
            patience,
            alpha_ul=ul_pretrain_alpha,
            lam_inv_ul=ul_pretrain_lam_inv,
            lam_flip_ul=ul_pretrain_lam_flip,
            lam_cont_ul=ul_pretrain_lam_cont,
            temperature_ul=ul_pretrain_temperature)

        logger.info("========== 阶段1: 无标签数据预训练完成 ==========")

        # 评估预训练效果
        if validation_dataset is not None:
            try:
                import NLItest
                test_metric = NLItest.test_nli(model=model_instance,
                                               tokenizer=model.tokenizer,
                                               test_dataset=validation_dataset,
                                               batch_size=batch_size)
                logger.info(f"无标签预训练后验证准确率: {test_metric['accuracy']:.4f}")
            except ImportError:
                logger.warning("无法导入NLItest模块，跳过预训练后验证")
    else:
        logger.warning("未提供无标签数据集，跳过无标签预训练阶段")

    # ========================================== 阶段2: 有标签数据微调 =========================================
    logger.info("========== 阶段2: 有标签数据微调 ==========")
    logger.info("使用有标签数据进行精确微调，提升模型准确性...")

    # 创建包含有标签数据的加载器
    loaders_stage2 = create_dataloaders(model, labeled_dataset, None, None,
                                        validation_dataset, batch_size)

    # 恢复标准学习率进行有标签数据微调
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * 0.3

    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(f"有标签微调阶段学习率: {current_lr}")

    # 使用有标签数据进行微调
    model_instance, best_loss, current_alpha_labeled = train_with_labeled_data(
        model_instance, loaders_stage2, optimizer, device, labeled_epochs,
        patience, labeled_alpha, labeled_gamma)

    logger.info("========== 阶段2: 有标签数据微调完成 ==========")

    # 评估微调效果
    if validation_dataset is not None:
        try:
            import NLItest
            test_metric = NLItest.test_nli(model=model_instance,
                                           tokenizer=model.tokenizer,
                                           test_dataset=validation_dataset,
                                           batch_size=batch_size)
            logger.info(f"有标签微调后验证准确率: {test_metric['accuracy']:.4f}")
        except ImportError:
            logger.warning("无法导入NLItest模块，跳过微调后验证")

    # ========================================== 阶段3: 增强数据优化 =========================================
    if augmented_dataset is not None and len(augmented_dataset) > 0:
        logger.info("========== 阶段3: 增强数据优化 ==========")
        logger.info("使用增强数据进一步提升模型的泛化能力...")

        # 筛选增强数据集，保留高置信度样本
        logger.info("筛选高置信度的增强数据...")
        filtered_augmented = filter_dataset_by_confidence(
            model,
            augmented_dataset,
            device,
            confidence_threshold=0.80,  # 相对宽松的阈值
            batch_size=batch_size)

        if len(filtered_augmented) > 0:
            # 创建包含有标签数据和筛选后增强数据的加载器
            loaders_stage3 = create_dataloaders(model, labeled_dataset,
                                                filtered_augmented, None,
                                                validation_dataset, batch_size)

            # 进一步降低学习率进行增强数据训练
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * 0.1

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"增强数据优化阶段学习率: {current_lr}")

            # 使用增强数据进行训练
            model_instance, _, current_alpha_aug, current_lam_inv_aug, current_lam_flip_aug = train_with_augment(
                model_instance, loaders_stage3, optimizer, device,
                augmented_epochs, aug_alpha, aug_lam_inv, aug_lam_flip)

            logger.info(
                f"增强数据优化阶段完成。最终参数: alpha={current_alpha_aug:.3f}, lam_inv={current_lam_inv_aug:.3f}, lam_flip={current_lam_flip_aug:.3f}"
            )

            # 评估最终效果
            if validation_dataset is not None:
                try:
                    import NLItest
                    test_metric = NLItest.test_nli(
                        model=model_instance,
                        tokenizer=model.tokenizer,
                        test_dataset=validation_dataset,
                        batch_size=batch_size)
                    logger.info(f"增强数据优化后验证准确率: {test_metric['accuracy']:.4f}")
                except ImportError:
                    logger.warning("无法导入NLItest模块，跳过最终验证")
        else:
            logger.warning("增强数据筛选后为空，跳过增强数据优化阶段")

        logger.info("========== 阶段3: 增强数据优化完成 ==========")
    else:
        logger.warning("未提供增强数据集，跳过增强数据优化阶段")

    logger.info("========== 先无标签后有标签训练流程完成 ==========")
    return model_instance


if __name__ == "__main__":
    logger.info("程序开始，初始化数据集与模型")

    # 初始化数据集
    from NLIdataset import NLIDataset
    dataset_handler = NLIDataset(dataset_name="snli",
                                 labeled_ratio=0.011,
                                 unlabeled_ratio=0.011,
                                 use_original_val=True,
                                 use_data_augmentation=False,
                                 llm_model_name="llama-3.2-3b-instruct")

    labeled_dataset = dataset_handler.get_labeled_dataset()
    # unlabeled_dataset = dataset_handler.get_unlabeled_dataset()
    validation_dataset = dataset_handler.get_validation_dataset()
    # augmented_dataset = dataset_handler.get_augmented_dataset()

    logger.info("数据集加载完成，准备初始化模型")
    model = NLIModel(model_name="bert-base-uncased")

    logger.info("模型初始化完成，开始训练")
    model.model = train_nli(
        model,
        labeled_dataset=labeled_dataset,
        validation_dataset=validation_dataset,
    )

    logger.info("训练流程结束，保存训练后的模型")
    model.save_model("nli_saved_model/train_test_model")
    logger.info("模型保存完成: nli_saved_model/train_test_model")
