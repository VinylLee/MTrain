from mylogging import logger
from NLI_train_one_epoch import train_one_epoch_RDrop, train_one_epoch_safe_MR_multiview, train_one_epoch_labeled_CE, train_one_epoch_three_consistency, train_one_epoch_triplet_consistency
from NLI_train_one_epoch import train_one_epoch_dual_consistency, train_one_epoch_pairwise_metamorphic, train_one_epoch_pairwise_dataloader, create_pairwise_dataloader_from_dataset

import torch
from tqdm import tqdm
from mylogging import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math  # Ensure math is imported
import os  # For path creation

# ============================== 验证函数 ==============================


def validate(device,
             model_instance,
             val_loader,
             prefix="") -> tuple[float, float]:
    """
    在验证集上评估模型
    
    参数:
    - device: 计算设备
    - model_instance: 模型实例
    - val_loader: 验证数据加载器
    - prefix: 日志前缀（可选）
    
    返回:
    - float: 验证集上的平均损失
    """
    model_instance.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # 只对tensor类型的数据移动到device，忽略其他类型（如字符串、列表等）
            batch_on_device = {}
            for k, v in batch.items():
                if hasattr(v, 'to') and callable(getattr(v, 'to')):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            # 从batch中移除模型不需要的额外参数，只保留模型forward方法需要的参数
            model_inputs = {}
            for k, v in batch_on_device.items():
                if k in [
                        'input_ids', 'attention_mask', 'labels',
                        'token_type_ids'
                ]:
                    model_inputs[k] = v

            outputs = model_instance(**model_inputs)
            val_loss += outputs.loss.item()

            # 收集预测值和真实标签
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(model_inputs["labels"].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"{prefix} 验证结束，平均损失: {avg_val_loss:.4f}")

    # 计算accuracy,f1分数,recall 和 precision
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")

    logger.info(f"{prefix} 验证集 Accuracy: {accuracy:.4f}")
    logger.info(f"{prefix} 验证集 Recall: {recall:.4f}")
    logger.info(f"{prefix} 验证集 Precision: {precision:.4f}")
    logger.info(f"{prefix} 验证集 F1 Score: {f1:.4f}")
    return avg_val_loss, accuracy


# ============================== 训练主函数 ==============================
import copy  # ★修改: 为深拷贝最优权重
import torch
from tqdm import tqdm


def train_with_labeled_data(
        model_instance,
        loaders,
        optimizer,
        device,
        epochs,
        patience,  # 早停容忍轮数
        alpha_init,  # 初始 CE 权重
        gamma):  # KL 权重
    """
    使用有标签数据训练模型（阶段一）

    参数:
    - model_instance: 模型实例
    - loaders: 包含各类 DataLoader 的字典 (需要 'labeled'，可选 'validation')
    - optimizer: 优化器
    - device: 计算设备
    - epochs: 总训练轮数
    - patience: 早停容忍轮数
    - alpha_init: CE 初始权重
    - gamma: KL 散度权重

    返回:
    - tuple: (加载最佳权重后的模型实例, 最佳验证损失, 最终 alpha)
    """
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    # ★修改: α 线性 warm-up（不再依赖验证集提升）
    current_alpha = alpha_init
    alpha_target = 0.3
    alpha_increment = (alpha_target - alpha_init) / max(1, epochs)

    for epoch in range(epochs):
        logger.info(f"======== 真实数据训练阶段 Epoch {epoch + 1} ========")

        # 训练一个 epoch
        # train_one_epoch_labeled(model_instance,
        #                         loaders["labeled"],
        #                         optimizer,
        #                         device,
        #                         epoch,
        #                         phase_name="真实数据训练KL",
        #                         alpha=current_alpha,
        #                         lam=gamma)
        train_one_epoch_labeled_CE(
            model_instance,
            loaders["labeled"],
            optimizer,
            device,
            epoch,
            phase_name="真实数据训练CE",
        )
        # train_one_epoch_RDrop(
        #     model_instance=model_instance,
        #     dataloader=loaders["labeled"],
        #     optimizer=optimizer,
        #     device=device,
        #     epoch_idx=epoch,
        #     phase_name="RDrop真实数据训练阶段",
        # )
        logger.info(f"======== 真实数据训练阶段 结束 Epoch {epoch + 1} ========")

        # ---------- 验证 ----------
        if "validation" not in loaders:
            logger.info("未提供验证集，跳过验证阶段。")
        else:
            logger.info(f"======== 真实数据验证阶段 Epoch {epoch + 1} ========")

            # ★修改: 关闭梯度以节约显存
            model_instance.eval()
            with torch.no_grad():
                avg_val_loss, _ = validate(device, model_instance,
                                           loaders["validation"], "真实数据验证阶段:")

            # 判断是否改善
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(
                    model_instance.state_dict())  # ★修改
                logger.info(f"验证损失改善 -> {best_val_loss:.4f}，保存当前最优权重。")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"验证损失未改善 ({epochs_without_improvement}/{patience})")

            logger.info(f"======== 真实数据验证阶段 结束 Epoch {epoch + 1} ========")

        # ---------- α warm-up ----------
        if current_alpha < alpha_target:
            current_alpha = min(current_alpha + alpha_increment, alpha_target)
        logger.info(f"Epoch {epoch + 1}: 当前 alpha = {current_alpha:.3f}")

        # ---------- 早停 ----------
        if epochs_without_improvement >= patience:
            logger.info(f"早停触发：验证损失已连续 {patience} 轮未改善，停止训练。")
            break

    # ---------- 训练结束 ----------
    if best_model_state is not None:
        model_instance.load_state_dict(best_model_state)
        logger.info(f"加载验证集最优权重 (损失: {best_val_loss:.4f})。")

    return model_instance, best_val_loss, current_alpha


def train_with_consistency(model_instance, loaders, optimizer, device, epochs,
                           patience, alpha_init):
    """
    使用一致性训练方法训练模型
    
    参数:
    - model_instance: 模型实例
    - loaders: 包含各种数据加载器的字典
    - optimizer: 优化器
    - device: 计算设备
    - epochs: 训练轮数
    - patience: 早停的容忍轮数
    - alpha_init: 初始CE权重
    - best_val_loss: 当前最佳验证损失
    
    返回:
    - tuple: (更新后的模型实例, 最佳验证损失, 当前alpha值, 当前MR-KL权重)
    """
    best_val_loss = float("inf")
    alpha = alpha_init
    lam_mr = 1.0
    epochs_without_improvement = 0

    if "augmented" not in loaders or loaders["augmented"] is None or len(
            loaders["augmented"]) == 0:
        logger.info("======== 未提供增强数据集，跳过一致性训练阶段 ========")
        return model_instance, best_val_loss, alpha_init, 1.0

    for epoch in range(epochs):
        logger.info(f"======== 一致性训练阶段 Epoch {epoch}========")
        train_one_epoch_safe_MR_multiview(model_instance,
                                          loaders["augmented"],
                                          optimizer,
                                          device,
                                          epoch,
                                          phase_name="一致性训练",
                                          alpha=alpha,
                                          lam=lam_mr)
        logger.info(f"======== 一致性训练阶段 结束 Epoch {epoch} ========")

        # 验证阶段
        logger.info(f"======== 一致性验证阶段 Epoch {epoch} ========")
        if "validation" not in loaders:
            logger.info("======== 未提供验证集，跳过一致性验证阶段 ========")
        else:
            avg_val_loss, _ = validate(device, model_instance,
                                       loaders["validation"], "一致性验证阶段:")

            # 检查验证集损失是否改善
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                logger.info(f"一致性验证阶段: 验证集损失改善，新最佳损失: {best_val_loss:.4f}")
                # 调整权重
                alpha = min(alpha + 0.05, 0.6)  # 上限 0.6
                lam_mr = min(lam_mr + 0.2, 2.0)
                logger.info(f"一致性验证阶段: 验证集损失改善，合成样本权重 α 调整为 {alpha:.2f}")
                logger.info(f"一致性验证阶段: 验证集损失改善，MR-KL 权重 λ 调整为 {lam_mr:.2f}")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"一致性验证阶段: 验证集损失未改善，轮数: {epochs_without_improvement}")

            # 检查是否触发早停
            if epochs_without_improvement >= patience:
                logger.info(f"一致性验证阶段: 早停触发！验证集损失连续 {patience} 轮未改善，停止训练。")
                break

        logger.info(f"======== 一致性验证阶段 结束 Epoch {epoch} ========")

    return model_instance, best_val_loss, alpha, lam_mr


def train_with_augment(model_instance,
                       loaders,
                       optimizer,
                       device,
                       epochs,
                       patience,
                       alpha_init,
                       lam_inv_init=1.0,
                       lam_flip_init=0.5):
    """
    使用双重一致性训练方法训练模型，并动态调整权重参数, 当前已改成无标签数据训练
    
    参数:
    - model_instance: 模型实例
    - loaders: 包含各种数据加载器的字典
    - optimizer: 优化器
    - device: 计算设备
    - epochs: 训练轮数
    - patience: 早停的容忍轮数，已废弃
    - alpha_init: 初始CE权重 (此阶段将被动态调度覆盖)
    - lam_inv_init: 标签保持KL一致性损失的初始权重
    - lam_flip_init: 标签翻转KL一致性损失的初始权重
    
    返回:
    - tuple: (更新后的模型实例, 最佳验证损失, 最终alpha值, 最终标签保持权重, 最终标签翻转权重)
    """

    best_val_loss = float("inf")
    best_model_state = None  # To save the best model state
    data_type = "augmented"  # unlabeled  augmented

    if data_type not in loaders or loaders[data_type] is None or len(
            loaders[data_type]) == 0:
        logger.info("======== 未提供增强数据集，跳过双重一致性训练阶段 ========")
        return model_instance, float(
            "inf"
        ), alpha_init, lam_inv_init, lam_flip_init  # Return initial values if skipped

    # Create ckpt_base directory if it doesn't exist
    ckpt_base_dir = "./ckpt_base"
    if not os.path.exists(ckpt_base_dir):
        os.makedirs(ckpt_base_dir)
        logger.info(f"已创建目录: {ckpt_base_dir}")

    for epoch in range(epochs):
        # alpha 从 0.5 到 0.8 线性调度，+是增加，-是减少
        current_alpha = 0.6 - 0.4 * (epoch / max(1, epochs - 1))
        # 余弦调度, + 是增加
        current_lam_flip = current_lam_inv = 0.01 + 0.75 * (epoch /
                                                            max(1, epochs - 1))

        logger.info(f"======== 增强数据集训练阶段 Epoch {epoch + 1}/{epochs} ========")
        logger.info(
            f"当前参数: alpha={current_alpha:.3f}, lam_inv={current_lam_inv:.3f}, lam_flip={current_lam_flip:.3f}"
        )
        train_one_epoch_RDrop(
            model_instance=model_instance,
            dataloader=loaders[data_type],
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            phase_name="RDrop增强数据集训练阶段",
        )
        # train_one_epoch_dual_consistency(
        #     model_instance,
        #     loaders[data_type],
        #     optimizer,
        #     device,
        #     epoch,
        #     phase_name="双重一致性训练",
        #     alpha=current_alpha,
        #     lam_inv=current_lam_inv,
        #     lam_flip=current_lam_flip,
        #     # 新增正则化参数
        #     dropout_rate=0.2,
        #     weight_decay_loss=0.01,
        #     label_smoothing=0.1,
        #     l2_reg_weight=0.001)
        # train_one_epoch_labeled_CE(
        #     model_instance,
        #     loaders[data_type],
        #     optimizer,
        #     device,
        #     epoch,
        #     phase_name="增强数据训练CE",
        # )
        logger.info(
            f"======== 增强数据集训练阶段 结束 Epoch {epoch + 1}/{epochs} ========")

        # 验证阶段
        logger.info(f"======== 增强数据集验证阶段 Epoch {epoch + 1}/{epochs} ========")
        if "validation" not in loaders:
            logger.info("======== 未提供验证集，跳过增强数据集验证阶段 ========")
        else:
            avg_val_loss, _ = validate(device, model_instance,
                                       loaders["validation"], "增强数据集验证阶段:")

            # 检查验证集损失是否改善
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"增强数据集验证阶段: 验证集损失改善，新最佳损失: {best_val_loss:.4f}")
                try:
                    best_model_state = model_instance.state_dict()
                    logger.info(
                        f"增强数据集验证阶段: 保存当前最优模型状态 (损失: {best_val_loss:.4f})")
                except Exception as e:
                    logger.error(f"保存模型状态失败: {e}")

        logger.info(
            f"======== 增强数据集验证阶段 结束 Epoch {epoch + 1}/{epochs} ========")

    # After all epochs, load the best model state if available
    if best_model_state is not None:
        logger.info(f"增强数据集训练结束，加载验证集上表现最优的模型状态 (损失: {best_val_loss:.4f})。")
        model_instance.load_state_dict(best_model_state)
    else:
        logger.warning("增强数据集训练结束，但没有找到最佳模型状态进行保存。")

    return model_instance, best_val_loss, current_alpha, current_lam_inv, current_lam_flip


def train_with_unlabeled_data(
        model_instance,
        loaders,
        optimizer,
        device,
        epochs,
        patience,  # 添加 patience 参数
        alpha_ul=0.3,
        lam_inv_ul=1.0,
        lam_flip_ul=0.5,
        lam_cont_ul=0.4,
        temperature_ul=0.1):
    """
    使用无标签数据训练模型（利用一致性和对比损失），并包含验证和早停逻辑
    
    参数:
    - model_instance: 模型实例
    - loaders: 包含各种数据加载器的字典
    - optimizer: 优化器
    - device: 计算设备
    - epochs: 训练轮数
    - patience: 早停的容忍轮数
    - alpha_ul: 无标签数据中合成样本CE损失的权重
    - lam_inv_ul: 无标签数据中标签保持KL一致性损失的权重
    - lam_flip_ul: 无标签数据中标签翻转KL一致性损失的权重
    - lam_cont_ul: 无标签数据中对比损失的权重
    - temperature_ul: 无标签数据中对比损失的温度参数
    
    返回:
    - tuple: (更新后的模型实例, 最佳验证损失, 当前alpha_ul, 当前lam_inv_ul, 当前lam_flip_ul, 当前lam_cont_ul)
    """
    data_type = "unlabeled"
    if data_type not in loaders or loaders[data_type] is None or len(
            loaders[data_type]) == 0:
        logger.info("======== 未提供无标签数据集，跳过无标签数据训练阶段 ========")
        return model_instance, float(
            "inf"), alpha_ul, lam_inv_ul, lam_flip_ul, lam_cont_ul

    logger.info("======== 开始无标签数据训练阶段 ========")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None  # 用于保存最佳模型状态
    epochs_without_improvement = 0
    # 将传入的超参数作为可变副本，以便在函数内部修改
    current_alpha_ul = alpha_ul
    current_lam_inv_ul = lam_inv_ul
    current_lam_flip_ul = lam_flip_ul
    current_lam_cont_ul = lam_cont_ul

    for epoch in range(epochs):
        # alpha 从 0.5 到 0.8 线性调度，+是增加，-是减少
        current_alpha_ul = 0.5 - 0.4 * (epoch / max(1, epochs - 1))
        # 余弦调度, + 是增加
        current_lam_flip_ul = current_lam_inv_ul = 0.01 + 0.6 * (
            epoch / max(1, epochs - 1))
        current_temp_ul = 0.2 + 0.1 * (epoch / max(1, epochs - 1))
        logger.info(
            f"当前参数: alpha_ul={current_alpha_ul:.2f}, lam_inv_ul={current_lam_inv_ul:.2f}, lam_flip_ul={current_lam_flip_ul:.2f}, lam_cont_ul={current_lam_cont_ul:.2f}, temperature_ul={current_temp_ul:.2f}"
        )
        train_one_epoch_RDrop(
            model_instance=model_instance,
            dataloader=loaders[data_type],
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            phase_name="RDrop无标签数据训练阶段",
        )
        # train_one_epoch_labeled_CE(
        #     model_instance,
        #     loaders[data_type],
        #     optimizer,
        #     device,
        #     epoch,
        #     phase_name="无标签数据训练CE",
        # )
        # train_one_epoch_dual_consistency(
        #     model_instance,
        #     loaders[data_type],
        #     optimizer,
        #     device,
        #     epoch,
        #     phase_name="双重一致训练",
        #     alpha=current_alpha_ul,
        #     lam_inv=current_lam_inv_ul,
        #     lam_flip=current_lam_flip_ul,
        #     # 新增正则化参数
        #     dropout_rate=0.2,
        #     weight_decay_loss=0.01,
        #     label_smoothing=0.1,
        #     l2_reg_weight=0.001)
        logger.info(
            f"======== 无标签数据训练阶段 Epoch {epoch + 1}/{epochs} 完成 ========")

        # 验证阶段
        logger.info(f"======== 无标签数据验证阶段 Epoch {epoch + 1}/{epochs} ========")
        if "validation" not in loaders:
            logger.info("======== 未提供验证集，跳过无标签数据验证阶段 ========")
        else:
            avg_val_loss, acc = validate(device, model_instance,
                                         loaders["validation"], "无标签数据验证阶段:")

            if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = copy.deepcopy(model_instance.state_dict())
                logger.info(f"无标签数据验证阶段: 验证集准确率改善，新最佳准确率: {best_val_acc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                logger.info(f"无标签数据验证阶段: 验证集损失改善，新最佳损失: {best_val_loss:.4f}")

                logger.info(
                    f"无标签数据验证阶段: 动态调整参数: alpha_ul={current_alpha_ul:.2f}, lam_inv_ul={current_lam_inv_ul:.2f}, lam_flip_ul={current_lam_flip_ul:.2f}, lam_cont_ul={current_lam_cont_ul:.2f}"
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"无标签数据验证阶段: 验证集损失未改善，轮数: {epochs_without_improvement}")

                # 如果连续多轮无改善，可以考虑调整策略，例如减小某些权重
                if epochs_without_improvement >= 2:  # 连续2轮无改善
                    current_lam_cont_ul = max(current_lam_cont_ul * 0.9,
                                              0.1)  # 尝试减少对比损失权重
                    current_lam_flip_ul = max(current_lam_flip_ul * 0.9,
                                              0.1)  # 尝试减少翻转KL损失权重
                    logger.info(
                        f"无标签数据验证阶段: 验证损失连续未改善，调整权重平衡: lam_flip_ul={current_lam_flip_ul:.2f}, lam_cont_ul={current_lam_cont_ul:.2f}"
                    )

            # if epochs_without_improvement >= patience:
            #     logger.info(
            #         f"无标签数据验证阶段: 早停触发！验证集损失连续 {patience} 轮未改善，停止无标签数据训练。")
            #     break

        logger.info(
            f"======== 无标签数据验证阶段 Epoch {epoch + 1}/{epochs} 结束 ========")

    logger.info("======== 无标签数据训练阶段完成 ========")
    if best_model_state is not None:
        logger.info(f"增强数据集训练结束，加载验证集上表现最优的模型状态 (ACC: {best_val_acc:.4f})。")
        model_instance.load_state_dict(best_model_state)
    else:
        logger.warning("增强数据集训练结束，但没有找到最佳模型状态进行保存。")
    return model_instance, best_val_loss, current_alpha_ul, current_lam_inv_ul, current_lam_flip_ul, current_lam_cont_ul


def train_with_metamorphic_relations(
        model_instance,
        augmented_dataset,  # 直接传入增强数据集
        tokenizer,  # 分词器
        val_loader,  # 验证数据加载器
        optimizer,
        device,
        epochs,
        patience=5,
        # 蜕变关系损失权重参数
        alpha_inv=1.0,  # inv-MR权重 (标签不变)
        alpha_flip=1.0,  # flip-MR权重 (标签翻转)
        alpha_neutral=0.8,  # neutral-MR权重 (标签变中性)
        beta_consistency=0.5,  # 一致性损失权重
        gamma_ce=1.0,  # 交叉熵损失权重
        # 其他参数
    temperature=1.0,
        label_smoothing=0.1,
        l2_reg_weight=1e-4,
        # 动态调整参数
        enable_dynamic_adjustment=True,  # 默认关闭动态调整
        save_best_model=True):
    """
    使用蜕变关系进行数据增强训练的完整训练函数
    
    参数:
    - model_instance: 模型实例
    - augmented_dataset: 包含源数据和蜕变数据的增强数据集
    - tokenizer: 分词器（用于文本编码）
    - val_loader: 验证数据加载器
    - optimizer: 优化器
    - device: 计算设备
    - epochs: 训练轮数
    - patience: 早停的容忍轮数
    - alpha_inv: 标签不变蜕变关系权重 (如同义词替换)
    - alpha_flip: 标签翻转蜕变关系权重 (如添加矛盾)
    - alpha_neutral: 标签变中性蜕变关系权重 (如添加无关信息)
    - beta_consistency: 一致性损失权重
    - gamma_ce: 交叉熵损失权重
    - temperature: KL散度温度参数
    - label_smoothing: 标签平滑参数
    - l2_reg_weight: L2正则化权重
    - enable_dynamic_adjustment: 是否启用动态权重调整
    - save_best_model: 是否保存最佳模型状态
    
    返回:
    - tuple: (更新后的模型实例, 最佳验证损失, 最佳验证准确率, 最终权重配置)
    """
    import copy
    from collections import defaultdict

    logger.info("======== 开始蜕变关系增强训练阶段 ========")
    logger.info(f"数据集大小: {len(augmented_dataset)}")

    # 分析数据集中的蜕变关系分布
    mr_type_counts = defaultdict(int)
    pair_id_counts = defaultdict(int)

    for i in range(len(augmented_dataset)):
        item = augmented_dataset[i]
        mr_type = item.get('mr_type', 'inv')
        mr_id = item.get('mr_id', 'none')
        pair_id = item.get('pair_id', i)

        mr_type_counts[mr_type] += 1
        pair_id_counts[pair_id] += 1

    logger.info(f"蜕变关系分布: {dict(mr_type_counts)}")
    logger.info(f"数据对组数: {len(pair_id_counts)}")
    logger.info(
        f"平均每组样本数: {sum(pair_id_counts.values()) / len(pair_id_counts):.2f}")

    # 初始化最佳模型追踪
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # 当前权重配置（用于动态调整）
    current_alpha_inv = alpha_inv
    current_alpha_flip = alpha_flip
    current_alpha_neutral = alpha_neutral
    current_beta_consistency = beta_consistency
    current_gamma_ce = gamma_ce
    current_temperature = temperature

    # 创建ckpt目录
    ckpt_dir = "./ckpt_base"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        logger.info(f"已创建模型保存目录: {ckpt_dir}")

    for epoch in range(epochs):
        logger.info(f"======== 蜕变增强训练 Epoch {epoch + 1}/{epochs} ========")

        # 动态权重调整策略
        if enable_dynamic_adjustment:
            # 温度参数：从1.0逐渐降到0.5，使一致性要求更严格
            current_temperature = 1.0 - 0.5 * (epoch / max(1, epochs - 1))

            # inv-MR权重：保持稳定，因为这是最重要的一致性
            current_alpha_inv = alpha_inv

            # flip-MR权重：逐渐增加，让模型更好地学习翻转关系
            current_alpha_flip = alpha_flip * (0.5 + 0.5 *
                                               (epoch / max(1, epochs - 1)))

            # neutral-MR权重：先增加后稳定
            if epoch < epochs // 2:
                current_alpha_neutral = alpha_neutral * (
                    0.3 + 0.7 * (epoch / max(1, epochs // 2)))
            else:
                current_alpha_neutral = alpha_neutral

            # 一致性损失权重：逐渐增加重要性
            current_beta_consistency = beta_consistency * (
                0.7 + 0.3 * (epoch / max(1, epochs - 1)))

        logger.info(f"当前权重配置:")
        logger.info(f"  - alpha_inv: {current_alpha_inv:.3f} (标签不变)")
        logger.info(f"  - alpha_flip: {current_alpha_flip:.3f} (标签翻转)")
        logger.info(f"  - alpha_neutral: {current_alpha_neutral:.3f} (标签变中性)")
        logger.info(
            f"  - beta_consistency: {current_beta_consistency:.3f} (一致性)")
        logger.info(f"  - gamma_ce: {current_gamma_ce:.3f} (交叉熵)")
        logger.info(f"  - temperature: {current_temperature:.3f}")

        # 训练一个epoch
        try:
            avg_loss = train_one_epoch_pairwise_metamorphic(
                model_instance=model_instance,
                dataset=augmented_dataset,
                tokenizer=tokenizer,
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch,
                phase_name="蜕变增强训练",
                alpha_inv=current_alpha_inv,
                alpha_flip=current_alpha_flip,
                alpha_neutral=current_alpha_neutral,
                beta_consistency=current_beta_consistency,
                gamma_ce=current_gamma_ce,
                temperature=current_temperature,
                label_smoothing=label_smoothing,
                l2_reg_weight=l2_reg_weight)

            logger.info(f"训练完成，平均损失: {avg_loss:.4f}")

        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            continue

        logger.info(f"======== 蜕变增强训练 Epoch {epoch + 1}/{epochs} 结束 ========")

        # 验证阶段
        if val_loader is not None:
            logger.info(f"======== 蜕变增强验证 Epoch {epoch + 1}/{epochs} ========")

            try:
                avg_val_loss, val_acc = validate(device=device,
                                                 model_instance=model_instance,
                                                 val_loader=val_loader,
                                                 prefix="蜕变增强验证:")

                # 检查是否有改善
                improved = False

                # 优先考虑准确率改善
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    improved = True
                    logger.info(f"验证准确率改善: {val_acc:.4f} > {best_val_acc:.4f}")

                # 其次考虑损失改善
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    improved = True
                    logger.info(
                        f"验证损失改善: {avg_val_loss:.4f} < {best_val_loss:.4f}")

                if improved:
                    epochs_without_improvement = 0

                    # 保存最佳模型状态
                    if save_best_model:
                        try:
                            best_model_state = copy.deepcopy(
                                model_instance.state_dict())
                            logger.info("已保存当前最佳模型状态")
                        except Exception as e:
                            logger.error(f"保存模型状态失败: {e}")

                    # 根据验证结果动态调整权重
                    if enable_dynamic_adjustment and epoch > 0:
                        # 如果准确率提升明显，可以增加一致性损失权重
                        if val_acc - best_val_acc > 0.01:
                            current_beta_consistency = min(
                                current_beta_consistency * 1.1, 1.0)
                            logger.info(
                                f"准确率显著提升，增加一致性权重至: {current_beta_consistency:.3f}"
                            )

                        # 如果损失下降明显，可以调整蜕变关系权重
                        if best_val_loss - avg_val_loss > 0.05:
                            current_alpha_neutral = min(
                                current_alpha_neutral * 1.05, 1.2)
                            logger.info(
                                f"损失显著下降，调整中性权重至: {current_alpha_neutral:.3f}")

                else:
                    epochs_without_improvement += 1
                    logger.info(
                        f"验证指标未改善 ({epochs_without_improvement}/{patience})")

                    # 如果连续多轮无改善，尝试调整权重
                    if enable_dynamic_adjustment and epochs_without_improvement >= 2:
                        # 减少过拟合风险：降低一致性权重，增加正则化
                        current_beta_consistency = max(
                            current_beta_consistency * 0.95, 0.2)
                        current_temperature = min(current_temperature * 1.05,
                                                  1.5)
                        logger.info(
                            f"连续无改善，调整权重: beta={current_beta_consistency:.3f}, temp={current_temperature:.3f}"
                        )

                logger.info(
                    f"======== 蜕变增强验证 Epoch {epoch + 1}/{epochs} 结束 ========")

                # 早停检查
                if epochs_without_improvement >= patience:
                    logger.info(f"早停触发: 验证指标连续 {patience} 轮未改善，停止训练")
                    break

            except Exception as e:
                logger.error(f"验证过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info("未提供验证集，跳过验证阶段")

    # 训练结束处理
    logger.info("======== 蜕变关系增强训练阶段完成 ========")

    if best_model_state is not None and save_best_model:
        logger.info(
            f"加载最佳模型状态 (准确率: {best_val_acc:.4f}, 损失: {best_val_loss:.4f})")
        model_instance.load_state_dict(best_model_state)
    else:
        logger.warning("未找到最佳模型状态或未启用模型保存")

    # 返回最终权重配置
    final_weights = {
        'alpha_inv': current_alpha_inv,
        'alpha_flip': current_alpha_flip,
        'alpha_neutral': current_alpha_neutral,
        'beta_consistency': current_beta_consistency,
        'gamma_ce': current_gamma_ce,
        'temperature': current_temperature
    }

    logger.info(f"最终权重配置: {final_weights}")
    logger.info(f"最佳验证准确率: {best_val_acc:.4f}")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")

    return model_instance, best_val_loss, best_val_acc, final_weights


def train_with_pairwise_dataloader(
        model_instance,
        augmented_dataset,
        tokenizer,
        val_loader,
        optimizer,
        device,
        epochs,
        patience=5,
        max_pairs_per_batch=4,  # 每个batch最大包含的pair数量
        # 其他参数与上面函数相同
    alpha_inv=1.0,
        alpha_flip=1.0,
        alpha_neutral=0.8,
        beta_consistency=0.5,
        gamma_ce=1.0,
        temperature=1.0,
        label_smoothing=0.1,
        l2_reg_weight=1e-4,
        enable_dynamic_adjustment=False,  # 默认关闭动态调整
        save_best_model=True):
    """
    使用pairwise dataloader的蜕变关系训练函数（替代方案）
    
    这个函数使用预处理的pairwise dataloader，可能在某些情况下更高效
    """
    import copy

    logger.info("======== 开始Pairwise蜕变关系训练阶段 ========")

    # 创建pairwise dataloader
    logger.info("创建pairwise数据加载器...")
    try:
        pairwise_loader = create_pairwise_dataloader_from_dataset(
            dataset=augmented_dataset,
            tokenizer=tokenizer,
            max_pairs_per_batch=max_pairs_per_batch)
        logger.info(f"成功创建pairwise数据加载器，每batch最多{max_pairs_per_batch}个数据对组")
    except Exception as e:
        logger.error(f"创建pairwise数据加载器失败: {e}")
        # 回退到直接使用dataset的方法
        return train_with_metamorphic_relations(
            model_instance, augmented_dataset, tokenizer, val_loader,
            optimizer, device, epochs, patience, alpha_inv, alpha_flip,
            alpha_neutral, beta_consistency, gamma_ce, temperature,
            label_smoothing, l2_reg_weight, enable_dynamic_adjustment,
            save_best_model)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # 当前权重配置
    current_alpha_inv = alpha_inv
    current_alpha_flip = alpha_flip
    current_alpha_neutral = alpha_neutral
    current_beta_consistency = beta_consistency
    current_gamma_ce = gamma_ce
    current_temperature = temperature

    for epoch in range(epochs):
        logger.info(
            f"======== Pairwise蜕变训练 Epoch {epoch + 1}/{epochs} ========")

        # 动态权重调整（与上面函数相同的策略）
        if enable_dynamic_adjustment:
            current_temperature = 1.0 - 0.5 * (epoch / max(1, epochs - 1))
            current_alpha_flip = alpha_flip * (0.5 + 0.5 *
                                               (epoch / max(1, epochs - 1)))
            if epoch < epochs // 2:
                current_alpha_neutral = alpha_neutral * (
                    0.3 + 0.7 * (epoch / max(1, epochs // 2)))
            else:
                current_alpha_neutral = alpha_neutral
            current_beta_consistency = beta_consistency * (
                0.7 + 0.3 * (epoch / max(1, epochs - 1)))

        logger.info(
            f"权重配置: inv={current_alpha_inv:.3f}, flip={current_alpha_flip:.3f}, "
            f"neutral={current_alpha_neutral:.3f}, consistency={current_beta_consistency:.3f}"
        )

        # 训练一个epoch
        try:
            avg_loss = train_one_epoch_pairwise_dataloader(
                model_instance=model_instance,
                pairwise_dataloader=pairwise_loader,
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch,
                phase_name="Pairwise蜕变训练",
                alpha_inv=current_alpha_inv,
                alpha_flip=current_alpha_flip,
                alpha_neutral=current_alpha_neutral,
                beta_consistency=current_beta_consistency,
                gamma_ce=current_gamma_ce,
                temperature=current_temperature,
                label_smoothing=label_smoothing,
                l2_reg_weight=l2_reg_weight)

            logger.info(f"训练完成，平均损失: {avg_loss:.4f}")

        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            continue

        # 验证阶段（与上面函数相同的逻辑）
        if val_loader is not None:
            logger.info(
                f"======== Pairwise蜕变验证 Epoch {epoch + 1}/{epochs} ========")

            try:
                avg_val_loss, val_acc = validate(device, model_instance,
                                                 val_loader, "Pairwise验证:")

                improved = False
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    improved = True
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    improved = True

                if improved:
                    epochs_without_improvement = 0
                    if save_best_model:
                        best_model_state = copy.deepcopy(
                            model_instance.state_dict())
                        logger.info("已保存最佳模型状态")
                else:
                    epochs_without_improvement += 1
                    logger.info(
                        f"验证指标未改善 ({epochs_without_improvement}/{patience})")

                if epochs_without_improvement >= patience:
                    logger.info("早停触发，停止训练")
                    break

            except Exception as e:
                logger.error(f"验证过程中出现错误: {e}")

    # 训练结束处理
    logger.info("======== Pairwise蜕变关系训练阶段完成 ========")

    if best_model_state is not None and save_best_model:
        logger.info(
            f"加载最佳模型状态 (准确率: {best_val_acc:.4f}, 损失: {best_val_loss:.4f})")
        model_instance.load_state_dict(best_model_state)

    final_weights = {
        'alpha_inv': current_alpha_inv,
        'alpha_flip': current_alpha_flip,
        'alpha_neutral': current_alpha_neutral,
        'beta_consistency': current_beta_consistency,
        'gamma_ce': current_gamma_ce,
        'temperature': current_temperature
    }

    return model_instance, best_val_loss, best_val_acc, final_weights
