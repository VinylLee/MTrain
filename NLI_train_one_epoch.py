import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mylogging import logger
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict


def train_one_epoch_labeled_CE(model_instance,
                               dataloader,
                               optimizer,
                               device,
                               epoch_idx,
                               phase_name="",
                               scheduler=None):
    """
    使用普通交叉熵损失训练一个epoch
    
    参数:
    - model_instance: 模型实例
    - dataloader: 数据加载器
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    
    返回:
    - float: 平均损失值
    """
    model_instance.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad(set_to_none=True)

        # 数据准备
        _ = batch.pop("is_syn", 0)
        _ = batch.pop("pair_id", 0)
        _ = batch.pop("mr_type", 0)
        batch = {
            k:
            v.to(device)
            if hasattr(v, 'to') and callable(getattr(v, 'to')) else v
            for k, v in batch.items()
        }

        # 单次前向传播
        outputs = model_instance(**batch)

        # 使用模型内置的交叉熵损失
        loss = outputs.loss

        # 检查损失是否有效
        if not torch.isnan(loss) and loss.requires_grad:
            loss.backward()
            # 梯度裁剪
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
        else:
            logger.warning(
                f"无效的损失值: {loss.item() if not torch.isnan(loss) else 'NaN'}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 结束，平均损失: {avg_loss:.4f}")
    return avg_loss


def train_one_epoch_RDrop(model_instance,
                          dataloader,
                          optimizer,
                          device,
                          epoch_idx,
                          phase_name="",
                          alpha=0.5,
                          lam=1.0,
                          label_smoothing=0.1,
                          scheduler=None):
    """
    使用有标签数据训练一个epoch
    
    参数:
    - model_instance: 模型实例
    - dataloader: 数据加载器
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - alpha: 第二次CE损失的权重（0~1, 默认0.5表示与第一次取平均）
    - lam: KL散度损失的权重
    - label_smoothing: 交叉熵损失的标签平滑系数
    
    返回:
    - float: 平均损失值
    """
    model_instance.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad(set_to_none=True)

        # 数据准备
        _ = batch.pop("is_syn", 0)
        _ = batch.pop("pair_id", 0)
        _ = batch.pop("mr_type", 0)
        batch = {
            k:
            v.to(device)
            if hasattr(v, 'to') and callable(getattr(v, 'to')) else v
            for k, v in batch.items()
        }

        # 两次前向传播（用于 R-Drop）
        outputs1 = model_instance(**batch)
        outputs2 = model_instance(**batch)

        # 计算各个损失分量
        logits1 = outputs1.logits
        logits2 = outputs2.logits
        labels = batch["labels"]

        ce_first = F.cross_entropy(logits1,
                                   labels,
                                   label_smoothing=label_smoothing)
        ce_second = F.cross_entropy(logits2,
                                    labels,
                                    label_smoothing=label_smoothing)

        # 对称 KL：让两次前向预测彼此接近
        log_p1 = F.log_softmax(logits1, dim=-1)
        log_p2 = F.log_softmax(logits2, dim=-1)
        prob2 = F.softmax(logits2, dim=-1)
        prob1 = F.softmax(logits1, dim=-1)

        kl_forward = F.kl_div(log_p1,
                              prob2,
                              reduction='batchmean',
                              log_target=False)
        kl_backward = F.kl_div(log_p2,
                               prob1,
                               reduction='batchmean',
                               log_target=False)
        kl_loss = 0.5 * (kl_forward + kl_backward)

        # 组合损失：两次CE按alpha加权平均 + KL正则
        ce_mix = (1 - alpha) * ce_first + alpha * ce_second
        loss = ce_mix + lam * kl_loss

        # 检查损失是否有效
        if not torch.isnan(loss) and loss.requires_grad:
            loss.backward()
            # 梯度裁剪
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
        else:
            logger.warning(
                f"无效的损失值: {loss.item() if not torch.isnan(loss) else 'NaN'}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 结束，平均损失: {avg_loss:.4f}")
    return avg_loss


def train_one_epoch_safe_MR_multiview(model_instance,
                                      dataloader,
                                      optimizer,
                                      device,
                                      epoch_idx,
                                      phase_name="",
                                      alpha=0.3,
                                      lam=1.0,
                                      scheduler=None):
    """
    使用蜕变一致性训练一个epoch
    
    参数:
    - model_instance: 模型实例
    - dataloader: 包含原始和蜕变样本的数据加载器
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - alpha: 合成样本CE损失的权重
    - lam: MR-KL损失的权重
    
    返回:
    - float: 平均损失值
    """
    model_instance.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"{phase_name} {epoch_idx + 1}"):
        optimizer.zero_grad(set_to_none=True)

        # 提取和处理数据
        is_syn = batch.pop("is_syn").to(device)
        pair_id = batch.pop("pair_id")
        mr_type = batch.pop("mr_type")  # 提取蜕变类型："inv"或"flip"
        labels = batch["labels"].to(device)

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 提取原始样本的 forward 输出
        orig_mask = ~is_syn
        orig_inputs = {k: v[orig_mask] for k, v in batch.items()}
        orig_pair_ids = [
            pair_id[i] for i in range(len(pair_id)) if not is_syn[i]
        ]

        orig_outputs = model_instance(**orig_inputs)
        ce_loss = orig_outputs.loss

        # 存储原始 logits 映射（pair_id -> logits）
        orig_logits_dict = {
            pid: orig_outputs.logits[i]
            for i, pid in enumerate(orig_pair_ids)
        }

        # 初始化 KL loss 累加
        total_kl = 0.0
        count = 0

        # 遍历蜕变样本逐个对齐原样本做 KL
        for i in range(len(pair_id)):
            if is_syn[i]:  # 如果是蜕变样本
                pid = pair_id[i]
                if pid in orig_logits_dict:
                    syn_input = {
                        k: v[i].unsqueeze(0)
                        for k, v in batch.items()
                    }
                    syn_output = model_instance(**syn_input)

                    kl = F.kl_div(F.log_softmax(
                        orig_logits_dict[pid].unsqueeze(0), dim=-1),
                                  F.softmax(syn_output.logits, dim=-1),
                                  reduction="batchmean")
                    total_kl += kl
                    count += 1

        kl_loss = total_kl / count if count > 0 else torch.tensor(0.0).to(
            device)

        # 计算蜕变样本的 CE（可选）
        syn_mask = is_syn
        if syn_mask.any():
            syn_inputs = {k: v[syn_mask] for k, v in batch.items()}
            syn_outputs = model_instance(**syn_inputs)
            ce_loss_syn = syn_outputs.loss
        else:
            ce_loss_syn = torch.tensor(0.0).to(device)

        # 总损失
        loss = ce_loss + alpha * ce_loss_syn + lam * kl_loss
        # 后向传播
        if not torch.isnan(loss) and loss.requires_grad:
            loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
        else:
            logger.warning(
                f"无效的损失值: {loss.item() if not torch.isnan(loss) else 'NaN'}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 完成，平均损失: {avg_loss:.4f}")
    return avg_loss


def train_one_epoch_three_consistency(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="",
        alpha=0.3,
        lam_inv=1.0,
        lam_flip=1.0,
        # neutral-specific coeffs ↓↓↓
        lam_neu_ent=0.1,  # weight for entropy term
        lam_neu_margin=0.1,  # weight for margin term
        delta_margin=0.10,  # margin threshold |p_e - p_c| ≤ δ
        # generic regularisers ↓↓↓
    dropout_rate=0.3,
        weight_decay_loss=0.01,
        label_smoothing=0.1,
        l2_reg_weight=0.001):
    """
    Train one epoch with dual consistency + neutral-class regularisation.

    Neutral-class extras
    --------------------
    * Entropy regulariser  :  -lam_neu_ent · H(p(x))                (maximise entropy)
    * Margin regulariser   :   lam_neu_margin · ReLU(|p_e-p_c|-δ)   (keep extremes close)

    All other arguments keep the same semantics as the original routine.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    from tqdm import tqdm

    phase_name = "三重一致性训练"
    model_instance.train()
    total_loss = 0.0

    # 1. Dynamic dropout (unchanged) -------------------------------------------------
    current_dropout = max(dropout_rate * (1.0 - epoch_idx / 20.0), 0.1)
    for m in model_instance.modules():
        if isinstance(m, nn.Dropout):
            m.p = current_dropout
    logger.info(f"{phase_name}: current dropout = {current_dropout:.3f}")

    # 2. Label-flip matrix (unchanged) ----------------------------------------------
    label_flip_matrix = torch.tensor(
        [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]], device=device)

    # 3. Smoothed CE loss ------------------------------------------------------------
    ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------------------------------------------------------------------------------
    for batch in tqdm(dataloader, desc=f"{phase_name} {epoch_idx+1}"):
        optimizer.zero_grad()

        # --- unpack ---------------------------------------------------------------
        is_syn = batch.pop("is_syn").to(device)  # bool
        pair_id = batch.pop("pair_id")
        mr_type = batch.pop("mr_type")  # ["inv" | "flip"]
        labels = batch["labels"].to(device)  # tensor[N]

        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        # --- forward on ORIGINAL samples -----------------------------------------
        orig_mask = ~is_syn
        orig_inputs = {k: v[orig_mask] for k, v in batch.items()}
        orig_pair_ids = [
            pair_id[i] for i in range(len(pair_id)) if not is_syn[i]
        ]
        orig_outputs = model_instance(**orig_inputs)

        ce_loss = (ce_fn(orig_outputs.logits, orig_inputs["labels"])
                   if orig_outputs.logits.size(0) else torch.tensor(
                       0., device=device))

        # map pid -> logits for later KL
        orig_logits_dict = {
            pid: orig_outputs.logits[i]
            for i, pid in enumerate(orig_pair_ids)
        }

        # --- dual consistency KL for synthetic samples ---------------------------
        total_kl_inv = total_kl_flip = 0.
        count_inv = count_flip = 0

        for i in range(len(pair_id)):
            if is_syn[i]:
                pid = pair_id[i]
                if pid in orig_logits_dict:
                    syn_input = {
                        k: v[i].unsqueeze(0)
                        for k, v in batch.items()
                    }
                    syn_output = model_instance(**syn_input)

                    if mr_type[i] == "inv":
                        kl = F.kl_div(F.log_softmax(
                            orig_logits_dict[pid].unsqueeze(0), dim=-1),
                                      F.softmax(syn_output.logits, dim=-1),
                                      reduction="batchmean")
                        total_kl_inv += kl
                        count_inv += 1

                    elif mr_type[i] == "flip":
                        orig_probs = F.softmax(
                            orig_logits_dict[pid].unsqueeze(0), dim=-1)
                        mapped_prob = orig_probs @ label_flip_matrix
                        kl = F.kl_div(mapped_prob,
                                      F.log_softmax(syn_output.logits, dim=-1),
                                      reduction="batchmean")
                        total_kl_flip += kl
                        count_flip += 1

        kl_loss_inv = total_kl_inv / count_inv if count_inv else torch.tensor(
            0., device=device)
        kl_loss_flip = total_kl_flip / count_flip if count_flip else torch.tensor(
            0., device=device)

        # --- CE on synthetic ------------------------------------------------------
        syn_mask = is_syn
        if syn_mask.any():
            syn_inputs = {k: v[syn_mask] for k, v in batch.items()}
            syn_outputs = model_instance(**syn_inputs)
            ce_loss_syn = ce_fn(syn_outputs.logits, syn_inputs["labels"])
        else:
            ce_loss_syn = torch.tensor(0., device=device)

        # -------------------------------------------------------------------------
        # 4. Neutral-class REGULARISERS  (entropy + margin) ------------------------
        #
        # helper to accumulate from both original and synthetic sets
        def neutral_regularisers(logits, lbls):
            """Return (entropy_term, margin_term)."""
            neu_mask = (lbls == 1)
            if not neu_mask.any():
                return torch.tensor(0.,
                                    device=device), torch.tensor(0.,
                                                                 device=device)

            neu_logits = logits[neu_mask]
            probs = F.softmax(neu_logits, dim=-1)  # [B,3]
            # --- entropy (want HIGH) --------------------------
            entropy = (-probs * probs.log()).sum(dim=-1).mean()  # scalar
            ent_term = -lam_neu_ent * entropy  # maximise entropy
            # --- margin (want SMALL) --------------------------
            p_e, p_c = probs[:, 0], probs[:, 2]
            margin = (p_e - p_c).abs() - delta_margin
            margin_loss = F.relu(margin).mean()  # scalar
            margin_term = lam_neu_margin * margin_loss
            return ent_term, margin_term

        ent_orig, mar_orig = neutral_regularisers(orig_outputs.logits,
                                                  orig_inputs["labels"])
        ent_syn, mar_syn = (neutral_regularisers(
            syn_outputs.logits, syn_inputs["labels"]) if syn_mask.any() else
                            (torch.tensor(0., device=device),
                             torch.tensor(0., device=device)))

        # -------------------------------------------------------------------------
        # 5. Parameter regularisers (L2 + weight-decay) ---------------------------
        l2_reg = torch.stack(
            [torch.norm(p) for p in model_instance.parameters()]).sum()
        wd_reg = torch.stack([
            torch.norm(p, p=2) for n, p in model_instance.named_parameters()
            if ('weight' in n and 'classifier' in n)
        ]).sum()

        # -------------------------------------------------------------------------
        # 6. Total loss -----------------------------------------------------------
        loss = (
            ce_loss + alpha * ce_loss_syn + lam_inv * kl_loss_inv +
            lam_flip * kl_loss_flip + ent_orig + ent_syn  # neutral entropy
            + mar_orig + mar_syn  # neutral margin
            + l2_reg_weight * l2_reg + weight_decay_loss * wd_reg)

        # --- backward & update ---------------------------------------------------
        if not torch.isnan(loss) and loss.requires_grad:
            loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()
        else:
            logger.warning(f"Invalid loss encountered: {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    logger.info(
        f"{phase_name} Epoch {epoch_idx+1} | avg loss = {avg_loss:.4f}")
    logger.info(f"{phase_name}: L2={l2_reg_weight}  WD={weight_decay_loss}  "
                f"Neu-ent={lam_neu_ent}  Neu-margin={lam_neu_margin}")
    return avg_loss


def train_one_epoch_triplet_consistency(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="",
        # ------- Triplet & Neutral ----------
        margin=0.5,  # Triplet margin
        lam_triplet=1.0,  # Triplet loss 权重
        lam_neu_push=0.5,  # Neutral-push 权重
        # ------- 伪标签 CE（可关）----------
    use_ce=True,
        lam_ce=0.2,
        label_smoothing=0.1,
        # ------- 其他正则 -------------------
        dropout_rate=0.3,
        l2_reg_weight=1e-3,
        weight_decay_loss=1e-2):
    """
    纯无标签+蜕变场景的 Triplet–Consistency 训练循环.

    dataloader 每条样本字段：
      - input_ids / attention_mask ...（模型输入张量）
      - 'premise', 'hypothesis' (可忽略, 仅供调试)
      - 'label'   : 0=ent, 1=neu, 2=con         # 伪标签 or 真标签
      - 'pair_id' : 同一原句对及其蜕变样本共用
      - 'is_syn'  : bool,  False→Anchor, True→蜕变
      - 'weight'  : 样本权重 (float)，可用于伪标签 CE
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    from tqdm import tqdm

    # ---------------- Helper：取 L2 归一化向量 ----------------
    def _embedding(outputs):
        if hasattr(outputs,
                   "pooler_output") and outputs.pooler_output is not None:
            vec = outputs.pooler_output  # [B,hidden]
        else:  # 退化用 CLS
            vec = outputs.hidden_states[-1][:, 0]
        return F.normalize(vec, p=2, dim=-1)

    model_instance.train()
    total_loss = 0.0

    # 1. 动态 dropout -----------------------------------------------------------
    cur_dp = max(dropout_rate * (1.0 - epoch_idx / 20.0), 0.1)
    for m in model_instance.modules():
        if isinstance(m, nn.Dropout):
            m.p = cur_dp
    logger.info(f"{phase_name}: dropout={cur_dp:.3f}")

    ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                reduction="none")

    # 2. 逐 batch ---------------------------------------------------------------
    for batch in tqdm(dataloader, desc=f"{phase_name} {epoch_idx+1}"):
        optimizer.zero_grad()

        # ---------- 设备搬运 & 元信息拆包 ----------------
        labels = batch.pop("labels").to(device)  # tensor[N]
        pair_id = batch.pop("pair_id")  # list[str/int]
        is_syn = batch.pop("is_syn").to(device)  # bool mask
        mr_type = batch.pop("mr_type")  # list[str] - 移除mr_type避免传递给模型

        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        # ---------- forward ----------------------------
        outputs = model_instance(**batch, output_hidden_states=True)
        embeds = _embedding(outputs)  # [N,D]

        # ---------- 构造 Anchor / Pos / Neg / Neu -------
        anchor_dict, pos_dict, neg_dict, neu_dict = {}, {}, {}, {}
        for i, pid in enumerate(pair_id):
            lbl = labels[i].item()
            if not is_syn[i]:  # 原始样本 ≡ Anchor
                anchor_dict[pid] = embeds[i]
            else:  # 蜕变样本
                if lbl == 0: pos_dict.setdefault(pid, []).append(embeds[i])
                elif lbl == 2: neg_dict.setdefault(pid, []).append(embeds[i])
                else: neu_dict.setdefault(pid, []).append(embeds[i])

        # ---------- Triplet & Neutral-push ------------
        t_losses, n_losses = [], []
        for pid, anchor in anchor_dict.items():
            if pid in pos_dict and pid in neg_dict:
                pos = pos_dict[pid][0]
                neg = neg_dict[pid][0]
                t = F.triplet_margin_loss(anchor.unsqueeze(0),
                                          pos.unsqueeze(0),
                                          neg.unsqueeze(0),
                                          margin=margin,
                                          p=2)
                t_losses.append(t)
            if pid in pos_dict and pid in neu_dict:
                pos = pos_dict[pid][0]
                neu = neu_dict[pid][0]
                d_ap = (anchor - pos).pow(2).sum().sqrt()
                d_an = (anchor - neu).pow(2).sum().sqrt()
                n = F.relu(d_ap - d_an)  # neutral 不应比 positive 更近
                n_losses.append(n)

        triplet_loss = torch.stack(
            t_losses).mean() if t_losses else torch.tensor(0., device=device)
        neutral_loss = torch.stack(
            n_losses).mean() if n_losses else torch.tensor(0., device=device)

        # ---------- 伪标签 CE（可关） --------------------
        if use_ce:
            ce = ce_fn(outputs.logits, labels)  # [N]
            ce_loss = ce.mean()
        else:
            ce_loss = torch.tensor(0., device=device)

        # ---------- 参数正则 ----------------------------
        l2_reg = torch.stack(
            [torch.norm(p) for p in model_instance.parameters()]).sum()
        wd_reg = torch.stack([
            torch.norm(p, p=2) for n, p in model_instance.named_parameters()
            if 'weight' in n
        ]).sum()

        # ---------- 总损失 ------------------------------
        loss = (lam_triplet * triplet_loss + lam_neu_push * neutral_loss +
                lam_ce * ce_loss + l2_reg_weight * l2_reg +
                weight_decay_loss * wd_reg)

        if not torch.isnan(loss) and loss.requires_grad:
            loss.backward()
            clip_grad_norm_(model_instance.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        else:
            logger.warning(f"Invalid loss: {loss.item()}")

    avg = total_loss / len(dataloader)
    logger.info(f"{phase_name} Epoch {epoch_idx+1} | avg loss={avg:.4f}")
    logger.info(
        f"{phase_name}: λ_triplet={lam_triplet}  λ_neu={lam_neu_push}  λ_CE={lam_ce}"
    )
    return avg


def train_one_epoch_dual_consistency(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="",
        alpha=0.3,
        lam_inv=1.0,
        lam_flip=1.0,
        # 新增正则化参数
        dropout_rate=0.3,
        weight_decay_loss=0.01,
        label_smoothing=0.1,
        l2_reg_weight=0.001,
        scheduler=None):
    """
    使用双重一致性训练一个epoch，并引入更强的正则化技术
    
    修改说明：
    - 将同一批数据（具有相同pair_id）一起送入同一个batch进行训练
    - 通过一次性前向传播处理所有样本，提高计算效率
    - 利用索引映射来关联原始样本和蜕变样本
    
    参数:
    - model_instance: 模型实例
    - dataloader: 包含原始和蜕变样本的数据加载器
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - alpha: 合成样本CE损失的权重
    - lam_inv: 标签保持KL一致性损失的权重
    - lam_flip: 标签翻转KL一致性损失的权重
    - dropout_rate: 动态dropout概率
    - weight_decay_loss: 权重衰减损失权重
    - label_smoothing: 标签平滑参数
    - l2_reg_weight: L2正则化权重
    
    返回:
    - float: 平均损失值
    """
    import torch.nn as nn
    model_instance.train()
    total_loss = 0.0

    # 动态调整dropout率
    current_dropout = dropout_rate * (1.0 - epoch_idx / 20.0)
    current_dropout = max(current_dropout, 0.1)
    for module in model_instance.modules():
        if isinstance(module, nn.Dropout):
            module.p = current_dropout
    if logger:
        logger.info(f"{phase_name}: 当前dropout率: {current_dropout:.3f}")
    else:
        print(f"{phase_name}: 当前dropout率: {current_dropout:.3f}")

    # 标签翻转映射矩阵（NLI三分类：蕴含0，中性1，矛盾2）
    label_flip_matrix = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # 蕴含 -> 矛盾
            [0.0, 1.0, 0.0],  # 中性 -> 中性
            [1.0, 0.0, 0.0]  # 矛盾 -> 蕴含
        ],
        device=device)

    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    logit_clip_val = 20.0

    def clip_logits(logits, max_abs=logit_clip_val):
        return torch.clamp(logits, -max_abs, max_abs)

    for batch in tqdm(dataloader, desc=f"{phase_name} {epoch_idx + 1}"):
        optimizer.zero_grad(set_to_none=True)

        # 数据准备
        is_syn = batch.pop("is_syn").to(device)
        pair_id = batch.pop("pair_id")
        mr_type = batch.pop("mr_type")

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 一次性处理所有数据（原始+蜕变），获得所有logits
        all_outputs = model_instance(**batch)
        all_logits = clip_logits(all_outputs.logits)
        all_labels = batch["labels"]

        # 分离原始样本和蜕变样本的索引
        orig_indices = []
        syn_indices = []
        for i, is_s in enumerate(is_syn):
            if is_s:
                syn_indices.append(i)
            else:
                orig_indices.append(i)

        # 计算原始样本的CE损失
        if len(orig_indices) > 0:
            orig_logits = all_logits[orig_indices]
            orig_labels = all_labels[orig_indices]
            ce_loss = label_smooth_loss(orig_logits, orig_labels)
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # 建立pair_id到原始样本索引的映射
        orig_pair_to_idx = {}
        for local_idx, global_idx in enumerate(orig_indices):
            pid = pair_id[global_idx]
            orig_pair_to_idx[pid] = local_idx

        # 批量计算KL一致性损失
        kl_inv_losses = []
        kl_flip_losses = []

        for syn_idx in syn_indices:
            pid = pair_id[syn_idx]
            if pid in orig_pair_to_idx:
                orig_local_idx = orig_pair_to_idx[pid]
                orig_logits_single = all_logits[orig_indices[orig_local_idx]]
                syn_logits_single = all_logits[syn_idx]

                if mr_type[syn_idx] == "inv":
                    # KL(σ(f(MR_inv(x))) || σ(f(x)))
                    kl = F.kl_div(F.log_softmax(syn_logits_single.unsqueeze(0),
                                                dim=-1),
                                  F.softmax(orig_logits_single.unsqueeze(0),
                                            dim=-1),
                                  reduction="batchmean",
                                  log_target=False)
                    kl_inv_losses.append(kl)

                elif mr_type[syn_idx] == "flip":
                    # 标签翻转：原始分布经过映射后与蜕变分布计算KL
                    orig_probs = F.softmax(orig_logits_single.unsqueeze(0),
                                           dim=-1)
                    mapped_probs = torch.matmul(orig_probs, label_flip_matrix)
                    kl = F.kl_div(F.log_softmax(syn_logits_single.unsqueeze(0),
                                                dim=-1),
                                  mapped_probs,
                                  reduction="batchmean",
                                  log_target=False)
                    kl_flip_losses.append(kl)

        # 计算平均KL损失
        kl_loss_inv = torch.stack(kl_inv_losses).mean(
        ) if kl_inv_losses else torch.tensor(0.0, device=device)
        kl_loss_flip = torch.stack(kl_flip_losses).mean(
        ) if kl_flip_losses else torch.tensor(0.0, device=device)

        # 计算蜕变样本的CE损失
        if len(syn_indices) > 0:
            syn_logits = all_logits[syn_indices]
            syn_labels = all_labels[syn_indices]
            ce_loss_syn = label_smooth_loss(syn_logits, syn_labels)
        else:
            ce_loss_syn = torch.tensor(0.0, device=device)

        # 总损失
        loss = (ce_loss + alpha * ce_loss_syn + lam_inv * kl_loss_inv +
                lam_flip * kl_loss_flip)

        # 数值异常监控
        if torch.isnan(loss) or not loss.requires_grad:
            msg = f"无效的损失值: {loss.item() if not torch.isnan(loss) else 'NaN'}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            continue

        # 反向传播与梯度裁剪
        loss.backward()
        max_grad_norm = 1.0
        clip_grad_norm_(model_instance.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

        # 梯度监控（可选）
        for n, p in model_instance.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN in gradient: {n}")

    avg_loss = total_loss / len(dataloader)
    if logger:
        logger.info(
            f"{phase_name} Epoch {epoch_idx+1} 完成，平均损失: {avg_loss:.4f}")
    else:
        print(f"{phase_name} Epoch {epoch_idx+1} 完成，平均损失: {avg_loss:.4f}")
    return avg_loss


def train_one_epoch_multi_signal_joint(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="多信号联合训练",
        # 损失权重
        alpha=1.0,  # MR一致性权重
        beta=0.3,  # 监督对比权重  
        gamma=0.5,  # UDA权重（如果有无标签数据）
        # MR一致性参数
    mr_temperature=1.0,
        # 对比学习参数
        contrast_temperature=0.1,
        margin=0.5,  # triplet margin
        # UDA参数
    confidence_threshold=0.8,
        # 正则化参数
        dropout_rate=0.3,
        label_smoothing=0.1,
        l2_reg_weight=1e-4,
        scheduler=None,
        **kwargs):
    """
    多信号联合损失函数训练一个epoch
    
    损失组成：
    L_total = L_CE + α*L_MR_cons + β*L_sup_contrast + γ*L_UDA
    
    参数:
    - model_instance: 模型实例
    - dataloader: 数据加载器，期望包含pair_id, is_syn, mr_type等字段
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - alpha: MR一致性损失权重
    - beta: 监督对比损失权重  
    - gamma: UDA损失权重
    - mr_temperature: MR一致性KL散度的温度参数
    - contrast_temperature: 对比学习温度参数
    - margin: triplet损失的边界
    - confidence_threshold: UDA的置信度阈值
    - dropout_rate: dropout概率
    - label_smoothing: 标签平滑参数
    - l2_reg_weight: L2正则化权重
    
    返回:
    - float: 平均损失值
    """
    import torch.nn as nn
    from collections import defaultdict

    model_instance.train()
    total_loss = 0.0

    # 动态dropout调整
    current_dropout = max(dropout_rate * (1.0 - epoch_idx / 20.0), 0.1)
    for module in model_instance.modules():
        if isinstance(module, nn.Dropout):
            module.p = current_dropout

    logger.info(
        f"{phase_name}: dropout={current_dropout:.3f}, α={alpha}, β={beta}, γ={gamma}"
    )

    # 蜕变关系转换矩阵
    transform_matrices = {
        "inv":
        torch.eye(3, device=device),  # 标签保持不变
        "flip":
        torch.tensor(
            [
                [0., 0., 1.],  # entailment -> contradiction
                [0., 1., 0.],  # neutral -> neutral  
                [1., 0., 0.]
            ],
            device=device),  # contradiction -> entailment
        "inv-flip":
        torch.tensor(
            [
                [0., 0., 1.],  # 可根据具体定义调整
                [0., 1., 0.],
                [1., 0., 0.]
            ],
            device=device)
    }

    # 损失函数
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                     reduction='none')

    for batch in tqdm(dataloader, desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad(set_to_none=True)

        # 数据准备
        is_syn = batch.pop("is_syn").to(device)
        pair_id = batch.pop("pair_id")
        mr_type = batch.pop("mr_type")
        weights = batch.pop("weight", torch.ones(len(is_syn))).to(device)
        labels = batch["labels"].to(device)

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 前向传播获取所有输出
        outputs = model_instance(**batch, output_hidden_states=True)
        logits = outputs.logits  # [N, 3]
        hidden_states = outputs.hidden_states[
            -1][:, 0, :]  # [N, hidden_size] CLS向量

        # 1. 基础交叉熵损失 L_CE
        ce_losses = ce_loss_fn(logits, labels)
        weighted_ce = (ce_losses * weights).mean()

        # 2. MR一致性损失 L_MR_cons
        mr_loss = torch.tensor(0.0, device=device)
        pair_groups = defaultdict(list)

        # 按pair_id分组
        for i, pid in enumerate(pair_id):
            pair_groups[pid].append(i)

        mr_count = 0
        for pid, indices in pair_groups.items():
            if len(indices) < 2:
                continue

            # 找到原始样本和蜕变样本
            orig_idx = None
            syn_indices = []

            for idx in indices:
                if not is_syn[idx]:
                    orig_idx = idx
                else:
                    syn_indices.append(idx)

            if orig_idx is None or len(syn_indices) == 0:
                continue

            orig_logits = logits[orig_idx]  # [3]
            orig_probs = F.softmax(orig_logits / mr_temperature, dim=-1)

            # 计算每个蜕变样本的MR一致性损失
            for syn_idx in syn_indices:
                syn_logits = logits[syn_idx]  # [3]
                syn_probs = F.softmax(syn_logits / mr_temperature, dim=-1)

                mr_t = mr_type[syn_idx]
                if mr_t in transform_matrices:
                    # 计算期望的目标分布
                    T = transform_matrices[mr_t]
                    target_probs = torch.matmul(orig_probs, T)

                    # KL散度损失：期望蜕变后的分布 vs 实际分布
                    kl_loss = F.kl_div(F.log_softmax(syn_logits /
                                                     mr_temperature,
                                                     dim=-1),
                                       target_probs.detach(),
                                       reduction='batchmean',
                                       log_target=False)
                    mr_loss += kl_loss
                    mr_count += 1

        if mr_count > 0:
            mr_loss = mr_loss / mr_count

        # 3. 监督对比损失 L_sup_contrast
        contrast_loss = torch.tensor(0.0, device=device)
        if beta > 0:
            # 归一化表示向量
            normalized_hidden = F.normalize(hidden_states, p=2,
                                            dim=1)  # [N, hidden_size]

            # 构造正负样本对
            contrast_count = 0
            for pid, indices in pair_groups.items():
                if len(indices) < 2:
                    continue

                # 对于同一个pair_id的样本
                for i, anchor_idx in enumerate(indices):
                    anchor_vec = normalized_hidden[anchor_idx]  # [hidden_size]
                    anchor_label = labels[anchor_idx]
                    anchor_is_syn = is_syn[anchor_idx]
                    anchor_mr = mr_type[anchor_idx] if anchor_is_syn else "orig"

                    positives = []
                    negatives = []

                    for j, other_idx in enumerate(indices):
                        if i == j:
                            continue

                        other_label = labels[other_idx]
                        other_is_syn = is_syn[other_idx]
                        other_mr = mr_type[
                            other_idx] if other_is_syn else "orig"

                        # 正样本：相同标签且为inv类型的蜕变
                        if (anchor_label == other_label
                                and (other_mr == "inv" or other_mr == "orig")):
                            positives.append(other_idx)
                        # 负样本：不同标签或flip类型的蜕变
                        elif (anchor_label != other_label
                              or other_mr == "flip"):
                            negatives.append(other_idx)

                    if len(positives) > 0 and len(negatives) > 0:
                        # 计算相似度
                        pos_vecs = normalized_hidden[
                            positives]  # [num_pos, hidden_size]
                        neg_vecs = normalized_hidden[
                            negatives]  # [num_neg, hidden_size]

                        # 计算余弦相似度
                        pos_sim = torch.matmul(
                            anchor_vec,
                            pos_vecs.T) / contrast_temperature  # [num_pos]
                        neg_sim = torch.matmul(
                            anchor_vec,
                            neg_vecs.T) / contrast_temperature  # [num_neg]

                        # InfoNCE损失
                        pos_exp = torch.exp(pos_sim)
                        neg_exp = torch.exp(neg_sim)

                        # 对每个正样本计算损失
                        for pos_s in pos_exp:
                            denominator = pos_s + neg_exp.sum()
                            contrast_loss += -torch.log(pos_s / denominator)
                            contrast_count += 1

            if contrast_count > 0:
                contrast_loss = contrast_loss / contrast_count

        # 4. UDA损失 L_UDA (如果有无标签数据)
        uda_loss = torch.tensor(0.0, device=device)
        if gamma > 0:
            # 找到高置信度的原始样本作为无标签数据
            orig_mask = ~is_syn
            if orig_mask.sum() > 0:
                orig_logits = logits[orig_mask]
                orig_probs = F.softmax(orig_logits, dim=-1)
                max_probs, pseudo_labels = orig_probs.max(dim=-1)

                # 高置信度样本
                confident_mask = max_probs > confidence_threshold
                if confident_mask.sum() > 0:
                    # 对应的蜕变样本应该保持一致（仅对inv类型）
                    uda_count = 0
                    orig_indices = torch.where(orig_mask)[0]
                    confident_orig_indices = orig_indices[confident_mask]

                    for orig_idx in confident_orig_indices:
                        orig_pid = pair_id[orig_idx]
                        orig_pseudo_label = pseudo_labels[confident_mask][
                            uda_count % confident_mask.sum()]

                        # 找到对应的inv类型蜕变样本
                        for i, pid in enumerate(pair_id):
                            if (pid == orig_pid and is_syn[i]
                                    and mr_type[i] == "inv"):
                                syn_logits = logits[i]
                                syn_log_probs = F.log_softmax(syn_logits,
                                                              dim=-1)

                                # 一致性损失：蜕变样本应该预测相同的伪标签
                                uda_loss += F.nll_loss(
                                    syn_log_probs.unsqueeze(0),
                                    orig_pseudo_label.unsqueeze(0))
                                uda_count += 1

                    if uda_count > 0:
                        uda_loss = uda_loss / uda_count

        # 5. L2正则化
        l2_reg = torch.stack(
            [torch.norm(p) for p in model_instance.parameters()]).sum()

        # 总损失
        total_batch_loss = (weighted_ce + alpha * mr_loss +
                            beta * contrast_loss + gamma * uda_loss +
                            l2_reg_weight * l2_reg)

        # 反向传播
        if not torch.isnan(
                total_batch_loss) and total_batch_loss.requires_grad:
            total_batch_loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += total_batch_loss.item()
        else:
            logger.warning(f"无效损失: {total_batch_loss.item()}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 完成")
    logger.info(f"平均损失: {avg_loss:.4f}")
    logger.info(
        f"损失组成 - CE: {weighted_ce:.4f}, MR: {alpha*mr_loss:.4f}, "
        f"Contrast: {beta*contrast_loss:.4f}, UDA: {gamma*uda_loss:.4f}")

    return avg_loss


def train_one_epoch_multi_signal_joint_RDrop(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="多信号联合训练",
        # 损失权重
        alpha=1.0,  # MR一致性权重
        beta=0.3,  # 监督对比权重  
        gamma=0.5,  # UDA权重（如果有无标签数据）
        # MR一致性参数
    mr_temperature=1.0,
        # 对比学习参数
        contrast_temperature=0.1,
        margin=0.5,  # triplet margin
        # UDA参数
    confidence_threshold=0.8,
        # 正则化参数
        dropout_rate=0.3,
        label_smoothing=0.1,
        l2_reg_weight=1e-4,
        lam=1.0,
        scheduler=None,
        **kwargs):
    """
    多信号联合损失函数训练一个epoch
    
    损失组成：
    L_total = L_CE + α*L_MR_cons + β*L_sup_contrast + γ*L_UDA
    
    参数:
    - model_instance: 模型实例
    - dataloader: 数据加载器，期望包含pair_id, is_syn, mr_type等字段
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - alpha: MR一致性损失权重
    - beta: 监督对比损失权重  
    - gamma: UDA损失权重
    - mr_temperature: MR一致性KL散度的温度参数
    - contrast_temperature: 对比学习温度参数
    - margin: triplet损失的边界
    - confidence_threshold: UDA的置信度阈值
    - dropout_rate: dropout概率
    - label_smoothing: 标签平滑参数
    - l2_reg_weight: L2正则化权重
    
    返回:
    - float: 平均损失值
    """
    import torch.nn as nn
    from collections import defaultdict

    model_instance.train()
    total_loss = 0.0

    # 动态dropout调整
    current_dropout = max(dropout_rate * (1.0 - epoch_idx / 20.0), 0.1)
    for module in model_instance.modules():
        if isinstance(module, nn.Dropout):
            module.p = current_dropout

    logger.info(
        f"{phase_name}: dropout={current_dropout:.3f}, α={alpha}, β={beta}, γ={gamma}"
    )

    # 蜕变关系转换矩阵
    transform_matrices = {
        "inv":
        torch.eye(3, device=device),  # 标签保持不变
        "flip":
        torch.tensor(
            [
                [0., 0., 1.],  # entailment -> contradiction
                [0., 1., 0.],  # neutral -> neutral  
                [1., 0., 0.]
            ],
            device=device),  # contradiction -> entailment
        "inv-flip":
        torch.tensor(
            [
                [0., 0., 1.],  # 可根据具体定义调整
                [0., 1., 0.],
                [1., 0., 0.]
            ],
            device=device)
    }

    # 损失函数
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                     reduction='none')

    for batch in tqdm(dataloader, desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad(set_to_none=True)

        # 数据准备
        is_syn = batch.pop("is_syn").to(device)
        pair_id = batch.pop("pair_id")
        mr_type = batch.pop("mr_type")
        weights = batch.pop("weight", torch.ones(len(is_syn))).to(device)
        labels = batch["labels"].to(device)

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 前向传播获取所有输出
        outputs = model_instance(**batch, output_hidden_states=True)
        logits = outputs.logits  # [N, 3]
        hidden_states = outputs.hidden_states[
            -1][:, 0, :]  # [N, hidden_size] CLS向量

        # 1. 基础交叉熵损失 L_CE
        ce_losses = ce_loss_fn(logits, labels)
        weighted_ce = (ce_losses * weights).mean()

        # 2. MR一致性损失 L_MR_cons
        mr_loss = torch.tensor(0.0, device=device)
        pair_groups = defaultdict(list)

        # 按pair_id分组
        for i, pid in enumerate(pair_id):
            pair_groups[pid].append(i)

        mr_count = 0
        for pid, indices in pair_groups.items():
            if len(indices) < 2:
                continue

            # 找到原始样本和蜕变样本
            orig_idx = None
            syn_indices = []

            for idx in indices:
                if not is_syn[idx]:
                    orig_idx = idx
                else:
                    syn_indices.append(idx)

            if orig_idx is None or len(syn_indices) == 0:
                continue

            orig_logits = logits[orig_idx]  # [3]
            orig_probs = F.softmax(orig_logits / mr_temperature, dim=-1)

            # 计算每个蜕变样本的MR一致性损失
            for syn_idx in syn_indices:
                syn_logits = logits[syn_idx]  # [3]
                syn_probs = F.softmax(syn_logits / mr_temperature, dim=-1)

                mr_t = mr_type[syn_idx]
                if mr_t in transform_matrices:
                    # 计算期望的目标分布
                    T = transform_matrices[mr_t]
                    target_probs = torch.matmul(orig_probs, T)

                    # KL散度损失：期望蜕变后的分布 vs 实际分布
                    kl_loss = F.kl_div(F.log_softmax(syn_logits /
                                                     mr_temperature,
                                                     dim=-1),
                                       target_probs.detach(),
                                       reduction='batchmean',
                                       log_target=False)
                    mr_loss += kl_loss
                    mr_count += 1

        if mr_count > 0:
            mr_loss = mr_loss / mr_count

        # 3. 监督对比损失 L_sup_contrast
        contrast_loss = torch.tensor(0.0, device=device)
        if beta > 0:
            # 归一化表示向量
            normalized_hidden = F.normalize(hidden_states, p=2,
                                            dim=1)  # [N, hidden_size]

            # 构造正负样本对
            contrast_count = 0
            for pid, indices in pair_groups.items():
                if len(indices) < 2:
                    continue

                # 对于同一个pair_id的样本
                for i, anchor_idx in enumerate(indices):
                    anchor_vec = normalized_hidden[anchor_idx]  # [hidden_size]
                    anchor_label = labels[anchor_idx]
                    anchor_is_syn = is_syn[anchor_idx]
                    anchor_mr = mr_type[anchor_idx] if anchor_is_syn else "orig"

                    positives = []
                    negatives = []

                    for j, other_idx in enumerate(indices):
                        if i == j:
                            continue

                        other_label = labels[other_idx]
                        other_is_syn = is_syn[other_idx]
                        other_mr = mr_type[
                            other_idx] if other_is_syn else "orig"

                        # 正样本：相同标签且为inv类型的蜕变
                        if (anchor_label == other_label
                                and (other_mr == "inv" or other_mr == "orig")):
                            positives.append(other_idx)
                        # 负样本：不同标签或flip类型的蜕变
                        elif (anchor_label != other_label
                              or other_mr == "flip"):
                            negatives.append(other_idx)

                    if len(positives) > 0 and len(negatives) > 0:
                        # 计算相似度
                        pos_vecs = normalized_hidden[
                            positives]  # [num_pos, hidden_size]
                        neg_vecs = normalized_hidden[
                            negatives]  # [num_neg, hidden_size]

                        # 计算余弦相似度
                        pos_sim = torch.matmul(
                            anchor_vec,
                            pos_vecs.T) / contrast_temperature  # [num_pos]
                        neg_sim = torch.matmul(
                            anchor_vec,
                            neg_vecs.T) / contrast_temperature  # [num_neg]

                        # InfoNCE损失
                        pos_exp = torch.exp(pos_sim)
                        neg_exp = torch.exp(neg_sim)

                        # 对每个正样本计算损失
                        for pos_s in pos_exp:
                            denominator = pos_s + neg_exp.sum()
                            contrast_loss += -torch.log(pos_s / denominator)
                            contrast_count += 1

            if contrast_count > 0:
                contrast_loss = contrast_loss / contrast_count

        # 4. UDA损失 L_UDA (如果有无标签数据)
        uda_loss = torch.tensor(0.0, device=device)
        if gamma > 0:
            # 找到高置信度的原始样本作为无标签数据
            orig_mask = ~is_syn
            if orig_mask.sum() > 0:
                orig_logits = logits[orig_mask]
                orig_probs = F.softmax(orig_logits, dim=-1)
                max_probs, pseudo_labels = orig_probs.max(dim=-1)

                # 高置信度样本
                confident_mask = max_probs > confidence_threshold
                if confident_mask.sum() > 0:
                    # 对应的蜕变样本应该保持一致（仅对inv类型）
                    uda_count = 0
                    orig_indices = torch.where(orig_mask)[0]
                    confident_orig_indices = orig_indices[confident_mask]

                    for orig_idx in confident_orig_indices:
                        orig_pid = pair_id[orig_idx]
                        orig_pseudo_label = pseudo_labels[confident_mask][
                            uda_count % confident_mask.sum()]

                        # 找到对应的inv类型蜕变样本
                        for i, pid in enumerate(pair_id):
                            if (pid == orig_pid and is_syn[i]
                                    and mr_type[i] == "inv"):
                                syn_logits = logits[i]
                                syn_log_probs = F.log_softmax(syn_logits,
                                                              dim=-1)

                                # 一致性损失：蜕变样本应该预测相同的伪标签
                                uda_loss += F.nll_loss(
                                    syn_log_probs.unsqueeze(0),
                                    orig_pseudo_label.unsqueeze(0))
                                uda_count += 1

                    if uda_count > 0:
                        uda_loss = uda_loss / uda_count

        # RDrop
        # 两次前向传播（用于 R-Drop）
        outputs1 = model_instance(**batch)
        outputs2 = model_instance(**batch)

        # 计算各个损失分量
        ce_loss = outputs1.loss
        ce_loss_syn = outputs2.loss

        # 计算 KL 散度（用于 R-Drop）
        kl_loss = F.kl_div(F.log_softmax(outputs1.logits, dim=-1),
                           F.softmax(outputs2.logits, dim=-1),
                           reduction='batchmean')

        # 5. L2正则化
        l2_reg = torch.stack(
            [torch.norm(p) for p in model_instance.parameters()]).sum()

        # 总损失
        total_batch_loss = (weighted_ce + alpha * mr_loss +
                            beta * contrast_loss + gamma * uda_loss +
                            l2_reg_weight * l2_reg + lam * kl_loss)

        # 反向传播
        if not torch.isnan(
                total_batch_loss) and total_batch_loss.requires_grad:
            total_batch_loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += total_batch_loss.item()
        else:
            logger.warning(f"无效损失: {total_batch_loss.item()}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 完成")
    logger.info(f"平均损失: {avg_loss:.4f}")
    logger.info(
        f"损失组成 - CE: {weighted_ce:.4f},\nMR: {alpha*mr_loss:.4f}, \nRDrop: {lam*kl_loss:.4f}, "
        f"\nContrast: {beta*contrast_loss:.4f}, \nUDA: {gamma*uda_loss:.4f}")

    return avg_loss


class SupConLoss(nn.Module):
    """监督对比学习损失函数
    
    支持MR增强的对比学习：
    - 正样本：同标签样本和inv-MR变体
    - 负样本：不同标签样本和flip-MR变体
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, mr_types=None, is_syn=None):
        """
        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]
            mr_types: MR类型列表 [batch_size] ("inv", "flip", "none")
            is_syn: 是否为合成样本 [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        if mr_types is None:
            mr_types = ["none"] * batch_size
        if is_syn is None:
            is_syn = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 计算相似度矩阵
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm,
                                         features_norm.T) / self.temperature

        # 创建正样本mask
        labels_expanded = labels.unsqueeze(0)
        same_label_mask = torch.eq(labels_expanded, labels_expanded.T)

        # 为inv-MR变体创建额外的正样本关系
        positive_mask = same_label_mask.clone()
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # inv-MR与原样本标签相同，应为正样本
                    if mr_types[i] == "inv" and mr_types[
                            j] == "none" and labels[i] == labels[j]:
                        positive_mask[i, j] = True
                    elif mr_types[i] == "none" and mr_types[
                            j] == "inv" and labels[i] == labels[j]:
                        positive_mask[i, j] = True

        # 创建负样本mask（排除自己）
        negative_mask = ~positive_mask
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        negative_mask = negative_mask & ~identity_mask

        # 为flip-MR创建额外的负样本关系
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and mr_types[i] == "flip":
                    negative_mask[i, j] = True

        # 移除对角线
        positive_mask = positive_mask & ~identity_mask

        # 计算对比损失
        total_loss = 0.0
        num_positives = 0

        for i in range(batch_size):
            # 找到正样本和负样本
            pos_indices = positive_mask[i]
            neg_indices = negative_mask[i]

            if pos_indices.any():
                # 正样本分子
                pos_similarities = similarity_matrix[i][pos_indices]
                pos_exp = torch.exp(pos_similarities)

                # 分母：正样本 + 负样本
                neg_similarities = similarity_matrix[i][neg_indices]
                neg_exp = torch.exp(neg_similarities)

                denominator = pos_exp.sum() + neg_exp.sum()

                # 对每个正样本计算损失
                for pos_sim in pos_exp:
                    loss_i = -torch.log(pos_sim / denominator)
                    total_loss += loss_i
                    num_positives += 1

        return total_loss / max(num_positives, 1)


def train_one_epoch_composite_mr_loss(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="",
        # 损失权重参数
        alpha_inv=1.0,  # inv-MR权重
        alpha_flip=1.0,  # flip-MR权重  
        alpha_composite=1.0,  # 复合MR权重
        beta=0.5,  # SupCon权重
        lambda_neu=0.1,  # 中性类熵正则权重
        lambda_reg=1e-4,  # L2正则化权重
        lambda_wd=1e-3,  # 分类层权重衰减
        mr_temp=1.0,  # MR温度参数
        supcon_temp=0.07,  # 对比学习温度
        label_smoothing=0.1):  # 标签平滑
    """
    使用复合MR驱动的多目标损失函数训练一个epoch
    
    实现论文中提出的损失函数：
    L_total = L_CE + Σα_i*L_MR^(i) + β*L_SupCon + λ_neu*L_Neutral + λ_reg*L_Reg
    
    支持特性：
    - label-preserving, label-flipping, 复合MR
    - MR一致性自适应权重
    - 监督对比学习
    - 中性类熵正则化
    - 参数正则化
    """
    model_instance.train()
    total_loss = 0.0

    # 初始化SupCon损失
    supcon_criterion = SupConLoss(temperature=supcon_temp).to(device)

    # 标签变换矩阵 (3x3 for entailment, neutral, contradiction)
    # Identity for inv-MR (label-preserving)
    T_inv = torch.eye(3, device=device)
    # Flip matrix for flip-MR (entailment<->contradiction)
    T_flip = torch.tensor(
        [
            [0, 0, 1],  # entailment -> contradiction
            [0, 1, 0],  # neutral -> neutral  
            [1, 0, 0]
        ],
        dtype=torch.float,
        device=device)  # contradiction -> entailment

    for batch in tqdm(dataloader, desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad()

        # 数据准备
        is_syn = batch.pop("is_syn",
                           torch.zeros(len(batch["labels"]), dtype=torch.bool))
        pair_id = batch.pop("pair_id", list(range(len(batch["labels"]))))
        mr_type = batch.pop("mr_type", ["none"] * len(batch["labels"]))

        batch = {
            k:
            v.to(device)
            if hasattr(v, 'to') and callable(getattr(v, 'to')) else v
            for k, v in batch.items()
        }
        labels = batch["labels"]

        # 前向传播
        outputs = model_instance(**batch)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1][:, 0, :] if hasattr(
            outputs, 'hidden_states') and outputs.hidden_states else None

        # 1. 交叉熵损失（带标签平滑）
        ce_loss = F.cross_entropy(logits,
                                  labels,
                                  label_smoothing=label_smoothing)

        # 2. MR一致性损失
        mr_losses = {'inv': 0.0, 'flip': 0.0, 'composite': 0.0}
        mr_counts = {'inv': 0, 'flip': 0, 'composite': 0}

        # 构建原始样本映射
        orig_logits_dict = {}
        for i, (pid, is_s) in enumerate(zip(pair_id, is_syn)):
            if not is_s:  # 原始样本
                orig_logits_dict[pid] = logits[i]

        # 计算MR一致性损失
        for i, (pid, is_s, mr_t) in enumerate(zip(pair_id, is_syn, mr_type)):
            if is_s and pid in orig_logits_dict:  # 合成样本且有对应原始样本
                orig_logits = orig_logits_dict[pid]
                syn_logits = logits[i]

                # 选择变换矩阵
                if mr_t == "inv":
                    T = T_inv
                    mr_type_key = 'inv'
                elif mr_t == "flip":
                    T = T_flip
                    mr_type_key = 'flip'
                else:  # 复合或其他类型
                    T = T_inv  # 默认使用inv变换
                    mr_type_key = 'composite'

                # 计算目标概率分布
                p_orig = F.softmax(orig_logits / mr_temp, dim=-1)
                target_probs = torch.matmul(p_orig, T)

                # KL散度损失
                kl_loss = F.kl_div(F.log_softmax(syn_logits / mr_temp, dim=-1),
                                   target_probs,
                                   reduction='batchmean')

                mr_losses[mr_type_key] += kl_loss
                mr_counts[mr_type_key] += 1

        # 平均MR损失
        mr_loss_total = 0.0
        if mr_counts['inv'] > 0:
            mr_loss_total += alpha_inv * (mr_losses['inv'] / mr_counts['inv'])
        if mr_counts['flip'] > 0:
            mr_loss_total += alpha_flip * (mr_losses['flip'] /
                                           mr_counts['flip'])
        if mr_counts['composite'] > 0:
            mr_loss_total += alpha_composite * (mr_losses['composite'] /
                                                mr_counts['composite'])

        # 3. 监督对比学习损失
        supcon_loss = torch.tensor(0.0, device=device)
        if hidden_states is not None:
            supcon_loss = supcon_criterion(hidden_states, labels, mr_type,
                                           is_syn)

        # 4. 中性类熵正则化
        neutral_loss = torch.tensor(0.0, device=device)
        neutral_mask = (labels == 1)  # 假设中性类标签为1
        if neutral_mask.any():
            neutral_probs = F.softmax(logits[neutral_mask], dim=-1)
            # 最大化熵（负熵损失）
            entropy = -torch.sum(
                neutral_probs * torch.log(neutral_probs + 1e-8), dim=-1)
            neutral_loss = -entropy.mean()  # 负号用于最大化熵

        # 5. 参数正则化
        l2_loss = torch.tensor(0.0, device=device)
        classifier_weight_loss = torch.tensor(0.0, device=device)

        for name, param in model_instance.named_parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2)
                # 对分类层应用额外的权重衰减
                if 'classifier' in name and 'weight' in name:
                    classifier_weight_loss += torch.norm(param, p=2)

        reg_loss = lambda_reg * l2_loss + lambda_wd * classifier_weight_loss

        # 6. 总损失计算
        total_batch_loss = (ce_loss + mr_loss_total + beta * supcon_loss +
                            lambda_neu * neutral_loss + reg_loss)

        # 反向传播
        if not torch.isnan(
                total_batch_loss) and total_batch_loss.requires_grad:
            total_batch_loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_batch_loss.item()
        else:
            logger.warning(f"无效损失: {total_batch_loss.item()}")

    avg_loss = total_loss / len(dataloader)

    # 详细日志
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 结束，平均损失: {avg_loss:.4f}")
    logger.info(f"损失组成详情:")
    logger.info(f"  - 交叉熵损失: {ce_loss.item():.4f}")
    logger.info(f"  - MR一致性损失 (总): {mr_loss_total.item():.4f}")
    logger.info(
        f"    * inv-MR: {mr_losses['inv']:.4f} (样本数: {mr_counts['inv']})")
    logger.info(
        f"    * flip-MR: {mr_losses['flip']:.4f} (样本数: {mr_counts['flip']})")
    logger.info(
        f"    * composite-MR: {mr_losses['composite']:.4f} (样本数: {mr_counts['composite']})"
    )
    logger.info(f"  - 对比学习损失: {supcon_loss.item():.4f}")
    logger.info(f"  - 中性类熵正则: {neutral_loss.item():.4f}")
    logger.info(f"  - 参数正则化: {reg_loss.item():.4f}")

    return avg_loss


def train_one_epoch_simple_composite_mr(
        model_instance,
        dataloader,
        optimizer,
        device,
        epoch_idx,
        phase_name="",
        # 简化的权重参数
        alpha=1.0,  # MR一致性权重
        beta=0.3,  # 对比学习权重
        lambda_neu=0.05,  # 中性类权重
        lambda_reg=1e-4,  # 正则化权重
        use_supcon=True,  # 是否使用对比学习
        use_neutral_reg=True,  # 是否使用中性类正则
        scheduler=None):
    """
    简化版的复合MR损失函数，更易使用
    
    主要特性：
    - 统一的MR一致性损失（自动处理inv/flip类型）
    - 可选的监督对比学习
    - 可选的中性类熵正则化
    - 简化的参数设置
    """
    model_instance.train()
    total_loss = 0.0

    # 损失统计
    loss_stats = {
        'ce': 0.0,
        'mr': 0.0,
        'supcon': 0.0,
        'neutral': 0.0,
        'reg': 0.0,
        'total': 0.0
    }
    batch_count = 0

    # 初始化SupCon损失
    if use_supcon:
        supcon_criterion = SupConLoss(temperature=0.07).to(device)

    for batch in tqdm(dataloader, desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad(set_to_none=True)

        # 数据准备
        is_syn = batch.pop("is_syn",
                           torch.zeros(len(batch["labels"]), dtype=torch.bool))
        pair_id = batch.pop("pair_id", list(range(len(batch["labels"]))))
        mr_type = batch.pop("mr_type", ["none"] * len(batch["labels"]))

        batch = {
            k:
            v.to(device)
            if hasattr(v, 'to') and callable(getattr(v, 'to')) else v
            for k, v in batch.items()
        }
        labels = batch["labels"]

        # 前向传播
        outputs = model_instance(**batch)
        logits = outputs.logits

        # 1. 基础交叉熵损失
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        loss_stats['ce'] += ce_loss.item()

        # 2. MR一致性损失（统一处理）
        mr_loss = torch.tensor(0.0, device=device)
        mr_count = 0

        # 构建原始样本映射
        orig_logits_dict = {}
        for i, (pid, is_s) in enumerate(zip(pair_id, is_syn)):
            if not is_s:
                orig_logits_dict[pid] = logits[i]

        # 计算MR损失
        for i, (pid, is_s, mr_t) in enumerate(zip(pair_id, is_syn, mr_type)):
            if is_s and pid in orig_logits_dict and mr_t in ["inv", "flip"]:
                orig_logits = orig_logits_dict[pid]
                syn_logits = logits[i]

                # 根据MR类型选择一致性策略
                if mr_t == "inv":
                    # label-preserving: 预测分布应该一致
                    target_probs = F.softmax(orig_logits, dim=-1)
                elif mr_t == "flip":
                    # label-flipping: entailment <-> contradiction
                    orig_probs = F.softmax(orig_logits, dim=-1)
                    # 简单的翻转策略：交换0和2位置的概率
                    target_probs = torch.stack(
                        [orig_probs[2], orig_probs[1], orig_probs[0]])

                # KL散度损失
                kl_loss = F.kl_div(F.log_softmax(syn_logits, dim=-1),
                                   target_probs.detach(),
                                   reduction='batchmean')
                mr_loss += kl_loss
                mr_count += 1

        if mr_count > 0:
            mr_loss = mr_loss / mr_count
        loss_stats['mr'] += mr_loss.item()

        # 3. 对比学习损失（可选）
        supcon_loss = torch.tensor(0.0, device=device)
        if use_supcon:
            try:
                # 使用最后一层的[CLS]表示作为特征
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    features = outputs.hidden_states[-1][:,
                                                         0, :]  # [CLS] token
                else:
                    # 如果没有hidden_states，使用logits作为特征
                    features = logits
                supcon_loss = supcon_criterion(features, labels, mr_type,
                                               is_syn)
            except Exception as e:
                logger.warning(f"对比学习损失计算失败: {e}")
                supcon_loss = torch.tensor(0.0, device=device)
        loss_stats['supcon'] += supcon_loss.item()

        # 4. 中性类熵正则化（可选）
        neutral_loss = torch.tensor(0.0, device=device)
        if use_neutral_reg:
            neutral_mask = (labels == 1)  # 假设中性类标签为1
            if neutral_mask.any():
                neutral_probs = F.softmax(logits[neutral_mask], dim=-1)
                entropy = -torch.sum(
                    neutral_probs * torch.log(neutral_probs + 1e-8), dim=-1)
                neutral_loss = -entropy.mean()  # 最大化熵
        loss_stats['neutral'] += neutral_loss.item()

        # 5. L2正则化
        reg_loss = torch.tensor(0.0, device=device)
        for param in model_instance.parameters():
            if param.requires_grad:
                reg_loss += torch.norm(param, p=2)
        reg_loss = lambda_reg * reg_loss
        loss_stats['reg'] += reg_loss.item()

        # 总损失
        total_batch_loss = (ce_loss + alpha * mr_loss + beta * supcon_loss +
                            lambda_neu * neutral_loss + reg_loss)

        loss_stats['total'] += total_batch_loss.item()
        batch_count += 1

        # 反向传播
        if not torch.isnan(
                total_batch_loss) and total_batch_loss.requires_grad:
            total_batch_loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += total_batch_loss.item()
        else:
            logger.warning(f"跳过无效批次，损失值: {total_batch_loss.item()}")

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    for key in loss_stats:
        loss_stats[key] /= batch_count

    # 日志输出
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 完成")
    logger.info(f"平均总损失: {avg_loss:.4f}")
    logger.info(
        f"损失分解: CE={loss_stats['ce']:.4f}, MR={loss_stats['mr']:.4f}, "
        f"SupCon={loss_stats['supcon']:.4f}, Neutral={loss_stats['neutral']:.4f}, "
        f"Reg={loss_stats['reg']:.4f}")

    return avg_loss


# 便捷的训练函数工厂
def create_composite_mr_trainer(mr_weight=1.0,
                                contrast_weight=0.3,
                                neutral_weight=0.05,
                                enable_contrast=True,
                                enable_neutral_reg=True):
    """
    创建预配置的复合MR训练函数
    
    Args:
        mr_weight: MR一致性损失权重
        contrast_weight: 对比学习损失权重  
        neutral_weight: 中性类正则化权重
        enable_contrast: 是否启用对比学习
        enable_neutral_reg: 是否启用中性类正则化
    
    Returns:
        配置好的训练函数
    """

    def trainer(model_instance,
                dataloader,
                optimizer,
                device,
                epoch_idx,
                phase_name=""):
        return train_one_epoch_simple_composite_mr(
            model_instance=model_instance,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch_idx,
            phase_name=phase_name,
            alpha=mr_weight,
            beta=contrast_weight,
            lambda_neu=neutral_weight,
            use_supcon=enable_contrast,
            use_neutral_reg=enable_neutral_reg)

    return trainer


def train_one_epoch_pairwise_metamorphic(
        model_instance,
        dataset,  # 直接传入dataset而非dataloader
        tokenizer,  # 添加tokenizer参数
        optimizer,
        device,
        epoch_idx,
        phase_name="蜕变关系训练",
        # 损失权重参数
        alpha_inv=1.0,  # inv-MR权重
        alpha_flip=1.0,  # flip-MR权重  
        alpha_neutral=1.0,  # neutral-MR权重
        beta_consistency=0.5,  # 一致性损失权重
        gamma_ce=1.0,  # 交叉熵损失权重
        # 温度参数
    temperature=1.0,
        # 正则化参数
        label_smoothing=0.1,
        l2_reg_weight=1e-4):
    """
    按pair_id分组训练，充分利用蜕变关系进行数据增强的训练函数
    
    参数:
    - model_instance: 模型实例
    - dataset: 增强数据集（包含源数据和蜕变数据）
    - tokenizer: 分词器（用于文本编码）
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - alpha_inv/flip/neutral: 不同MR类型的权重
    - beta_consistency: 一致性损失权重
    - gamma_ce: 交叉熵损失权重
    - temperature: KL散度的温度参数
    - label_smoothing: 标签平滑参数
    - l2_reg_weight: L2正则化权重
    
    返回:
    - float: 平均损失值
    """
    model_instance.train()
    total_loss = 0.0
    processed_pairs = 0

    # 损失函数
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                     reduction='none')

    # 按pair_id分组数据
    logger.info("按pair_id分组数据...")
    pair_groups = defaultdict(list)

    for i in range(len(dataset)):
        item = dataset[i]
        pair_id = item.get('pair_id', i)  # 如果没有pair_id则使用索引
        pair_groups[pair_id].append((i, item))

    logger.info(f"共找到 {len(pair_groups)} 个数据组")

    # 按组处理数据
    for pair_id, group_items in tqdm(pair_groups.items(),
                                     desc=f"{phase_name} Epoch {epoch_idx+1}"):
        if len(group_items) == 1:
            # 只有源数据，没有增强数据，跳过
            continue

        optimizer.zero_grad()

        # 分离源数据和蜕变数据
        source_data = None
        metamorphic_data = {'inv': [], 'flip': [], 'neutral': []}

        for idx, item in group_items:
            is_syn = item.get('is_syn', False)
            mr_id = item.get('mr_id', 'none')
            mr_type = item.get('mr_type', 'inv')

            if mr_id == 'none' or not is_syn:
                # 源数据
                source_data = item
            else:
                # 蜕变数据
                if mr_type in metamorphic_data:
                    metamorphic_data[mr_type].append(item)

        if source_data is None:
            # 没有源数据，跳过这个组
            continue

        # 准备模型输入
        def prepare_inputs(items_list):
            if not items_list:
                return None

            premises = [item["premise"] for item in items_list]
            hypotheses = [item["hypothesis"] for item in items_list]
            labels = [item["label"] for item in items_list]

            # 使用传入的tokenizer进行编码
            encodings = tokenizer(premises,
                                  hypotheses,
                                  truncation=True,
                                  padding=True,
                                  return_tensors="pt")

            # 移动到设备
            for key in encodings:
                if isinstance(encodings[key], torch.Tensor):
                    encodings[key] = encodings[key].to(device)

            encodings["labels"] = torch.tensor(labels,
                                               dtype=torch.long,
                                               device=device)
            return encodings  # 计算源数据的输出

        source_inputs = prepare_inputs([source_data])
        source_outputs = model_instance(**source_inputs)
        source_logits = source_outputs.logits
        source_probs = F.softmax(source_logits / temperature, dim=-1)

        # 计算源数据的CE损失
        ce_loss = gamma_ce * source_outputs.loss
        total_batch_loss = ce_loss

        # 处理不同类型的蜕变数据
        consistency_losses = []

        # 1. inv-MR: 标签不变，预测分布应该一致
        if metamorphic_data['inv'] and alpha_inv > 0:
            inv_inputs = prepare_inputs(metamorphic_data['inv'])
            inv_outputs = model_instance(**inv_inputs)
            inv_probs = F.softmax(inv_outputs.logits / temperature, dim=-1)

            # KL散度一致性损失：蜕变样本应该与源样本预测一致
            kl_inv = F.kl_div(F.log_softmax(inv_outputs.logits / temperature,
                                            dim=-1),
                              source_probs.expand(inv_probs.size(0), -1),
                              reduction='batchmean')

            # 添加蜕变样本的CE损失（标签应该与源数据相同）
            inv_ce = ce_loss_fn(inv_outputs.logits,
                                inv_inputs["labels"]).mean()

            inv_loss = alpha_inv * (beta_consistency * kl_inv +
                                    gamma_ce * inv_ce)
            consistency_losses.append(inv_loss)
            total_batch_loss += inv_loss

        # 2. flip-MR: 标签翻转（entailment <-> contradiction）
        if metamorphic_data['flip'] and alpha_flip > 0:
            flip_inputs = prepare_inputs(metamorphic_data['flip'])
            flip_outputs = model_instance(**flip_inputs)

            # 创建翻转的目标分布：entailment(0) <-> contradiction(2)
            flip_target_probs = source_probs.clone()
            flip_target_probs[:, [0, 2]] = flip_target_probs[:,
                                                             [2,
                                                              0]]  # 交换0和2的概率

            # KL散度一致性损失
            kl_flip = F.kl_div(F.log_softmax(flip_outputs.logits / temperature,
                                             dim=-1),
                               flip_target_probs.expand(
                                   flip_outputs.logits.size(0), -1),
                               reduction='batchmean')

            # 添加翻转样本的CE损失
            flip_ce = ce_loss_fn(flip_outputs.logits,
                                 flip_inputs["labels"]).mean()

            flip_loss = alpha_flip * (beta_consistency * kl_flip +
                                      gamma_ce * flip_ce)
            consistency_losses.append(flip_loss)
            total_batch_loss += flip_loss

        # 3. neutral-MR: 标签变为中性
        if metamorphic_data['neutral'] and alpha_neutral > 0:
            neutral_inputs = prepare_inputs(metamorphic_data['neutral'])
            neutral_outputs = model_instance(**neutral_inputs)

            # 创建中性目标分布：neutral类概率应该很高
            neutral_target_probs = torch.zeros_like(source_probs)
            neutral_target_probs[:, 1] = 1.0  # neutral类为1

            # KL散度一致性损失
            kl_neutral = F.kl_div(
                F.log_softmax(neutral_outputs.logits / temperature, dim=-1),
                neutral_target_probs.expand(neutral_outputs.logits.size(0),
                                            -1),
                reduction='batchmean')

            # 添加中性样本的CE损失
            neutral_ce = ce_loss_fn(neutral_outputs.logits,
                                    neutral_inputs["labels"]).mean()

            neutral_loss = alpha_neutral * (beta_consistency * kl_neutral +
                                            gamma_ce * neutral_ce)
            consistency_losses.append(neutral_loss)
            total_batch_loss += neutral_loss

        # L2正则化
        if l2_reg_weight > 0:
            l2_reg = l2_reg_weight * sum(
                torch.norm(p, p=2) for p in model_instance.parameters())
            total_batch_loss += l2_reg

        # 反向传播
        if not torch.isnan(
                total_batch_loss) and total_batch_loss.requires_grad:
            total_batch_loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_batch_loss.item()
            processed_pairs += 1
        else:
            logger.warning(
                f"无效的损失值: {total_batch_loss.item() if not torch.isnan(total_batch_loss) else 'NaN'}"
            )

    if processed_pairs == 0:
        logger.warning("没有处理任何数据对！")
        return 0.0

    avg_loss = total_loss / processed_pairs
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 结束")
    logger.info(f"处理了 {processed_pairs} 个数据组，平均损失: {avg_loss:.4f}")
    logger.info(
        f"损失权重配置: inv={alpha_inv}, flip={alpha_flip}, neutral={alpha_neutral}, consistency={beta_consistency}"
    )

    return avg_loss


def create_pairwise_dataloader_from_dataset(dataset,
                                            tokenizer,
                                            max_pairs_per_batch=4):
    """
    创建按pair_id分组的特殊数据加载器
    
    参数:
    - dataset: 增强数据集
    - tokenizer: 分词器
    - max_pairs_per_batch: 每个batch最大包含的pair数量
    
    返回:
    - generator: 产生按pair分组的batch数据
    """
    # 按pair_id分组
    pair_groups = defaultdict(list)
    for i in range(len(dataset)):
        item = dataset[i]
        pair_id = item.get('pair_id', i)
        pair_groups[pair_id].append(item)

    # 过滤只有一个样本的组（没有增强数据）
    valid_pairs = {k: v for k, v in pair_groups.items() if len(v) > 1}

    logger.info(f"找到 {len(valid_pairs)} 个有效的数据对组（包含增强数据）")

    # 按批次生成数据
    pair_ids = list(valid_pairs.keys())
    for i in range(0, len(pair_ids), max_pairs_per_batch):
        batch_pair_ids = pair_ids[i:i + max_pairs_per_batch]
        batch_data = []

        for pair_id in batch_pair_ids:
            batch_data.extend(valid_pairs[pair_id])

        # 准备batch数据
        premises = [item["premise"] for item in batch_data]
        hypotheses = [item["hypothesis"] for item in batch_data]
        labels = [item["label"] for item in batch_data]
        is_syn = [item.get("is_syn", False) for item in batch_data]
        pair_ids_list = [item.get("pair_id", -1) for item in batch_data]
        mr_types = [item.get("mr_type", "inv") for item in batch_data]
        mr_ids = [item.get("mr_id", "none") for item in batch_data]

        # 编码
        encodings = tokenizer(premises,
                              hypotheses,
                              truncation=True,
                              padding=True,
                              return_tensors="pt")

        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        encodings["is_syn"] = torch.tensor(is_syn, dtype=torch.bool)
        encodings["pair_id"] = pair_ids_list
        encodings["mr_type"] = mr_types
        encodings["mr_id"] = mr_ids

        yield encodings


def train_one_epoch_pairwise_dataloader(
        model_instance,
        pairwise_dataloader,  # 使用特殊的pairwise dataloader
        optimizer,
        device,
        epoch_idx,
        phase_name="蜕变关系训练",
        # 损失权重参数
        alpha_inv=1.0,  # inv-MR权重
        alpha_flip=1.0,  # flip-MR权重  
        alpha_neutral=1.0,  # neutral-MR权重
        beta_consistency=0.5,  # 一致性损失权重
        gamma_ce=1.0,  # 交叉熵损失权重
        # 温度参数
    temperature=1.0,
        # 正则化参数
        label_smoothing=0.1,
        l2_reg_weight=1e-4):
    """
    使用pairwise dataloader的简化版训练函数
    
    参数:
    - model_instance: 模型实例
    - pairwise_dataloader: 按pair_id分组的数据加载器
    - optimizer: 优化器
    - device: 计算设备
    - epoch_idx: 当前epoch索引
    - phase_name: 训练阶段名称
    - 其他参数同上一个函数
    
    返回:
    - float: 平均损失值
    """
    model_instance.train()
    total_loss = 0.0
    batch_count = 0

    # 损失函数
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                     reduction='none')

    for batch in tqdm(pairwise_dataloader,
                      desc=f"{phase_name} Epoch {epoch_idx+1}"):
        optimizer.zero_grad()

        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # 提取元数据
        labels = batch["labels"]
        is_syn = batch["is_syn"]
        pair_ids = batch["pair_id"]
        mr_types = batch["mr_type"]
        mr_ids = batch["mr_id"]

        # 移除元数据，准备模型输入
        model_inputs = {
            k: v
            for k, v in batch.items()
            if k not in ["is_syn", "pair_id", "mr_type", "mr_id"]
        }

        # 前向传播
        outputs = model_instance(**model_inputs)
        logits = outputs.logits

        # 按pair_id分组处理损失
        unique_pair_ids = list(set(pair_ids))
        total_batch_loss = 0.0

        for pair_id in unique_pair_ids:
            # 找到属于当前pair的所有样本
            pair_mask = torch.tensor([pid == pair_id for pid in pair_ids],
                                     device=device)
            if pair_mask.sum() <= 1:
                continue  # 跳过只有一个样本的pair

            pair_logits = logits[pair_mask]
            pair_labels = labels[pair_mask]
            pair_is_syn = is_syn[pair_mask]
            pair_mr_types = [
                mr_types[i] for i in range(len(mr_types))
                if pair_ids[i] == pair_id
            ]
            pair_mr_ids = [
                mr_ids[i] for i in range(len(mr_ids)) if pair_ids[i] == pair_id
            ]

            # 找到源数据（mr_id == 'none' 或 is_syn == False）
            source_indices = []
            metamorphic_indices = {'inv': [], 'flip': [], 'neutral': []}

            for i, (is_s, mr_id, mr_type) in enumerate(
                    zip(pair_is_syn, pair_mr_ids, pair_mr_types)):
                if mr_id == 'none' or not is_s:
                    source_indices.append(i)
                else:
                    if mr_type in metamorphic_indices:
                        metamorphic_indices[mr_type].append(i)

            if not source_indices:
                continue  # 没有源数据

            # 使用第一个源数据作为参考
            source_idx = source_indices[0]
            source_logit = pair_logits[source_idx:source_idx + 1]
            source_label = pair_labels[source_idx:source_idx + 1]
            source_prob = F.softmax(source_logit / temperature, dim=-1)

            # 计算源数据的CE损失
            ce_loss = gamma_ce * ce_loss_fn(source_logit, source_label).mean()
            total_batch_loss += ce_loss

            # 处理不同类型的蜕变数据
            # 1. inv-MR: 标签不变
            if metamorphic_indices['inv'] and alpha_inv > 0:
                inv_indices = metamorphic_indices['inv']
                inv_logits = pair_logits[inv_indices]
                inv_labels = pair_labels[inv_indices]

                # 一致性损失
                kl_inv = F.kl_div(F.log_softmax(inv_logits / temperature,
                                                dim=-1),
                                  source_prob.expand(len(inv_indices), -1),
                                  reduction='batchmean')

                # CE损失
                inv_ce = ce_loss_fn(inv_logits, inv_labels).mean()

                inv_loss = alpha_inv * (beta_consistency * kl_inv +
                                        gamma_ce * inv_ce)
                total_batch_loss += inv_loss

            # 2. flip-MR: 标签翻转
            if metamorphic_indices['flip'] and alpha_flip > 0:
                flip_indices = metamorphic_indices['flip']
                flip_logits = pair_logits[flip_indices]
                flip_labels = pair_labels[flip_indices]

                # 创建翻转目标分布
                flip_target_prob = source_prob.clone()
                flip_target_prob[:, [0, 2]] = flip_target_prob[:, [2, 0]]

                # 一致性损失
                kl_flip = F.kl_div(F.log_softmax(flip_logits / temperature,
                                                 dim=-1),
                                   flip_target_prob.expand(
                                       len(flip_indices), -1),
                                   reduction='batchmean')

                # CE损失
                flip_ce = ce_loss_fn(flip_logits, flip_labels).mean()

                flip_loss = alpha_flip * (beta_consistency * kl_flip +
                                          gamma_ce * flip_ce)
                total_batch_loss += flip_loss

            # 3. neutral-MR: 标签变为中性
            if metamorphic_indices['neutral'] and alpha_neutral > 0:
                neutral_indices = metamorphic_indices['neutral']
                neutral_logits = pair_logits[neutral_indices]
                neutral_labels = pair_labels[neutral_indices]

                # 创建中性目标分布
                neutral_target_prob = torch.zeros_like(source_prob)
                neutral_target_prob[:, 1] = 1.0  # neutral类

                # 一致性损失
                kl_neutral = F.kl_div(
                    F.log_softmax(neutral_logits / temperature, dim=-1),
                    neutral_target_prob.expand(len(neutral_indices), -1),
                    reduction='batchmean')

                # CE损失
                neutral_ce = ce_loss_fn(neutral_logits, neutral_labels).mean()

                neutral_loss = alpha_neutral * (beta_consistency * kl_neutral +
                                                gamma_ce * neutral_ce)
                total_batch_loss += neutral_loss

        # L2正则化
        if l2_reg_weight > 0:
            l2_reg = l2_reg_weight * sum(
                torch.norm(p, p=2) for p in model_instance.parameters())
            total_batch_loss += l2_reg

        # 反向传播
        if not torch.isnan(
                total_batch_loss) and total_batch_loss.requires_grad:
            total_batch_loss.backward()
            clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_batch_loss.item()
            batch_count += 1
        else:
            logger.warning(
                f"无效的损失值: {total_batch_loss.item() if not torch.isnan(total_batch_loss) else 'NaN'}"
            )

    if batch_count == 0:
        logger.warning("没有处理任何batch！")
        return 0.0

    avg_loss = total_loss / batch_count
    logger.info(f"{phase_name} Epoch {epoch_idx+1} 结束")
    logger.info(f"处理了 {batch_count} 个batch，平均损失: {avg_loss:.4f}")
    logger.info(
        f"损失权重配置: inv={alpha_inv}, flip={alpha_flip}, neutral={alpha_neutral}, consistency={beta_consistency}"
    )

    return avg_loss


def example_usage_pairwise_training():
    """
    使用示例：如何使用新的蜕变关系训练函数
    """
    # 示例代码，展示如何使用新的训练函数

    # 方法1: 直接使用dataset
    """
    # 假设你有一个增强数据集
    augmented_dataset = load_augmented_dataset("path/to/augmented_data.json")
    
    # 使用dataset直接训练
    avg_loss = train_one_epoch_pairwise_metamorphic(
        model_instance=model,
        dataset=augmented_dataset,
        optimizer=optimizer,
        device=device,
        epoch_idx=epoch,
        phase_name="蜕变增强训练",
        alpha_inv=1.0,      # 标签不变的蜕变关系权重
        alpha_flip=1.0,     # 标签翻转的蜕变关系权重  
        alpha_neutral=1.0,  # 标签变中性的蜕变关系权重
        beta_consistency=0.5,  # 一致性损失权重
        gamma_ce=1.0,       # 交叉熵损失权重
        temperature=1.0,
        label_smoothing=0.1,
        l2_reg_weight=1e-4
    )
    """

    # 方法2: 使用特殊的pairwise dataloader
    """
    # 创建按pair分组的数据加载器
    pairwise_loader = create_pairwise_dataloader_from_dataset(
        dataset=augmented_dataset,
        tokenizer=model.tokenizer,
        max_pairs_per_batch=4  # 每个batch包含最多4个pair组
    )
    
    # 使用pairwise dataloader训练
    avg_loss = train_one_epoch_pairwise_dataloader(
        model_instance=model,
        pairwise_dataloader=pairwise_loader,
        optimizer=optimizer,
        device=device,
        epoch_idx=epoch,
        phase_name="蜕变增强训练",
        alpha_inv=1.0,
        alpha_flip=1.0,
        alpha_neutral=1.0,
        beta_consistency=0.5,
        gamma_ce=1.0,
        temperature=1.0,
        label_smoothing=0.1,
        l2_reg_weight=1e-4
    )
    """

    # 推荐配置:
    # - alpha_inv=1.0: 标签保持不变的蜕变（如同义词替换）应该与源数据预测一致
    # - alpha_flip=1.0: 标签翻转的蜕变（如添加矛盾信息）应该翻转预测
    # - alpha_neutral=0.8: 标签变中性的蜕变（如添加无关信息）应该增加中性预测
    # - beta_consistency=0.5: 一致性损失与CE损失平衡
    # - temperature=1.0: KL散度的温度参数，可以调节软化程度

    pass
