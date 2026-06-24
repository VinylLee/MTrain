#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 nli_model.py
 ============
 
 一个可即插即用的替代实现，用来取代 Hugging Face 官方的
 `AutoModelForSequenceClassification`，并将损失函数的计算
 委托给用户自定义的函数或模块。
 
 主要特点
 --------
 - **SequenceClassifierWithCustomLoss**
   这是一个轻量级的 `nn.Module` 包装器：它先调用底层预训练模型
   得到 *logits*，随后在 `forward()` 中调用你传入的 `criterion`
   （既可以是普通的 Python 可调用对象，也可以是 `nn.Module`）。
   计算出的损失被封装到
   `transformers.modeling_outputs.SequenceClassifierOutput` 中，
   因而 `outputs.loss` 的行为与官方模型保持完全一致。
 
 - **NLIModel**
   封装了以下组件：
 
   * 分词器（Tokenizer）
   * （已包装的）模型
   * 设备管理
   * 随机种子固定
   * 可选的保存 / 加载助手
 
 因此，你可以保持现有的训练循环不变：只需在调用模型时提供
 `labels`，损失将按照你的自定义逻辑被自动计算。
 """

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (  # type: ignore
    AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore
from mylogging import logger  # type: ignore

# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子以保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 关闭 cuDNN 的随机性，开启确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------
# 核心类：支持自定义损失的包装器
# ---------------------------------------------------------------


class SequenceClassifierWithCustomLoss(nn.Module):
    """序列分类模型包装器，可内联任意自定义损失函数。

    参数
    ------
    base_model : AutoModelForSequenceClassification
        已初始化完成的序列分类模型（调用时 *不* 传入 ``labels``）。
    criterion : Callable
        任意可调用对象，接受 ``(logits, labels, **loss_kwargs)`` 并返回
        *标量* 张量。如果它是 ``nn.Module`` 的子类，将按
        ``criterion(logits, labels)`` 调用。
    loss_kwargs : dict[str, Any] | None, default None
        传递给 *普通可调用* ``criterion`` 的额外参数；若 ``criterion``
        是 ``nn.Module``，则忽略。

    备注
    ----
    * 该包装器对设备无感知：调用者可以自由地将其或底层模型
      ``.to(device)``。
    * 底层模型返回的 ``hidden_states`` 与 ``attentions`` 不会被丢弃，
      因此诸如 *Trainer* 等下游组件仍可正常工作。
    """

    def __init__(
        self,
        base_model: AutoModelForSequenceClassification,
        criterion: Callable,
        loss_kwargs: Optional[Dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.criterion = criterion
        self.loss_kwargs = loss_kwargs or {}

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        *args,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """前向传播接口。

        参数
        ------
        *args, **kwargs
            与 Hugging Face 模型相同的输入关键字（``input_ids``、
            ``attention_mask``、``token_type_ids`` 等）。
        labels : torch.Tensor | None, default None
            *可选*。如果提供，则计算自定义损失。
            
            **支持的形状**
            
            - *类别索引* —— 形状 ``(batch,)``，``dtype=torch.long``。
              适用于 ``nn.CrossEntropyLoss``。
            - *One‑hot / soft label* —— 形状 ``(batch, num_labels)``，
              浮点数。适用于 MSE / KL / label smoothing 等。

        返回
        ------
        SequenceClassifierOutput
            Hugging Face 的标准输出；若 ``labels`` 被提供，``loss`` 字段
            将由 *自定义* ``criterion`` 填充。
        """
        # 1. 先 **不带 labels** 调用底层预训练模型
        base_out = self.base_model(*args, **kwargs)

        logits = base_out.logits
        loss = None

        # 2. 若用户提供了 labels，则调用自定义损失
        if labels is not None:
            if isinstance(self.criterion, nn.Module):
                loss = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels, **self.loss_kwargs)

        # 3. 按 HF 习惯封装输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_out.hidden_states,
            attentions=base_out.attentions,
        )


# ---------------------------------------------------------------
# 高层便利封装（Tokenizer + Model）
# ---------------------------------------------------------------


class NLIModel:
    """自然语言推理（NLI）微调的一站式容器。

    该类让你可以在沿用原有 Hugging Face 工作流的同时，
    轻松替换损失函数。

    示例
    ----
    >>> model = NLIModel(criterion=nn.CrossEntropyLoss())
    >>> inputs = model.tokenizer("A", "B", return_tensors="pt",
    ...                          truncation=True, padding=True)
    >>> labels = torch.tensor([2])  # "contradiction"
    >>> out = model.model(**inputs, labels=labels)
    >>> out.loss.backward()
    """

    # ------------------------------------------------------------------
    # 构造函数
    # ------------------------------------------------------------------
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        num_labels: int = 3,
        cache_dir: str = "/home/ubuntu/WoYaoFaA/MetTrain/nli_model_cache",
        criterion: Optional[Callable] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        local_files_only: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.seed = seed
        set_seed(seed)

        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir  # 设置设备
        if device is not None:
            self.device = torch.device(device)
            # 如果是CUDA设备，也设置PyTorch的当前设备
            if device.startswith("cuda") and torch.cuda.is_available():
                if ":" in device:
                    gpu_id = int(device.split(":")[1])
                    torch.cuda.set_device(gpu_id)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.loss_kwargs = loss_kwargs or {}

        logger.info("Device set to %s", self.device)
        os.makedirs(self.cache_dir, exist_ok=True)

        # -----------------------------------------------------------
        # 判断缓存：若本地已有 config，则强制使用本地文件
        # -----------------------------------------------------------
        if not local_files_only:
            try:
                PretrainedConfig.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                local_files_only = True
                logger.info("Found cached model for %s — using local files",
                            model_name)
            except Exception:  # noqa: BLE001
                logger.info("No local cache found. Model will be downloaded.")

        # -----------------------------------------------------------
        # 初始化分词器
        # -----------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        # -----------------------------------------------------------
        # 初始化基础模型
        # -----------------------------------------------------------
        _base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(self.device)

        # -----------------------------------------------------------
        # 设置损失函数
        # -----------------------------------------------------------
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            logger.info("Default loss = CrossEntropyLoss")
        else:
            logger.info("Using custom loss: %s", criterion.__class__.__name__)

        # -----------------------------------------------------------
        # 包装模型
        # -----------------------------------------------------------
        self.model: SequenceClassifierWithCustomLoss = SequenceClassifierWithCustomLoss(
            _base, criterion, self.loss_kwargs).to(self.device)
        logger.info("Model %s loaded and wrapped.", model_name)

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------
    def to_device(self, batch: Dict[str,
                                    torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将一个 batch 输入移动到当前模型所在设备（CPU / GPU）。"""
        return {k: v.to(self.device) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # （可选）模型保存 / 加载
    # ------------------------------------------------------------------
    def save_model(self, save_dir: str) -> None:
        """保存 **基础 HF 模型** 与 **分词器**。

        仅保存 *内部* 预训练权重；包装器将在 :py:meth:`load_model` 中重建。
        """
        os.makedirs(save_dir, exist_ok=True)
        # 解包后保存
        self.model.base_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info("Model & tokenizer saved to %s", save_dir)

    @classmethod
    def load_model(
        cls,
        load_dir: str,
        *,
        criterion: Optional[Callable] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "NLIModel":
        """从 :py:meth:`save_model` 的输出路径重建 ``NLIModel``。

        其余关键字参数会被透传给构造函数（例如设备、随机种子等）。
        """
        logger.info(f"加载模型: {load_dir}")
        return cls(
            model_name=load_dir,
            local_files_only=True,
            criterion=criterion,
            loss_kwargs=loss_kwargs,
            **kwargs,
        )


# # ---------------------------------------------------------------------------
# # demo / 快速自检
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":

#     # -------------------------------------------------------
#     # 1. 示例自定义 *函数式* 损失（CrossEntropy + margin）
#     # -------------------------------------------------------
#     def ce_with_margin(logits: torch.Tensor,
#                        labels: torch.Tensor,
#                        *,
#                        margin=0.0):
#         """示例分类损失：交叉熵 + 固定边际 ``margin``。

#         接收类别索引形式的标签，适用于标准的 NLI 三分类任务。

#         参数
#         ------
#         logits : torch.Tensor
#             模型输出，形状为 (batch_size, num_classes)
#         labels : torch.Tensor
#             类别索引，形状为 (batch_size,)
#         margin : float
#             可选的额外损失边际，默认 0.0

#         返回
#         ------
#         torch.Tensor
#             单个标量损失值
#         """
#         ce_loss = nn.CrossEntropyLoss()(logits, labels)
#         return ce_loss + margin

#     # -------------------------------------------------------
#     # 2. 构建包装器
#     # -------------------------------------------------------
#     nli = NLIModel(
#         criterion=ce_with_margin,
#         loss_kwargs={"margin": 3333.3},
#     )

#     # 示例前提 / 假设
#     premise = "A man inspects the uniform of a figure in some East Asian country."
#     hypothesis = "The man is sleeping."

#     # 分词
#     inputs = nli.tokenizer(
#         premise,
#         hypothesis,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#     )
#     inputs = nli.to_device(inputs)

#     fake_label = torch.tensor([2], device=nli.device)

#     # -------------------------------------------------------
#     # 3. 前向传播（带 labels）→ 自动计算损失
#     # -------------------------------------------------------
#     nli.model.train()
#     out = nli.model(**inputs, labels=fake_label)

#     logger.info("=== 原始模型输出 ===")
#     logger.info("Logits: %s", out.logits)
#     logger.info("Built‑in loss: %s", out.loss)  # 由包装器计算
#     logger.info(
#         "my loss: %s",
#         ce_with_margin(logits=out.logits, labels=fake_label, margin=3333.3),
#     )  # 手动计算，对比
#     out.loss.backward()  # 反向传播无缝衔接

#     # -------------------------------------------------------
#     # 4. 保存并重新加载
#     # -------------------------------------------------------
#     save_dir = "/home/ubuntu/MTrain/MTrainNLI/nli_saved_model/NLImodel_test_model"
#     nli.save_model(save_dir)

#     reloaded = NLIModel.load_model(
#         load_dir=save_dir,
#         criterion=ce_with_margin,
#         loss_kwargs={"margin": 3333.3},
#     )
#     with torch.no_grad():
#         reloaded_out = reloaded.model(**inputs, labels=fake_label)

#     logger.info("=== 加载模型输出 ===")
#     logger.info("Reloaded logits: %s", reloaded_out.logits)
#     logger.info("Reloaded loss: %s", reloaded_out.loss)
