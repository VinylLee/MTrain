#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_composite_mr_loss.py
========================

测试新添加的复合MR驱动的多目标损失函数

主要测试内容：
1. SupConLoss的基本功能
2. train_one_epoch_composite_mr_loss的完整功能
3. train_one_epoch_simple_composite_mr的简化版本
4. 各种参数组合下的表现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NLI_train_one_epoch import (SupConLoss, train_one_epoch_composite_mr_loss,
                                 train_one_epoch_simple_composite_mr,
                                 create_composite_mr_trainer)
from mylogging import logger


class SimpleNLIModel(nn.Module):
    """简单的NLI模型用于测试"""

    def __init__(self, vocab_size=1000, hidden_size=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 嵌入
        embedded = self.embedding(input_ids)

        # 编码
        if attention_mask is not None:
            # 简单处理：使用attention_mask的长度信息
            lengths = attention_mask.sum(dim=1)
            packed = nn.utils.rnn.pack_padded_sequence(embedded,
                                                       lengths.cpu(),
                                                       batch_first=True,
                                                       enforce_sorted=False)
            output, (hidden, _) = self.encoder(packed)
            features = hidden[-1]  # 使用最后一层隐状态
        else:
            output, (hidden, _) = self.encoder(embedded)
            features = hidden[-1]

        # 分类
        features = self.dropout(features)
        logits = self.classifier(features)

        # 计算损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # 创建类似HuggingFace的输出结构
        class ModelOutput:

            def __init__(self, loss, logits, hidden_states=None):
                self.loss = loss
                self.logits = logits
                self.hidden_states = hidden_states

        # 添加hidden_states用于对比学习
        hidden_states = [embedded, features.unsqueeze(1)]  # 模拟多层输出

        return ModelOutput(loss=loss,
                           logits=logits,
                           hidden_states=hidden_states)


def create_test_data(batch_size=16, seq_len=32, vocab_size=1000):
    """创建测试数据"""

    # 生成随机数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 3, (batch_size, ))  # 3分类

    # 创建MR相关信息
    is_syn = torch.zeros(batch_size, dtype=torch.bool)
    pair_id = list(range(batch_size))
    mr_type = ["none"] * batch_size

    # 设置一些样本为合成样本
    num_syn = batch_size // 3
    is_syn[:num_syn] = True

    # 设置MR类型
    for i in range(num_syn):
        if i % 2 == 0:
            mr_type[i] = "inv"
        else:
            mr_type[i] = "flip"

    # 为合成样本设置对应的原始样本pair_id
    for i in range(num_syn):
        pair_id[i] = i + num_syn  # 对应到原始样本

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'is_syn': is_syn,
        'pair_id': pair_id,
        'mr_type': mr_type
    }


def test_supcon_loss():
    """测试SupConLoss"""
    logger.info("=== 测试SupConLoss ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    supcon = SupConLoss(temperature=0.07).to(device)

    # 创建测试数据
    batch_size = 8
    feature_dim = 64
    features = torch.randn(batch_size, feature_dim).to(device)
    labels = torch.randint(0, 3, (batch_size, )).to(device)
    mr_types = ["none", "inv", "flip", "none", "inv", "none", "flip", "none"]
    is_syn = torch.tensor([False, True, True, False, True, False, True,
                           False]).to(device)

    try:
        loss = supcon(features, labels, mr_types, is_syn)
        logger.info(f"SupConLoss 计算成功: {loss.item():.4f}")
        return True
    except Exception as e:
        logger.error(f"SupConLoss 测试失败: {e}")
        return False


def test_composite_mr_loss():
    """测试完整的复合MR损失函数"""
    logger.info("=== 测试完整复合MR损失函数 ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = SimpleNLIModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 创建数据
    batch_data = create_test_data(batch_size=16)
    dataset = TensorDataset(batch_data['input_ids'],
                            batch_data['attention_mask'], batch_data['labels'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 手动添加MR信息到batch
    def add_mr_info(batch_data, batch_idx):
        start_idx = batch_idx * 8
        end_idx = start_idx + len(batch_data['input_ids'])

        batch_dict = {
            'input_ids': batch_data['input_ids'],
            'attention_mask': batch_data['attention_mask'],
            'labels': batch_data['labels']
        }

        # 添加MR信息
        batch_dict['is_syn'] = batch_data['is_syn'][start_idx:end_idx]
        batch_dict['pair_id'] = batch_data['pair_id'][start_idx:end_idx]
        batch_dict['mr_type'] = batch_data['mr_type'][start_idx:end_idx]

        return batch_dict

    try:
        # 模拟一个batch的训练
        for batch_idx, (input_ids, attention_mask,
                        labels) in enumerate(dataloader):
            batch_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'is_syn':
                batch_data['is_syn'][batch_idx * 8:(batch_idx + 1) * 8],
                'pair_id':
                batch_data['pair_id'][batch_idx * 8:(batch_idx + 1) * 8],
                'mr_type':
                batch_data['mr_type'][batch_idx * 8:(batch_idx + 1) * 8]
            }

            # 模拟dataloader
            mock_dataloader = [batch_dict]

            avg_loss = train_one_epoch_composite_mr_loss(
                model_instance=model,
                dataloader=mock_dataloader,
                optimizer=optimizer,
                device=device,
                epoch_idx=0,
                phase_name="测试",
                alpha_inv=1.0,
                alpha_flip=1.0,
                beta=0.5,
                lambda_neu=0.1)

            logger.info(f"完整复合MR损失函数测试成功，平均损失: {avg_loss:.4f}")
            break  # 只测试一个batch

        return True
    except Exception as e:
        logger.error(f"完整复合MR损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_composite_mr():
    """测试简化版复合MR损失函数"""
    logger.info("=== 测试简化版复合MR损失函数 ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = SimpleNLIModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 创建数据
    batch_data = create_test_data(batch_size=8)

    try:
        # 准备batch
        batch_dict = {
            'input_ids': batch_data['input_ids'],
            'attention_mask': batch_data['attention_mask'],
            'labels': batch_data['labels'],
            'is_syn': batch_data['is_syn'],
            'pair_id': batch_data['pair_id'],
            'mr_type': batch_data['mr_type']
        }

        mock_dataloader = [batch_dict]

        avg_loss = train_one_epoch_simple_composite_mr(
            model_instance=model,
            dataloader=mock_dataloader,
            optimizer=optimizer,
            device=device,
            epoch_idx=0,
            phase_name="简化测试",
            alpha=1.0,
            beta=0.3,
            use_supcon=True,
            use_neutral_reg=True)

        logger.info(f"简化版复合MR损失函数测试成功，平均损失: {avg_loss:.4f}")
        return True
    except Exception as e:
        logger.error(f"简化版复合MR损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_factory():
    """测试训练函数工厂"""
    logger.info("=== 测试训练函数工厂 ===")

    try:
        # 创建不同配置的训练器
        trainer1 = create_composite_mr_trainer(mr_weight=1.5,
                                               contrast_weight=0.5,
                                               enable_contrast=True)

        trainer2 = create_composite_mr_trainer(mr_weight=1.0,
                                               contrast_weight=0.0,
                                               enable_contrast=False,
                                               enable_neutral_reg=False)

        logger.info("训练函数工厂创建成功")
        logger.info(f"训练器1: {trainer1}")
        logger.info(f"训练器2: {trainer2}")
        return True
    except Exception as e:
        logger.error(f"训练函数工厂测试失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("开始测试复合MR损失函数...")

    tests = [
        ("SupConLoss", test_supcon_loss),
        ("简化版复合MR损失", test_simple_composite_mr),
        ("训练函数工厂", test_trainer_factory),
        ("完整复合MR损失", test_composite_mr_loss),  # 最后测试，因为比较复杂
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")

        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"测试 {test_name} 出现异常: {e}")
            results[test_name] = False

    # 输出测试结果
    logger.info(f"\n{'='*50}")
    logger.info("测试结果汇总")
    logger.info(f"{'='*50}")

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())
    logger.info(
        f"\n总测试数: {total_tests}, 通过: {passed_tests}, 失败: {total_tests - passed_tests}"
    )

    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过！")
        return True
    else:
        logger.warning(f"⚠️  有 {total_tests - passed_tests} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
