#!/usr/bin/env python3
"""
多信号联合损失快速验证脚本

用于快速测试多信号联合损失函数的可行性，
使用小数据集进行概念验证。
"""

import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict

# 尝试导入，如果失败则使用简单的logger
try:
    from NLI_train_one_epoch import train_one_epoch_multi_signal_joint
    from mylogging import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 如果无法导入，我们在这里定义一个简化版本的训练函数
    def train_one_epoch_multi_signal_joint(model_instance,
                                           dataloader,
                                           optimizer,
                                           device,
                                           epoch_idx,
                                           phase_name="多信号联合训练",
                                           alpha=1.0,
                                           beta=0.3,
                                           gamma=0.5,
                                           **kwargs):
        """简化版的多信号联合训练函数"""
        model_instance.train()
        total_loss = 0.0

        # 蜕变关系转换矩阵
        transform_matrices = {
            "inv":
            torch.eye(3, device=device),
            "flip":
            torch.tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]],
                         device=device),
            "inv-flip":
            torch.tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]],
                         device=device)
        }

        for batch in tqdm(dataloader,
                          desc=f"{phase_name} Epoch {epoch_idx+1}"):
            optimizer.zero_grad()

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

            # 前向传播
            outputs = model_instance(**batch)
            logits = outputs.logits

            # 基础CE损失
            ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
            ce_losses = ce_loss_fn(logits, labels)
            weighted_ce = (ce_losses * weights).mean()

            # MR一致性损失
            mr_loss = torch.tensor(0.0, device=device)
            pair_groups = defaultdict(list)

            for i, pid in enumerate(pair_id):
                pair_groups[pid].append(i)

            mr_count = 0
            for pid, indices in pair_groups.items():
                if len(indices) < 2:
                    continue

                orig_idx = None
                syn_indices = []

                for idx in indices:
                    if not is_syn[idx]:
                        orig_idx = idx
                    else:
                        syn_indices.append(idx)

                if orig_idx is None or len(syn_indices) == 0:
                    continue

                orig_logits = logits[orig_idx]
                orig_probs = F.softmax(orig_logits, dim=-1)

                for syn_idx in syn_indices:
                    syn_logits = logits[syn_idx]

                    mr_t = mr_type[syn_idx]
                    if mr_t in transform_matrices:
                        T = transform_matrices[mr_t]
                        target_probs = torch.matmul(orig_probs, T)

                        kl_loss = F.kl_div(F.log_softmax(syn_logits, dim=-1),
                                           target_probs.detach(),
                                           reduction='batchmean',
                                           log_target=False)
                        mr_loss += kl_loss
                        mr_count += 1

            if mr_count > 0:
                mr_loss = mr_loss / mr_count

            # 总损失（简化版，仅包含CE和MR）
            total_batch_loss = weighted_ce + alpha * mr_loss

            if not torch.isnan(
                    total_batch_loss) and total_batch_loss.requires_grad:
                total_batch_loss.backward()
                clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += total_batch_loss.item()
            else:
                logger.warning(f"无效损失: {total_batch_loss.item()}")

        avg_loss = total_loss / len(dataloader)
        logger.info(
            f"{phase_name} Epoch {epoch_idx+1} 完成，平均损失: {avg_loss:.4f}")
        return avg_loss


class QuickNLIDataset(Dataset):
    """快速测试用的NLI数据集"""

    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 编码文本
        encoding = self.tokenizer(item['premise'],
                                  item['hypothesis'],
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long),
            'pair_id': item['pair_id'],
            'is_syn': torch.tensor(item['is_syn'], dtype=torch.bool),
            'mr_type': item.get('mr_type', 'inv'),
            'weight': torch.tensor(item.get('weight', 1.0), dtype=torch.float)
        }


class QuickNLIModel(nn.Module):
    """快速测试用的NLI模型"""

    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids,
                            attention_mask,
                            output_hidden_states=True)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # 模拟transformers输出格式
        class ModelOutput:

            def __init__(self, loss, logits, hidden_states):
                self.loss = loss
                self.logits = logits
                self.hidden_states = hidden_states

        return ModelOutput(loss, logits, outputs.hidden_states)


def create_sample_data():
    """创建示例数据，模拟你的蜕变关系数据格式"""

    sample_data = [
        # 原始样本
        {
            "premise": "a man squinting with two women behind him",
            "hypothesis": "a man squinting with two women behind him",
            "label": 0,  # entailment
            "pair_id": 1,
            "is_syn": False,
            "mr_type": "inv",
            "weight": 0.01
        },
        # 反义词替换 (flip)
        {
            "premise": "a man squinting with two women behind him",
            "hypothesis": "A man glaring with two men behind him.",
            "label": 2,  # contradiction  
            "pair_id": 1,
            "is_syn": True,
            "mr_type": "flip",
            "weight": 1.0
        },
        # 信息提取 (inv)
        {
            "premise": "a man squinting with two women behind him",
            "hypothesis": "A man stands with two women.",
            "label": 0,  # entailment
            "pair_id": 1,
            "is_syn": True,
            "mr_type": "inv",
            "weight": 1.0
        },
        # 计数蕴含 (inv-flip)
        {
            "premise": "a man squinting with two women behind him",
            "hypothesis": "3 persons are present",
            "label": 0,  # entailment
            "pair_id": 1,
            "is_syn": True,
            "mr_type": "inv",
            "weight": 1.0
        },

        # 第二组数据
        {
            "premise": "two dogs are playing in the park",
            "hypothesis": "two dogs are playing in the park",
            "label": 0,
            "pair_id": 2,
            "is_syn": False,
            "mr_type": "inv",
            "weight": 0.01
        },
        {
            "premise": "two dogs are playing in the park",
            "hypothesis": "two cats are sleeping at home",
            "label": 2,
            "pair_id": 2,
            "is_syn": True,
            "mr_type": "flip",
            "weight": 1.0
        },
        {
            "premise": "two dogs are playing in the park",
            "hypothesis": "animals are in the park",
            "label": 0,
            "pair_id": 2,
            "is_syn": True,
            "mr_type": "inv",
            "weight": 1.0
        },

        # 第三组数据
        {
            "premise": "the weather is sunny today",
            "hypothesis": "the weather is sunny today",
            "label": 0,
            "pair_id": 3,
            "is_syn": False,
            "mr_type": "inv",
            "weight": 0.01
        },
        {
            "premise": "the weather is sunny today",
            "hypothesis": "the weather is rainy today",
            "label": 2,
            "pair_id": 3,
            "is_syn": True,
            "mr_type": "flip",
            "weight": 1.0
        },
        {
            "premise": "the weather is sunny today",
            "hypothesis": "it is a good day",
            "label": 0,
            "pair_id": 3,
            "is_syn": True,
            "mr_type": "inv",
            "weight": 1.0
        }
    ]

    return sample_data


def collate_fn(batch):
    """批处理函数"""
    keys = batch[0].keys()
    result = {}

    for key in keys:
        if key in ['pair_id', 'mr_type']:
            result[key] = [item[key] for item in batch]
        else:
            result[key] = torch.stack([item[key] for item in batch])

    return result


def quick_test():
    """快速测试多信号联合损失函数"""

    logger.info("开始快速测试多信号联合损失函数")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建示例数据
    sample_data = create_sample_data()
    logger.info(f"创建了 {len(sample_data)} 个示例样本")

    # 数据分析
    pair_counts = {}
    mr_type_counts = {}
    for item in sample_data:
        pair_id = item['pair_id']
        mr_type = item['mr_type']
        pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
        mr_type_counts[mr_type] = mr_type_counts.get(mr_type, 0) + 1

    logger.info(f"Pair ID分布: {pair_counts}")
    logger.info(f"MR类型分布: {mr_type_counts}")

    # 初始化模型和数据
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = QuickNLIModel('bert-base-uncased', num_labels=3)
    model.to(device)

    dataset = QuickNLIDataset(sample_data, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    logger.info("模型和数据加载完成")

    # 测试不同的超参数配置
    test_configs = [
        {
            "name": "基线(α=0,β=0,γ=0)",
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0
        },
        {
            "name": "仅MR一致性(α=1.0)",
            "alpha": 1.0,
            "beta": 0.0,
            "gamma": 0.0
        },
        {
            "name": "MR+对比(α=1.0,β=0.3)",
            "alpha": 1.0,
            "beta": 0.3,
            "gamma": 0.0
        },
        {
            "name": "完整多信号(α=1.0,β=0.3,γ=0.2)",
            "alpha": 1.0,
            "beta": 0.3,
            "gamma": 0.2
        },
    ]

    results = {}

    for config in test_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试配置: {config['name']}")
        logger.info(f"{'='*50}")

        # 重新初始化模型
        model = QuickNLIModel('bert-base-uncased', num_labels=3)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        losses = []

        # 训练几个epoch
        for epoch in range(3):
            try:
                avg_loss = train_one_epoch_multi_signal_joint(
                    model,
                    dataloader,
                    optimizer,
                    device,
                    epoch,
                    phase_name=config['name'],
                    alpha=config['alpha'],
                    beta=config['beta'],
                    gamma=config['gamma'],
                    mr_temperature=1.0,
                    contrast_temperature=0.1,
                    confidence_threshold=0.8)
                losses.append(avg_loss)
                logger.info(f"Epoch {epoch+1}: 损失 = {avg_loss:.4f}")

            except Exception as e:
                logger.error(f"训练出错: {str(e)}")
                import traceback
                traceback.print_exc()
                break

        results[config['name']] = {
            'losses': losses,
            'final_loss': losses[-1] if losses else float('inf')
        }

        # 简单评估：计算训练数据上的准确率
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop('labels').to(device)
                pair_id = batch.pop('pair_id')
                is_syn = batch.pop('is_syn').to(device)
                mr_type = batch.pop('mr_type')
                weight = batch.pop('weight').to(device)

                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        results[config['name']]['accuracy'] = accuracy

        logger.info(
            f"{config['name']} - 最终损失: {results[config['name']]['final_loss']:.4f}, "
            f"训练准确率: {accuracy:.4f}")

    # 结果总结
    logger.info(f"\n{'='*60}")
    logger.info("快速测试结果总结")
    logger.info(f"{'='*60}")

    for name, result in results.items():
        logger.info(f"{name}:")
        logger.info(f"  最终损失: {result['final_loss']:.4f}")
        logger.info(f"  训练准确率: {result['accuracy']:.4f}")
        logger.info(f"  损失变化: {result['losses']}")
        logger.info("")

    # 保存结果
    with open('quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("快速测试完成！结果已保存到 quick_test_results.json")

    return results


if __name__ == "__main__":
    quick_test()
