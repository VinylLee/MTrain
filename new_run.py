#!/usr/bin/env python3
"""
多信号联合损失函数渐进式实验脚本

实验设计：
1. 基线实验：仅CE损失
2. 第一步：CE + MR一致性损失  
3. 第二步：CE + MR一致性 + 监督对比损失
4. 第三步：完整的多信号联合损失

每个阶段都会评估模型性能，逐步验证各个损失组件的贡献。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime
import numpy as np
import random

from transformers import get_linear_schedule_with_warmup

# 假设你已有的模块
from NLIdataset import NLIDataset
from NLImodel import NLIModel
from NLI_train_one_epoch import (train_one_epoch_labeled_CE,
                                 train_one_epoch_dual_consistency,
                                 train_one_epoch_multi_signal_joint,
                                 train_one_epoch_multi_signal_joint_RDrop)
from NLItest import test_nli
from mylogging import logger

# 导入随机种子检测工具
from seed_checker import set_random_seed, print_seed_status, detect_seed_status


class MultiSignalExperiment:

    def __init__(self, config):
        self.config = config

        # 随机种子设置
        if 'random_seed' in config:
            logger.info(f"设置随机种子: {config['random_seed']}")
            set_random_seed(config['random_seed'],
                            deterministic=config.get('deterministic', True))
            # 检查种子设置状态
            seed_status = detect_seed_status()
            if seed_status.get('likely_seeded', False):
                logger.info("✓ 随机种子设置成功")
            else:
                logger.warning("⚠ 随机种子可能未正确设置")
        else:
            logger.warning("未设置随机种子，实验结果可能不可重现")
            print_seed_status()

        # 设备设置逻辑
        device_config = config['device']
        if device_config == 0:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        elif device_config == 1:
            self.device = torch.device(
                'cuda:1' if torch.cuda.is_available() else 'cpu')
        elif device_config == -1:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device_config, str):
            self.device = torch.device(device_config)
        else:
            self.device = torch.device('cpu')

        # 如果指定了CUDA设备，检查设备是否可用
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA不可用，回退到CPU")
                self.device = torch.device('cpu')
            elif self.device.index is not None and self.device.index >= torch.cuda.device_count(
            ):
                logger.warning(f"CUDA设备{self.device.index}不存在，回退到cuda:0")
                self.device = torch.device('cuda:0')

        self.results = {}

        # 创建实验目录
        # 获取train_data_path的文件名
        import os
        dataname = os.path.basename(self.config['train_data_path'])
        datainfo = os.path.basename(
            os.path.dirname(self.config['train_data_path']))
        self.exp_dir = f"experiments/{datainfo}_{dataname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.exp_dir, exist_ok=True)

        logger.info(f"实验目录: {self.exp_dir}")
        logger.info(f"使用设备: {self.device}")

    def _get_model_save_path(self, model_name):
        """获取模型保存路径"""
        if self.config.get('save_model_dir'):
            # 如果指定了保存目录，使用指定目录
            save_dir = self.config['save_model_dir']
            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, model_name)
            return model_save_path
        else:
            # 否则使用默认的实验目录
            return os.path.join(self.exp_dir, model_name)

    def load_data(self, shuffle: bool = False):
        """加载数据集"""
        logger.info("加载数据集...")

        # 根据配置决定是否使用增强数据
        if 'use_augmented_data' in self.config and self.config[
                'use_augmented_data']:
            # 使用预先增强好的数据
            self.dataset_handler = NLIDataset(
                dataset_name=self.config.get('dataset_name', 'snli'),
                labeled_ratio=self.config.get('labeled_ratio', 0.1),
                unlabeled_ratio=self.config.get('unlabeled_ratio', 0.2),
                use_data_augmentation=True,
                augmented_dir=self.config.get('augmented_dir', None),
                llm_model_name=self.config.get('llm_model_name',
                                               'llama-3.2-3b-instruct'))
            # 获取增强后的数据集
            self.train_dataset = self.dataset_handler.get_augmented_dataset()
            if self.train_dataset is None:
                logger.warning("增强数据集不可用，使用有标签数据集")
                self.train_dataset = self.dataset_handler.get_labeled_dataset()
        else:
            # 使用指定路径的数据文件
            if 'train_data_path' in self.config:
                # 如果指定了数据路径，我们需要创建一个简单的数据集包装器
                self.train_dataset = self._load_json_dataset(
                    self.config['train_data_path'])
            else:
                # 使用默认的数据集加载方式
                self.dataset_handler = NLIDataset(
                    dataset_name=self.config.get('dataset_name', 'snli'),
                    labeled_ratio=self.config.get('labeled_ratio', 0.1),
                    use_data_augmentation=False)
                self.train_dataset = self.dataset_handler.get_labeled_dataset()

        # 加载验证数据集
        if 'val_data_path' in self.config:
            self.val_dataset = self._load_json_dataset(
                self.config['val_data_path'])
        else:
            if not hasattr(self, 'dataset_handler'):
                self.dataset_handler = NLIDataset(
                    dataset_name=self.config.get('dataset_name', 'snli'),
                    labeled_ratio=self.config.get('labeled_ratio', 0.1),
                    use_data_augmentation=False)
            self.val_dataset = self.dataset_handler.get_test_dataset()

        # 创建数据加载器（需要自定义collate_fn）
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.config['batch_size'],
                                       shuffle=shuffle,
                                       collate_fn=self._collate_fn)

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.config['batch_size'],
                                     shuffle=False,
                                     collate_fn=self._collate_fn)

        logger.info(f"训练集大小: {len(self.train_dataset)}")
        logger.info(f"验证集大小: {len(self.val_dataset)}")

        # 分析数据分布
        self.analyze_data_distribution(self.train_dataset)

    def analyze_data_distribution(self, dataset):
        """分析数据分布，为超参数设置提供参考"""
        logger.info("分析数据分布...")

        label_counts = {0: 0, 1: 0, 2: 0}  # entailment, neutral, contradiction
        mr_type_counts = {}
        pair_id_counts = {}

        # 根据数据集类型选择不同的访问方式
        try:
            # 对于HuggingFace Dataset，直接使用索引访问
            if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                logger.info(f"检测到HuggingFace Dataset格式，数据集大小: {len(dataset)}")

                # 只分析前1000个样本以提高速度（如果数据集很大）
                sample_size = min(1000, len(dataset))
                logger.info(f"分析前 {sample_size} 个样本...")

                for i in range(sample_size):
                    item = dataset[i]

                    # 处理label
                    label = item.get('label', 0) if isinstance(
                        item, dict) else item['label']
                    label_counts[label] += 1

                    # 处理mr_type
                    mr_type = item.get('mr_type', 'none') if isinstance(
                        item, dict) else item.get('mr_type', 'none')
                    if mr_type is None:
                        mr_type = 'none'
                    mr_type_counts[mr_type] = mr_type_counts.get(mr_type,
                                                                 0) + 1

                    # 处理pair_id
                    pair_id = item.get('pair_id', 0) if isinstance(
                        item, dict) else item.get('pair_id', 0)
                    if pair_id is not None:
                        pair_id_counts[pair_id] = pair_id_counts.get(
                            pair_id, 0) + 1

            elif hasattr(dataset, 'data'):  # 如果是自定义数据集格式
                logger.info("检测到自定义数据集格式")
                for item in dataset.data:
                    label_counts[item.get('label', 0)] += 1

                    mr_type = item.get('mr_type', 'none')
                    mr_type_counts[mr_type] = mr_type_counts.get(mr_type,
                                                                 0) + 1

                    pair_id = item.get('pair_id', 0)
                    pair_id_counts[pair_id] = pair_id_counts.get(pair_id,
                                                                 0) + 1

            else:
                logger.warning("无法识别的数据集格式，跳过数据分析")
                return

        except Exception as e:
            logger.error(f"数据分析过程中出错: {e}")
            logger.info("尝试备用分析方法...")

            # 备用方法：尝试获取数据集的特征信息
            try:
                if hasattr(dataset, 'features'):
                    logger.info(f"数据集特征: {dataset.features}")
                if hasattr(dataset, 'column_names'):
                    logger.info(f"数据集列名: {dataset.column_names}")

                # 尝试获取第一个样本进行格式分析
                if len(dataset) > 0:
                    sample = dataset[0]
                    logger.info(f"样本类型: {type(sample)}")
                    if isinstance(sample, dict):
                        logger.info(f"样本字段: {list(sample.keys())}")
                    else:
                        logger.info(f"样本内容: {sample}")
            except Exception as e2:
                logger.error(f"备用分析也失败: {e2}")
            return

        logger.info(f"标签分布: {label_counts}")
        logger.info(f"蜕变类型分布: {mr_type_counts}")
        if pair_id_counts:
            logger.info(
                f"平均每个pair_id的样本数: {np.mean(list(pair_id_counts.values())):.2f}"
            )

        # 保存分析结果
        analysis = {
            'label_distribution':
            label_counts,
            'mr_type_distribution':
            mr_type_counts,
            'avg_samples_per_pair':
            np.mean(list(pair_id_counts.values())) if pair_id_counts else 0
        }

        with open(os.path.join(self.exp_dir, 'data_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)

    def create_model_and_optimizer(self):
        """创建模型和优化器"""
        self.NLImodel = NLIModel(model_name=self.config['model_name'],
                                 device=str(self.device))
        model = self.NLImodel.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config['weight_decay'],
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters,
                                lr=self.config['learning_rate'],
                                betas=(0.9, 0.999),
                                eps=1e-8)

        for i, group in enumerate(optimizer.param_groups):
            logger.info(
                f"param_group[{i}] lr={group['lr']}, weight_decay={group['weight_decay']}"
            )

        return model, optimizer

    def create_scheduler(self, optimizer, num_epochs, dataloader):
        """基于线性 warmup + 线性衰减策略创建学习率调度器"""
        if num_epochs <= 0 or dataloader is None:
            logger.info("调度器创建被跳过：epoch 数或 dataloader 无效")
            return None

        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            logger.warning("该 dataloader 不支持 len()，无法创建调度器")
            return None

        total_steps = num_epochs * steps_per_epoch
        if total_steps == 0:
            logger.warning("总训练步数为 0，跳过调度器创建")
            return None

        warmup_ratio = self.config.get('warmup_ratio', 0.06)
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)
        logger.info(
            f"创建调度器: total_steps={total_steps}, warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio:.2f}"
        )
        return scheduler

    def run_baseline_experiment(self):
        """基线实验：仅使用交叉熵损失"""
        logger.info("=" * 50)
        logger.info("开始基线实验：仅CE损失")
        logger.info("=" * 50)

        model, optimizer = self.create_model_and_optimizer()
        scheduler = self.create_scheduler(optimizer,
                                          self.config['baseline_epochs'],
                                          self.train_loader)

        best_acc = 0.0
        train_losses = []
        val_accs = []

        for epoch in range(self.config['baseline_epochs']):
            # 训练
            train_loss = train_one_epoch_labeled_CE(model,
                                                    self.train_loader,
                                                    optimizer,
                                                    self.device,
                                                    epoch,
                                                    phase_name="基线训练",
                                                    scheduler=scheduler)
            train_losses.append(train_loss)

            # 验证
            val_results = test_nli(model, self.NLImodel.tokenizer,
                                   self.val_dataset)
            val_acc = val_results['accuracy']
            val_accs.append(val_acc)

            logger.info(
                f"Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                if self.config.get('save_best_model', True):
                    model_save_path = self._get_model_save_path(
                        'baseline_best')
                    self.NLImodel.save_model(model_save_path)
                    logger.info(f"保存最佳基线模型到: {model_save_path}")

        self.results['baseline'] = {
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accs
        }

        logger.info(f"基线实验完成，最佳验证准确率: {best_acc:.4f}")
        return model, best_acc

    def run_mr_consistency_experiment(self):
        """第一步：CE + MR一致性损失"""
        logger.info("=" * 50)
        logger.info("开始MR一致性实验：CE + MR一致性损失")
        logger.info("=" * 50)

        model, optimizer = self.create_model_and_optimizer()
        scheduler = self.create_scheduler(optimizer, self.config['mr_epochs'],
                                          self.train_loader)

        best_acc = 0.0
        train_losses = []
        val_accs = []

        # 使用双重一致性训练（已包含MR一致性逻辑）
        for epoch in range(self.config['mr_epochs']):
            train_loss = train_one_epoch_dual_consistency(
                model,
                self.train_loader,
                optimizer,
                self.device,
                epoch,
                phase_name="双重一致性训练",
                alpha=0.5,  # 适中的MR权重
                lam_inv=1.0,
                lam_flip=1.0,
                scheduler=scheduler)
            train_losses.append(train_loss)

            val_results = test_nli(model, self.NLImodel.tokenizer,
                                   self.val_dataset)
            val_acc = val_results['accuracy']
            val_accs.append(val_acc)

            logger.info(
                f"Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                if self.config.get('save_best_model', True):
                    model_save_path = self._get_model_save_path(
                        'mr_consistency_best')
                    self.NLImodel.save_model(model_save_path)
                    logger.info(f"保存最佳MR一致性模型到: {model_save_path}")

        self.results['mr_consistency'] = {
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accs
        }

        logger.info(f"MR一致性实验完成，最佳验证准确率: {best_acc:.4f}")
        return model, best_acc

    def run_full_multi_signal_experiment(self):
        """第三步：完整的多信号联合损失"""
        logger.info("=" * 50)
        logger.info("开始完整多信号实验：CE + MR一致性 + 监督对比 + UDA")
        logger.info("=" * 50)

        model, optimizer = self.create_model_and_optimizer()
        scheduler = self.create_scheduler(optimizer,
                                          self.config['full_epochs'],
                                          self.train_loader)

        best_acc = 0.0
        train_losses = []
        val_accs = []

        # 渐进式调整超参数
        for epoch in range(self.config['full_epochs']):
            # 动态调整权重：初期重视MR一致性，后期增加对比学习
            alpha = 1.0  # MR一致性权重
            beta = min(0.5, 0.1 +
                       0.4 * epoch / self.config['full_epochs'])  # 对比学习权重
            gamma = 0.3 if epoch > self.config[
                'full_epochs'] // 2 else 0.0  # UDA权重

            train_loss = train_one_epoch_multi_signal_joint_RDrop(
                model,
                self.train_loader,
                optimizer,
                self.device,
                epoch,
                phase_name="多信号RDrop联合训练",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                mr_temperature=1.0,
                contrast_temperature=0.1,
                confidence_threshold=0.8,
                scheduler=scheduler)
            train_losses.append(train_loss)

            val_results = test_nli(model, self.NLImodel.tokenizer,
                                   self.val_dataset)
            val_acc = val_results['accuracy']
            val_accs.append(val_acc)

            logger.info(
                f"Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.4f}")
            logger.info(f"当前权重 - α:{alpha:.2f}, β:{beta:.2f}, γ:{gamma:.2f}")

            if val_acc > best_acc:
                best_acc = val_acc
                if self.config.get('save_best_model', True):
                    model_save_path = self._get_model_save_path(
                        'full_multi_signal_best')
                    self.NLImodel.save_model(model_save_path)
                    logger.info(f"保存最佳完整多信号模型到: {model_save_path}")

        self.results['full_multi_signal'] = {
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accs
        }

        logger.info(f"完整多信号实验完成，最佳验证准确率: {best_acc:.4f}")
        return model, best_acc

    def run_ablation_study(self):
        """消融实验：测试不同组件组合的效果"""
        logger.info("=" * 50)
        logger.info("开始消融实验")
        logger.info("=" * 50)

        # 测试不同的超参数组合
        ablation_configs = [
            {
                "name": "仅MR一致性",
                "alpha": 1.0,
                "beta": 0.0,
                "gamma": 0.0
            },
            {
                "name": "仅监督对比",
                "alpha": 0.0,
                "beta": 0.5,
                "gamma": 0.0
            },
            {
                "name": "MR+对比",
                "alpha": 1.0,
                "beta": 0.3,
                "gamma": 0.0
            },
            {
                "name": "MR+UDA",
                "alpha": 1.0,
                "beta": 0.0,
                "gamma": 0.5
            },
            {
                "name": "对比+UDA",
                "alpha": 0.0,
                "beta": 0.3,
                "gamma": 0.5
            },
        ]

        ablation_results = {}

        for config in ablation_configs:
            logger.info(f"测试配置: {config['name']}")

            model, optimizer = self.create_model_and_optimizer()
            scheduler = self.create_scheduler(optimizer,
                                              self.config['ablation_epochs'],
                                              self.train_loader)
            best_acc = 0.0

            for epoch in range(self.config['ablation_epochs']):
                train_loss = train_one_epoch_multi_signal_joint(
                    model,
                    self.train_loader,
                    optimizer,
                    self.device,
                    epoch,
                    phase_name=config['name'],
                    alpha=config['alpha'],
                    beta=config['beta'],
                    gamma=config['gamma'],
                    scheduler=scheduler)

                val_results = test_nli(model, self.NLImodel.tokenizer,
                                       self.val_dataset)
                val_acc = val_results['accuracy']

                if val_acc > best_acc:
                    best_acc = val_acc

            ablation_results[config['name']] = best_acc
            logger.info(f"{config['name']} 最佳准确率: {best_acc:.4f}")

        self.results['ablation'] = ablation_results

    def generate_report(self):
        """生成实验报告"""
        logger.info("生成实验报告...")
        time = datetime.now().isoformat()
        report = {
            'experiment_config': self.config,
            'results': self.results,
            'timestamp': time
        }

        # 保存详细结果
        with open(
                os.path.join(self.exp_dir, f'experiment_results_{time}.json'),
                'w') as f:
            json.dump(report, f, indent=2)

        # 生成简要报告
        summary = "多信号联合损失实验总结\n" + "=" * 40 + "\n"

        if 'baseline' in self.results:
            summary += f"基线(仅CE): {self.results['baseline']['best_accuracy']:.4f}\n"

        if 'mr_consistency' in self.results:
            summary += f"MR一致性: {self.results['mr_consistency']['best_accuracy']:.4f}\n"

        if 'full_multi_signal' in self.results:
            summary += f"完整多信号: {self.results['full_multi_signal']['best_accuracy']:.4f}\n"

        if 'simple_composite_mr' in self.results:
            summary += f"简化复合MR: {self.results['simple_composite_mr']['best_accuracy']:.4f}\n"

        if 'augment_training' in self.results:
            summary += f"增强数据训练: {self.results['augment_training']['best_accuracy']:.4f}\n"
            summary += f"  - 最终alpha: {self.results['augment_training']['final_alpha']:.3f}\n"
            summary += f"  - 最终lam_inv: {self.results['augment_training']['final_lam_inv']:.3f}\n"
            summary += f"  - 最终lam_flip: {self.results['augment_training']['final_lam_flip']:.3f}\n"

        if 'ablation' in self.results:
            summary += "\n消融实验结果:\n"
            for name, acc in self.results['ablation'].items():
                summary += f"  {name}: {acc:.4f}\n"

        logger.info(summary)

        with open(os.path.join(self.exp_dir, 'summary.txt'), 'w') as f:
            f.write(summary)

    def run_simple_composite_mr_experiment(self):
        """简化版复合MR实验：CE + MR一致性 + 可选对比学习"""
        logger.info("=" * 50)
        logger.info("开始简化版复合MR实验：CE + MR一致性 + 对比学习")
        logger.info("=" * 50)

        model, optimizer = self.create_model_and_optimizer()
        scheduler = self.create_scheduler(optimizer,
                                          self.config['full_epochs'],
                                          self.train_loader)

        best_acc = 0.0
        train_losses = []
        val_accs = []

        # 导入简化版训练函数
        from NLI_train_one_epoch import train_one_epoch_simple_composite_mr

        # 训练配置
        training_config = {
            'alpha': 1.0,  # MR一致性权重
            'beta': 0.3,  # 对比学习权重
            'lambda_neu': 0.05,  # 中性类正则权重
            'lambda_reg': 1e-4,  # L2正则权重
            'use_supcon': True,  # 启用监督对比学习
            'use_neutral_reg': True  # 启用中性类正则化
        }

        logger.info(f"训练配置: {training_config}")

        for epoch in range(self.config['full_epochs']):
            # 可以根据epoch动态调整参数
            if epoch < self.config['full_epochs'] // 3:
                # 前期：专注MR一致性
                current_alpha = 1.0
                current_beta = 0.1
            elif epoch < 2 * self.config['full_epochs'] // 3:
                # 中期：逐步加入对比学习
                current_alpha = 1.0
                current_beta = 0.3
            else:
                # 后期：平衡所有损失
                current_alpha = 0.8
                current_beta = 0.5

            train_loss = train_one_epoch_simple_composite_mr(
                model_instance=model,
                dataloader=self.train_loader,
                optimizer=optimizer,
                device=self.device,
                epoch_idx=epoch,
                phase_name="简化复合MR训练",
                alpha=current_alpha,
                beta=current_beta,
                lambda_neu=training_config['lambda_neu'],
                lambda_reg=training_config['lambda_reg'],
                use_supcon=training_config['use_supcon'],
                use_neutral_reg=training_config['use_neutral_reg'],
                scheduler=scheduler)
            train_losses.append(train_loss)

            # 验证
            val_results = test_nli(model, self.NLImodel.tokenizer,
                                   self.val_dataset)
            val_acc = val_results['accuracy']
            val_accs.append(val_acc)

            logger.info(
                f"Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.4f}")
            logger.info(
                f"当前权重 - MR一致性:{current_alpha:.2f}, 对比学习:{current_beta:.2f}")

            if val_acc > best_acc:
                best_acc = val_acc
                if self.config.get('save_best_model', True):
                    model_save_path = self._get_model_save_path(
                        'simple_composite_mr_best')
                    self.NLImodel.save_model(model_save_path)
                    logger.info(f"保存最佳简化复合MR模型到: {model_save_path}")
                    logger.info(f"保存最佳简化复合MR模型到: {model_save_path}")

            # 早停机制（可选）
            if epoch > 20 and len(val_accs) > 5:
                recent_accs = val_accs[-5:]
                if max(recent_accs) - min(
                        recent_accs) < 0.001:  # 连续5个epoch准确率变化很小
                    logger.info("验证准确率趋于稳定，启用早停机制")
                    break

        self.results['simple_composite_mr'] = {
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accs,
            'training_config': training_config
        }

        logger.info(f"简化版复合MR实验完成，最佳验证准确率: {best_acc:.4f}")

        # 保存训练曲线数据
        training_data = {
            'epochs': list(range(len(train_losses))),
            'train_losses': train_losses,
            'val_accuracies': val_accs,
            'best_accuracy': best_acc,
            'config': training_config
        }

        with open(
                os.path.join(self.exp_dir,
                             'simple_composite_mr_training.json'), 'w') as f:
            json.dump(training_data, f, indent=2)

        return model, best_acc

    def train_with_augment(self,
                           model_instance,
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
        - tuple: (更新后的模型实例, 最佳验证准确率, 最终alpha值, 最终标签保持权重, 最终标签翻转权重)
        """

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_model_state = None  # To save the best model state

        # 记录训练过程数据
        train_losses = []
        val_losses = []
        val_accs = []
        alpha_history = []
        lam_inv_history = []
        lam_flip_history = []

        data_type = "augmented"  # unlabeled  augmented

        if data_type not in loaders or loaders[data_type] is None or len(
                loaders[data_type]) == 0:
            logger.info("======== 未提供增强数据集，跳过双重一致性训练阶段 ========")
            return model_instance, 0.0, alpha_init, lam_inv_init, lam_flip_init  # Return initial values if skipped

        # Create ckpt_base directory if it doesn't exist
        ckpt_base_dir = "./ckpt_base"
        if not os.path.exists(ckpt_base_dir):
            os.makedirs(ckpt_base_dir)
            logger.info(f"已创建目录: {ckpt_base_dir}")

        scheduler = self.create_scheduler(optimizer, epochs,
                                          loaders[data_type])

        for epoch in range(epochs):
            # alpha 从 0.5 到 0.8 线性调度，+是增加，-是减少
            current_alpha = 0.6 - 0.4 * (epoch / max(1, epochs - 1))
            # 余弦调度, + 是增加
            current_lam_flip = current_lam_inv = 0.01 + 0.75 * (
                epoch / max(1, epochs - 1))

            # 记录参数历史
            alpha_history.append(current_alpha)
            lam_inv_history.append(current_lam_inv)
            lam_flip_history.append(current_lam_flip)

            logger.info(
                f"======== 增强数据集训练阶段 Epoch {epoch + 1}/{epochs} ========")
            logger.info(
                f"当前参数: alpha={current_alpha:.3f}, lam_inv={current_lam_inv:.3f}, lam_flip={current_lam_flip:.3f}"
            )

            # 训练一个epoch
            from NLI_train_one_epoch import train_one_epoch_RDrop
            epoch_train_loss = train_one_epoch_RDrop(
                model_instance=model_instance,
                dataloader=loaders[data_type],
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch,
                phase_name="RDrop增强数据集训练阶段",
                scheduler=scheduler,
                lam=2.0,
            )

            # epoch_train_loss = train_one_epoch_labeled_CE(
            #     model_instance,
            #     loaders[data_type],
            #     optimizer,
            #     device,
            #     epoch,
            #     phase_name="训练CE",
            # )

            # 记录训练损失（如果train_one_epoch_RDrop返回了损失值）
            if epoch_train_loss is not None:
                train_losses.append(epoch_train_loss)
                logger.info(f"Epoch {epoch + 1} 训练损失: {epoch_train_loss:.4f}")
            else:
                train_losses.append(0.0)  # 占位符

            logger.info(
                f"======== 增强数据集训练阶段 结束 Epoch {epoch + 1}/{epochs} ========")

            # 验证阶段
            logger.info(
                f"======== 增强数据集验证阶段 Epoch {epoch + 1}/{epochs} ========")
            if "validation" not in loaders:
                logger.info("======== 未提供验证集，跳过增强数据集验证阶段 ========")
                # 即使没有验证集，也要记录占位符数据
                val_losses.append(float("inf"))
                val_accs.append(0.0)
            else:
                val_results = test_nli(model_instance, self.NLImodel.tokenizer,
                                       self.val_dataset)

                # 获取验证损失和准确率
                avg_val_loss = val_results.get('loss', float("inf"))
                val_acc = val_results.get('accuracy', 0.0)

                val_losses.append(avg_val_loss)
                val_accs.append(val_acc)

                logger.info(
                    f"Epoch {epoch + 1}: 验证损失={avg_val_loss:.4f}, 验证准确率={val_acc:.4f}"
                )

                # 检查验证集准确率是否改善
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = avg_val_loss
                    logger.info(
                        f"增强数据集验证阶段: 验证准确率改善，新最佳准确率: {best_val_acc:.4f}")
                    if self.config.get('save_best_model', True):
                        try:
                            # 保存最佳模型
                            model_save_path = self._get_model_save_path(
                                'augment_training_best')
                            self.NLImodel.save_model(model_save_path)
                            logger.info(
                                f"增强数据集验证阶段: 保存当前最优模型状态到 {model_save_path} (准确率: {best_val_acc:.4f})"
                            )
                        except Exception as e:
                            logger.error(f"保存模型状态失败: {e}")
                    else:
                        best_model_state = model_instance.state_dict()
                        logger.info(
                            f"增强数据集验证阶段: 最佳准确率更新，但未保存模型 (save_best_model=False)"
                        )

            logger.info(
                f"======== 增强数据集验证阶段 结束 Epoch {epoch + 1}/{epochs} ========")

            # 早停检查（可选）
            if epoch > 10 and len(val_accs) > 5:
                recent_accs = val_accs[-5:]
                if max(recent_accs) - min(recent_accs) < 0.001:
                    logger.info("验证准确率趋于稳定，考虑提前停止")

        # After all epochs, load the best model state if available
        if best_model_state is not None:
            logger.info(
                f"增强数据集训练结束，加载验证集上表现最优的模型状态 (准确率: {best_val_acc:.4f})。")
            model_instance.load_state_dict(best_model_state)
        else:
            logger.warning("增强数据集训练结束，但没有找到最佳模型状态进行保存。")

        # 保存训练记录到results中
        self.results['augment_training'] = {
            'best_accuracy': best_val_acc,
            'best_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'alpha_history': alpha_history,
            'lam_inv_history': lam_inv_history,
            'lam_flip_history': lam_flip_history,
            'final_alpha': current_alpha,
            'final_lam_inv': current_lam_inv,
            'final_lam_flip': current_lam_flip
        }

        # 保存详细训练曲线数据
        training_data = {
            'epochs': list(range(len(train_losses))),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'alpha_history': alpha_history,
            'lam_inv_history': lam_inv_history,
            'lam_flip_history': lam_flip_history,
            'best_accuracy': best_val_acc,
            'best_loss': best_val_loss,
            'config': {
                'alpha_init': alpha_init,
                'lam_inv_init': lam_inv_init,
                'lam_flip_init': lam_flip_init,
                'epochs': epochs
            }
        }

        with open(os.path.join(self.exp_dir, 'augment_training_curves.json'),
                  'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"增强数据集训练完成，最佳验证准确率: {best_val_acc:.4f}")

        return model_instance, best_val_acc, current_alpha, current_lam_inv, current_lam_flip

    def run_full_experiment(self):
        """运行完整的实验流程"""
        logger.info("开始多信号联合损失完整实验")

        # 加载数据
        self.load_data(shuffle=self.config['shuffle'])

        # 1. 基线实验
        # baseline_model, baseline_acc = self.run_baseline_experiment()

        # # 2. MR一致性实验
        # mr_model, mr_acc = self.run_mr_consistency_experiment()

        # 3. 完整多信号实验
        # full_model, full_acc = self.run_full_multi_signal_experiment()

        # # 4. 消融实验
        # self.run_ablation_study()

        # 5. 简化版复合MR实验
        # simple_model, simple_acc = self.run_simple_composite_mr_experiment()

        model, optimizer = self.create_model_and_optimizer()
        # from NLI_train_stage import train_with_metamorphic_relations
        # trained_model, best_loss, best_acc, final_lam_flip = train_with_metamorphic_relations(
        #     model_instance=model,
        #     tokenizer=self.NLImodel.tokenizer,
        #     augmented_dataset=self.train_dataset,
        #     optimizer=optimizer,
        #     val_loader=self.val_loader,
        #     device=self.device,
        #     epochs=self.config['full_epochs'],
        #     patience=20,
        # )
        trained_model, best_acc, final_alpha, final_lam_inv, final_lam_flip = self.train_with_augment(
            model_instance=model,
            loaders={
                'augmented': self.train_loader,
                'validation': self.val_loader
            },  # 修正数据加载器格式
            optimizer=optimizer,
            device=self.device,
            epochs=self.config['full_epochs'],
            patience=10,  # 添加patience参数
            alpha_init=0.5,
            lam_inv_init=1.0,
            lam_flip_init=0.5)

        logger.info(f"增强数据训练完成，最佳验证准确率: {best_acc:.4f}")

        # 保存最终模型
        if self.config.get('save_best_model', True):

            final_model_path = self._get_model_save_path('final_model')

            logger.info(f"保存最终模型到: {final_model_path}")
            self.NLImodel.save_model(final_model_path)
        else:
            logger.info("跳过最终模型保存 (save_best_model=False)")

        # 5. 生成报告
        self.generate_report()

        logger.info("实验完成！")
        return self.results

    def _load_json_dataset(self, data_path):
        """从JSON或JSONL文件加载数据集"""
        import json
        from datasets import Dataset

        logger.info(f"从文件加载数据：{data_path}")

        data = []
        try:
            # 先尝试作为标准JSON格式加载
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("文件格式：标准JSON")
        except json.JSONDecodeError:
            # 如果失败，尝试作为JSONL格式加载
            logger.info("尝试JSONL格式加载...")
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"第{line_num}行JSON解析失败: {e}")
                                continue
                logger.info(f"文件格式：JSONL，成功加载 {len(data)}条记录")
            except Exception as e:
                logger.error(f"JSONL格式加载也失败: {e}")
                raise

        if not data:
            raise ValueError(f"没有从文件 {data_path} 中加载到任何数据")

        logger.info(f"数据加载完成，共 {len(data)} 条记录")

        # 检查数据格式
        if len(data) > 0:
            sample = data[0]
            logger.info(f"数据样本字段: {list(sample.keys())}")

            # 检查必需字段
            required_fields = ['premise', 'hypothesis', 'label']
            missing_fields = [f for f in required_fields if f not in sample]
            if missing_fields:
                logger.warning(f"数据缺少必需字段: {missing_fields}")

        # 转换为HuggingFace Dataset格式
        return Dataset.from_list(data)

    def _collate_fn(self, batch):
        """数据批处理函数"""
        # 提取各个字段
        premises = [item.get("premise", "") for item in batch]
        hypotheses = [item.get("hypothesis", "") for item in batch]
        labels = [item.get("label", 0) for item in batch]
        pair_ids = [item.get("pair_id", 0) for item in batch]
        is_syns = [item.get("is_syn", False) for item in batch]
        mr_types = [item.get("mr_type", "inv") for item in batch]
        weights = [item.get("weight", 1.0) for item in batch]

        # 使用临时tokenizer进行编码
        if not hasattr(self, '_tokenizer'):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'])

        # 编码文本
        encodings = self._tokenizer(premises,
                                    hypotheses,
                                    truncation=True,
                                    padding=True,
                                    max_length=self.config.get(
                                        'max_length', 128),
                                    return_tensors="pt")

        # 组装batch
        batch_data = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'pair_id': pair_ids,
            'is_syn': torch.tensor(is_syns, dtype=torch.bool),
            'mr_type': mr_types,
        }

        # 如果有token_type_ids，也包含进去
        if 'token_type_ids' in encodings:
            batch_data['token_type_ids'] = encodings['token_type_ids']

        return batch_data


if __name__ == "__main__":
    """主函数：配置和运行实验"""

    # 实验配置
    config = {
        # 数据配置 - 可以选择使用预处理好的JSON文件或NLIDataset接口
        'use_augmented_data':
        False,  # 如果True，则使用NLIDataset接口；如果False，使用指定的JSON文件

        # 方式1：使用JSON文件路径（推荐用于你的已有数据）
        'train_data_path':
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data_三合一filter0.88.json',
        'val_data_path':
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/test.json',

        # 方式2：使用NLIDataset参数（如果use_augmented_data=True）
        'dataset_name': 'snli',
        'labeled_ratio': 0.0037,
        'unlabeled_ratio': 0.2,
        'augmented_dir':
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/',
        'llm_model_name': 'phi-4',

        # 模型配置
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'dropout_rate': 0.3,

        # 训练配置
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 1e-2,
        'warmup_ratio': 0.06,

        # 设备配置
        'device': 1,  # 0: cuda:0, 1: cuda:1, -1: cuda (auto), others: cpu

        # 模型保存配置
        'save_best_model': True,  # 是否保存最佳模型（True: 保存, False: 不保存）
        'save_model_dir': None,  # 模型保存目录，如果为None则使用默认实验目录
        # 示例: '/home/ubuntu/MTrain/MTrainNLI/saved_models'

        # 随机种子配置
        'random_seed': 42,  # 随机种子值
        'shuffle': False,  # 是否在加载数据时打乱顺序
        'deterministic': True,  # 是否启用确定性模式（可能影响性能但确保可重现性）

        # 实验轮数
        'baseline_epochs': 0,
        'mr_epochs': 0,
        'full_epochs': 20,
        'ablation_epochs': 0,
    }

    train_datasets = [
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data.json',
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data_三合一_0.8.json',
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4/augmented_data.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlimm_lr0.0051_phi-4/augmented_data.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/rte_lr0.8_phi-4/augmented_data.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/sick_lr1.0_phi-4/augmented_data.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0136_llama-3.2-3b-instruct/augmented_data copy.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4/augmented_data_三合一.json"  # 效果更好
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4/augmented_data_三合一_0.8.json'
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/rte_lr1.0_phi-4/labeled_data.json'
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/sick_lr1.0_phi-4/labeled_data.json'
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_llama-3.2-3b-instruct/augmented_data.json'
    ]
    test_dataset = [
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4/test.json',
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/test.json',
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/rte_lr1.0_phi-4/test.json'
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/sick_lr1.0_phi-4/test.json'
        # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0013_phi-4/test.json',
        # # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0013_phi-4/test.json',
        # # '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0013_phi-4/test.json',
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlimm_lr0.0013_phi-4/test.json",
        # # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlimm_lr0.0013_phi-4/test.json",
        # # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlimm_lr0.0013_phi-4/test.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/rte_lr0.4_phi-4/test.json",
        # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/sick_lr1.0_phi-4/test.json",
    ]

    # config['val_data_path'] = test_dataset[0]
    cnt = 0
    for train_dataset in train_datasets:
        config['train_data_path'] = train_dataset
        config['val_data_path'] = test_dataset[cnt]
        # config['random_seed'] = cnt
        cnt += 1

        # 运行实验
        print(f"正在运行实验，使用数据集: {config['train_data_path']}")
        experiment = MultiSignalExperiment(config)
        results = experiment.run_full_experiment()

# 使用示例：
#
# 1. 禁用模型保存（节省磁盘空间）:
# config['save_best_model'] = False
#
# 2. 指定自定义保存目录:
# config['save_best_model'] = True
# config['save_model_dir'] = '/home/ubuntu/MTrain/MTrainNLI/my_saved_models'
#
# 3. 默认行为（保存到实验目录）:
# config['save_best_model'] = True  # 默认值
# config['save_model_dir'] = None   # 默认值
