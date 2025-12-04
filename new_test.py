#!/usr/bin/env python3
"""
模型测试脚本 - 支持多模型、多数据集性能评估

功能特性：
1. 支持选择不同的预训练模型和已训练检查点
2. 支持多种数据集格式（JSON/JSONL文件、HuggingFace数据集）
3. 输出详细的性能指标（准确率、精确率、召回率、F1分数等）
4. 支持按类别分析和混淆矩阵
5. 可保存测试结果到文件

使用示例：
python new_test.py --model_path /path/to/model --test_data /path/to/test.json
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# 导入项目模块
from NLImodel import NLIModel
from NLIdataset import NLIDataset
from NLItest import test_nli
from mylogging import logger
import logging
from seed_checker import set_random_seed, detect_seed_status

logger.setLevel(logging.ERROR)  # 默认关闭 INFO 及 DEBUG 级别日志


class NLIModelTester:
    """NLI模型测试器 - 支持多种配置和详细的性能分析"""

    def __init__(self, config: Dict):
        """
        初始化测试器
        
        参数:
        - config: 配置字典，包含模型路径、数据路径、测试参数等
        """
        self.config = config
        self.device = self._setup_device(config.get('device', 'auto'))
        self.results = {}

        # 设置随机种子
        if 'random_seed' in config:
            logger.info(f"设置随机种子: {config['random_seed']}")
            set_random_seed(config['random_seed'])
            seed_status = detect_seed_status()
            if seed_status.get('likely_seeded', False):
                logger.info("随机种子设置成功")
            else:
                logger.warning("随机种子设置可能未生效")

        # 创建结果保存目录
        if config.get('save_results', True):
            self.result_dir = f"/home/ubuntu/MTrain/MTrainNLI/OODTestresult/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.result_dir, exist_ok=True)
            logger.info(f"测试结果将保存到: {self.result_dir}")

    def _setup_device(self, device_config) -> torch.device:
        """设置计算设备"""
        if device_config == 'auto':
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device_config, int):
            if device_config >= 0:
                device = torch.device(f'cuda:{device_config}' if torch.cuda.
                                      is_available() else 'cpu')
            else:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)

        # 检查设备可用性
        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU")
            device = torch.device('cpu')
        elif device.type == 'cuda' and device.index is not None:
            if device.index >= torch.cuda.device_count():
                logger.warning(f"CUDA设备{device.index}不存在，使用默认CUDA设备")
                device = torch.device('cuda')

        logger.info(f"使用设备: {device}")
        return device

    def load_model(self, model_path: str) -> NLIModel:
        """
        加载模型
        
        参数:
        - model_path: 模型路径，可以是检查点目录或预训练模型名
        
        返回:
        - NLIModel: 加载的模型实例
        """
        logger.info(f"加载模型: {model_path}")

        try:
            # 尝试从检查点加载
            if os.path.exists(model_path):
                model = NLIModel.load_model(load_dir=model_path)
                logger.info(f"从检查点加载模型成功: {model_path}")
            else:
                # 创建新模型
                model = NLIModel(model_name=model_path,
                                 device=str(self.device))
                logger.info(f"创建新模型成功: {model_path}")

            # 确保模型在正确的设备上
            model.model.to(self.device)
            logger.info(f"模型已移动到设备: {self.device}")

            return model

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def load_test_data(self, data_path: str) -> Dataset:
        """
        加载测试数据
        
        参数:
        - data_path: 数据路径，支持JSON/JSONL文件或数据集名称
        
        返回:
        - Dataset: 测试数据集
        """
        logger.info(f"加载测试数据: {data_path}")

        if os.path.exists(data_path):
            # 从文件加载
            return self._load_json_dataset(data_path)
        else:
            # 使用NLIDataset接口加载
            dataset_handler = NLIDataset(dataset_name=data_path,
                                         use_original_val=True)
            return dataset_handler.get_test_dataset()

    def _load_json_dataset(self, data_path: str) -> Dataset:
        """从JSON或JSONL文件加载数据集"""
        data = []

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                # JSONL格式
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

        logger.info(f"数据加载完成，共 {len(data)} 条记录")

        # 检查数据格式
        if len(data) > 0:
            sample = data[0]
            required_fields = ['premise', 'hypothesis', 'label']
            missing_fields = [
                field for field in required_fields if field not in sample
            ]
            if missing_fields:
                logger.warning(f"数据缺少字段: {missing_fields}")

        return Dataset.from_list(data)

    def _create_collate_fn(self, tokenizer):
        """创建批处理函数"""
        max_length = self.config.get('max_length', 128)

        def collate_fn(batch):
            premises = [item.get("premise", "") for item in batch]
            hypotheses = [item.get("hypothesis", "") for item in batch]
            labels = [item.get("label", 0) for item in batch]

            # 编码文本
            encodings = tokenizer(premises,
                                  hypotheses,
                                  truncation=True,
                                  padding=True,
                                  max_length=max_length,
                                  return_tensors="pt")

            encodings["labels"] = torch.tensor(labels, dtype=torch.long)
            return encodings

        return collate_fn

    def _determine_mapping_action(self, test_labels, model_predictions):
        """确定是否需要进行类别映射以及映射方向。"""

        label_set = set(test_labels)
        pred_set = set(model_predictions)

        test_is_binary = label_set <= {0, 1}
        test_is_ternary = label_set <= {0, 1, 2} and 2 in label_set
        preds_are_binary = pred_set <= {0, 1}
        preds_are_ternary = pred_set <= {0, 1, 2} and 2 in pred_set

        if test_is_binary and preds_are_ternary:
            return "pred_to_binary"
        if test_is_ternary and preds_are_binary:
            return "label_to_binary"
        return None

    def _map_values_to_binary(self, values):
        """将标签或预测值映射到二分类标签。"""

        mapped = []
        for val in values:
            if val == 0:
                mapped.append(0)
            elif val in [1, 2]:
                mapped.append(1)
            else:
                mapped.append(val)
        return mapped

    def evaluate_model(self, model: NLIModel, test_dataset: Dataset) -> Dict:
        """
        评估模型性能
        
        参数:
        - model: 待评估的模型
        - test_dataset: 测试数据集
        
        返回:
        - Dict: 包含各种性能指标的字典
        """
        logger.info("开始模型评估...")

        # 创建数据加载器
        collate_fn = self._create_collate_fn(model.tokenizer)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.config.get('batch_size', 16),
                                 shuffle=False,
                                 collate_fn=collate_fn)

        model.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        all_raw_predictions = []  # 保存原始预测结果

        with torch.no_grad():
            for batch in test_loader:
                labels = batch["labels"]

                # 移动数据到设备
                batch_device = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "labels" and isinstance(v, torch.Tensor)
                }

                # 前向传播
                outputs = model.model(**batch_device)
                logits = outputs.logits

                # 收集预测结果
                raw_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                all_raw_predictions.extend(raw_predictions)
                all_labels.extend(labels.numpy())
                all_logits.extend(logits.cpu().numpy())

        # 检测是否需要进行二分类/三分类映射
        unique_labels = sorted(set(all_labels))
        unique_raw_preds = sorted(set(all_raw_predictions))

        logger.info(f"测试数据标签类别: {unique_labels}")
        logger.info(f"模型原始预测类别: {unique_raw_preds}")

        mapping_action = self._determine_mapping_action(
            unique_labels, unique_raw_preds)

        labels_for_metrics = all_labels
        predictions_for_metrics = all_raw_predictions
        mapping_info = {'applied': False}

        if mapping_action == "pred_to_binary":
            logger.info("检测到模型为三分类预测但测试集为二分类标签，进行预测映射")
            logger.info(
                "映射规则: 0(Entailment)→0, 1(Neutral)→1, 2(Contradiction)→1")
            predictions_for_metrics = self._map_values_to_binary(
                all_raw_predictions)
            mapping_info = {
                'applied': True,
                'value_kind': 'predictions',
                'original_values': unique_raw_preds,
                'mapped_values': sorted(set(predictions_for_metrics)),
                'mapping_rule': "0→0, 1→1, 2→1"
            }
        elif mapping_action == "label_to_binary":
            logger.info("检测到模型为二分类预测但测试集为三分类标签，进行标签映射")
            logger.info(
                "映射规则: 0(Entailment)→0, 1(Neutral)→1, 2(Contradiction)→1")
            labels_for_metrics = self._map_values_to_binary(all_labels)
            mapping_info = {
                'applied': True,
                'value_kind': 'labels',
                'original_values': unique_labels,
                'mapped_values': sorted(set(labels_for_metrics)),
                'mapping_rule': "0→0, 1→1, 2→1"
            }
        else:
            logger.info("无需进行类别映射，直接使用原始标签和预测结果")

        # 计算各种指标
        metrics = self._calculate_metrics(labels_for_metrics,
                                          predictions_for_metrics, all_logits)

        metrics['mapping_info'] = mapping_info

        logger.info("模型评估完成")
        return metrics

    def _calculate_metrics(self,
                           labels: List[int],
                           predictions: List[int],
                           logits: Optional[List] = None) -> Dict:
        """计算详细的性能指标"""

        # 检测数据集类型
        unique_labels = sorted(set(labels + predictions))
        num_classes = len(unique_labels)

        logger.info(f"检测到 {num_classes} 个类别: {unique_labels}")

        # 根据类别数量确定类别名称
        if num_classes == 2:
            # RTE 类型数据集 (二分类)
            if set(unique_labels) == {0, 1}:
                class_names = ['Entailment', 'Non-Entailment']
                logger.info("检测到 二分类 类型数据集 (): 0=蕴含, 1=非蕴含")
            else:
                class_names = [f'Class_{i}' for i in unique_labels]
                logger.info(f"检测到二分类数据集，标签: {unique_labels}")
        elif num_classes == 3:
            # 标准 NLI 数据集 (三分类)
            class_names = ['Entailment', 'Neutral', 'Contradiction']
            logger.info("检测到标准 NLI 数据集 (三分类): 0=蕴含, 1=中立, 2=矛盾")
        else:
            # 其他情况
            class_names = [f'Class_{i}' for i in unique_labels]
            logger.info(f"检测到 {num_classes} 分类数据集，标签: {unique_labels}")

        # 基本指标
        accuracy = accuracy_score(labels, predictions)

        # Macro F1 (主要指标)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0)

        # Weighted F1 (作为补充)
        precision_weighted, recall_weighted, f1_weighted, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0)

        # 按类别指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0)

        # 分类报告
        class_report = classification_report(labels,
                                             predictions,
                                             target_names=class_names,
                                             output_dict=True,
                                             zero_division=0)

        # 混淆矩阵
        conf_matrix = confusion_matrix(labels, predictions)

        # 组装结果
        metrics = {
            'dataset_info': {
                'num_classes': int(num_classes),
                'class_names': class_names,
                'unique_labels': [int(x) for x in unique_labels]
            },
            'overall': {
                'accuracy': float(accuracy),
                # 主要指标 - Macro F1
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1_score': float(f1_macro),
                # 补充指标 - Weighted F1
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_score_weighted': float(f1_weighted),
                'num_samples': int(len(labels))
            },
            'per_class': {
                class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }

        # 如果有logits，计算置信度统计
        if logits is not None:
            logits_array = np.array(logits)
            probs = torch.softmax(torch.tensor(logits_array), dim=-1).numpy()
            max_probs = np.max(probs, axis=-1)

            metrics['confidence_stats'] = {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'min_confidence': float(np.min(max_probs)),
                'max_confidence': float(np.max(max_probs))
            }

        return metrics

    def print_results(self, metrics: Dict):
        """打印测试结果"""
        logger.info("=" * 60)
        logger.info("模型测试结果")
        logger.info("=" * 60)

        # 显示映射信息（如果有的话）
        if 'mapping_info' in metrics and metrics['mapping_info']['applied']:
            mapping_info = metrics['mapping_info']
            value_kind = mapping_info.get('value_kind', 'values')
            kind_display = '预测' if value_kind == 'predictions' else '标签'
            logger.info("🔄 三分类到二分类映射信息:")
            logger.info(f"  映射对象: {kind_display}")
            logger.info(f"  原始类别: {mapping_info.get('original_values', [])}")
            logger.info(f"  映射后类别: {mapping_info.get('mapped_values', [])}")
            logger.info(f"  映射规则: {mapping_info.get('mapping_rule', 'N/A')}")
            logger.info("-" * 60)

        # 总体指标
        overall = metrics['overall']
        logger.info(f"总体性能 (Macro F1):")
        logger.info(f"  准确率 (Accuracy): {overall['accuracy']:.4f}")
        logger.info(f"  精确率 (Precision Macro): {overall['precision']:.4f}")
        logger.info(f"  召回率 (Recall Macro): {overall['recall']:.4f}")
        logger.info(f"  F1分数 (F1 Macro): {overall['f1_score']:.4f}")
        logger.info(f"  样本数量: {overall['num_samples']}")

        # 显示 Weighted F1 作为补充
        if 'precision_weighted' in overall:
            logger.info(f"\n补充指标 (Weighted F1):")
            logger.info(
                f"  精确率 (Precision Weighted): {overall['precision_weighted']:.4f}"
            )
            logger.info(
                f"  召回率 (Recall Weighted): {overall['recall_weighted']:.4f}")
            logger.info(
                f"  F1分数 (F1 Weighted): {overall['f1_score_weighted']:.4f}")

        # 按类别指标
        logger.info(f"\n按类别性能:")
        for class_name, class_metrics in metrics['per_class'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    精确率: {class_metrics['precision']:.4f}")
            logger.info(f"    召回率: {class_metrics['recall']:.4f}")
            logger.info(f"    F1分数: {class_metrics['f1_score']:.4f}")

        # 置信度统计
        if 'confidence_stats' in metrics:
            conf_stats = metrics['confidence_stats']
            logger.info(f"\n置信度统计:")
            logger.info(f"  平均置信度: {conf_stats['mean_confidence']:.4f}")
            logger.info(f"  置信度标准差: {conf_stats['std_confidence']:.4f}")
            logger.info(f"  最小置信度: {conf_stats['min_confidence']:.4f}")
            logger.info(f"  最大置信度: {conf_stats['max_confidence']:.4f}")

        # 混淆矩阵
        logger.info(f"\n混淆矩阵:")
        conf_matrix = np.array(metrics['confusion_matrix'])

        # 根据类别数量确定简称
        num_classes = conf_matrix.shape[0]
        if num_classes == 2:
            class_short_names = ['Ent', 'Non-Ent']
        elif num_classes == 3:
            class_short_names = ['Ent', 'Neu', 'Con']
        else:
            class_short_names = [f'C{i}' for i in range(num_classes)]

        # 打印表头
        header = "真实\\预测 " + " ".join(f"{name:>8}"
                                      for name in class_short_names)
        logger.info(header)

        # 打印矩阵
        for i, (true_label,
                row) in enumerate(zip(class_short_names, conf_matrix)):
            row_str = f"{true_label:>8} " + " ".join(f"{val:>8}"
                                                     for val in row)
            logger.info(row_str)

    def save_results(self, metrics: Dict, model_info: Dict):
        """保存测试结果到文件"""
        if not self.config.get('save_results', True):
            return

        # 准备保存的数据
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'model_info': model_info,
            'metrics': metrics
        }

        # 保存JSON格式结果
        json_path = os.path.join(self.result_dir, 'test_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data,
                      f,
                      indent=2,
                      ensure_ascii=False,
                      cls=NumpyEncoder)

        # 保存简化的文本报告
        txt_path = os.path.join(self.result_dir, 'test_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("NLI模型测试报告\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"测试时间: {save_data['timestamp']}\n")
            f.write(f"模型路径: {model_info.get('model_path', 'Unknown')}\n")
            f.write(f"测试数据: {self.config.get('test_data_path', 'Unknown')}\n")

            # 添加数据集信息
            dataset_info = metrics.get('dataset_info', {})
            f.write(f"数据集类型: {dataset_info.get('num_classes', 'Unknown')}分类\n")
            f.write(
                f"类别名称: {', '.join(dataset_info.get('class_names', []))}\n")

            # 添加映射信息
            if 'mapping_info' in metrics and metrics['mapping_info']['applied']:
                mapping_info = metrics['mapping_info']
                kind_display = '预测值' if mapping_info.get(
                    'value_kind') == 'predictions' else '标签'
                f.write(f"预测映射: 已对{kind_display}应用三分类到二分类映射\n")
                f.write(f"原始类别: {mapping_info.get('original_values', [])}\n")
                f.write(f"映射后类别: {mapping_info.get('mapped_values', [])}\n")
                f.write(f"映射规则: {mapping_info.get('mapping_rule', 'N/A')}\n")
            else:
                f.write(f"预测映射: 无映射\n")
            f.write("\n")

            overall = metrics['overall']
            f.write(f"总体性能:\n")
            f.write(f"  准确率: {overall['accuracy']:.4f}\n")
            f.write(f"  精确率: {overall['precision']:.4f}\n")
            f.write(f"  召回率: {overall['recall']:.4f}\n")
            f.write(f"  F1分数: {overall['f1_score']:.4f}\n")
            f.write(f"  样本数: {overall['num_samples']}\n\n")

            f.write("按类别性能:\n")
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(
                    f"  {class_name}: P={class_metrics['precision']:.4f}, "
                    f"R={class_metrics['recall']:.4f}, F1={class_metrics['f1_score']:.4f}\n"
                )

        logger.info(f"测试结果已保存到: {self.result_dir}")

    def run_test(self, model_path: str, test_data_path: str) -> Dict:
        """
        运行完整的测试流程
        
        参数:
        - model_path: 模型路径
        - test_data_path: 测试数据路径
        
        返回:
        - Dict: 测试结果
        """
        try:
            # 加载模型
            model = self.load_model(model_path)

            # 加载测试数据
            test_dataset = self.load_test_data(test_data_path)

            # 评估模型
            metrics = self.evaluate_model(model, test_dataset)

            # 打印结果
            self.print_results(metrics)

            # 保存结果
            model_info = {
                'model_path': model_path,
                'model_name': getattr(model, 'model_name', 'Unknown'),
                'device': str(self.device)
            }
            self.save_results(metrics, model_info)

            return metrics

        except Exception as e:
            logger.error(f"测试过程中出现错误: {e}")
            raise


def create_test_config():
    """创建默认的测试配置"""
    return {
        # 模型配置
        'device': 'auto',  # 'auto', 0, 1, 'cpu', etc.
        'batch_size': 16,
        'max_length': 128,

        # 测试配置
        'save_results': True,
        'random_seed': 42,

        # 可选的高级配置
        'compute_confidence': True,
        'detailed_analysis': True
    }


def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='NLI模型测试脚本')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='模型路径（检查点目录或预训练模型名）')
    parser.add_argument('--test_data',
                        type=str,
                        required=True,
                        help='测试数据路径（JSON/JSONL文件或数据集名）')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--device',
                        type=str,
                        default='auto',
                        help='计算设备 (auto, 0, 1, cpu)')
    parser.add_argument('--max_length', type=int, default=128, help='输入序列最大长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_save', action='store_true', help='不保存测试结果')

    args = parser.parse_args()

    # 创建配置
    config = create_test_config()
    config.update({
        'batch_size': args.batch_size,
        'device': args.device,
        'max_length': args.max_length,
        'random_seed': args.seed,
        'save_results': not args.no_save,
        'test_data_path': args.test_data
    })

    # 运行测试
    tester = NLIModelTester(config)
    results = tester.run_test(args.model_path, args.test_data)

    return results


if __name__ == "__main__":
    # 可以直接运行main()使用命令行参数，或者自定义配置

    # 方式1: 使用命令行参数
    results = main()
    print("acc: " + str(results['overall']['accuracy']))

    # # 方式2: 直接配置参数（适合Jupyter等环境）
    # config = create_test_config()
    # config.update({
    #     'batch_size': 16,
    #     'device': 1,  # 使用GPU 0
    #     'random_seed': 22
    # })

    # # 测试配置示例
    # test_configs = [
    #     {
    #         'model_path':
    #         # '/home/ubuntu/MTrain/experiments/rte_lr1.0_phi-4_labeled_data.json_20251117_140739/final_model',
    #         # '/home/ubuntu/MTrain/MTrainNLI/experiments-old/rte_lr0.8_phi-4_20250725_142152/augment_training_best', # 7220
    #         # '/home/ubuntu/MTrain/experiments/sick_lr1.0_phi-4_labeled_data.json_20251117_140813/final_model',
    #         # '/home/ubuntu/MTrain/experiments-copy/sick_lr1.0_phi-4_augmented_data.json_20250728_134744/augment_training_best', # 8753
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_saved_model/snli/bert-base-uncased/0.0037_False_phi-4_2025-11-06 14:38:41_0.7520_0.7497',
    #         # '/home/ubuntu/MTrain/experiments/snli_lr0.0037_phi-4_augmented_data_三合一_0.8.json_20251105_153237/augment_training_best',
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_saved_model/mnlim/bert-base-uncased/0.0051_False_phi-4_2025-07-01 14:37:23_0.6239_0.6242',
    #         '/home/ubuntu/MTrain/experiments/mnlim_lr0.0051_phi-4_augmented_data_三合一.json_20251106_153720/augment_training_best',
    #         'test_data_path':
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/HANS/hans_test_formated.json'
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/snli-hard/snli_hard_formated.json'
    #         # "/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/dnli/dnli_test.json"
    #         # "/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/mnlimm/test.json"
    #         # "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/test.json"
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/stress_tests/antonym_mm.json'
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/stress_tests/numerical_reasoning.json'
    #         # '/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/stress_tests/word_overlap_mm.json'
    #         '/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/stress_tests/negation_m.json'
    #     },
    # ]

    # # 运行测试
    # for i, test_config in enumerate(test_configs):
    #     logger.info(f"开始第 {i+1} 个测试配置...")

    #     tester = NLIModelTester(config)
    #     try:
    #         results = tester.run_test(
    #             model_path=test_config['model_path'],
    #             test_data_path=test_config['test_data_path'])
    #         logger.info(f"测试配置 {i+1} 完成")
    #     except Exception as e:
    #         logger.error(f"测试配置 {i+1} 失败: {e}")
