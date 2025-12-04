#!/usr/bin/env python3
"""
测试修复后的设备设置是否正常工作
"""

import torch
import os
from new_run import MultiSignalExperiment


def test_device_consistency():
    """测试设备一致性"""
    print("=" * 50)
    print("测试设备一致性修复")
    print("=" * 50)

    # 检查可用的GPU
    if torch.cuda.is_available():
        print(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用，将使用CPU")

    # 测试配置
    config = {
        'model_name': 'microsoft/deberta-v3-base',
        'dataset_name': 'snli',
        'train_data_path':
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data纯增强.json',
        'labeled_ratio': 0.01,
        'unlabeled_ratio': 0.01,
        'use_augmented_data': False,

        # 损失函数权重
        'alpha': 0.5,
        'beta': 0.3,
        'gamma': 0.2,
        'mr_temperature': 1.0,
        'contrast_temperature': 0.1,
        'confidence_threshold': 0.8,
        'dropout_rate': 0.3,

        # 训练配置
        'batch_size': 8,  # 减小batch size以避免内存问题
        'learning_rate': 2e-5,
        'weight_decay': 1e-4,

        # 实验轮数 - 设置很小的值用于测试
        'baseline_epochs': 1,
        'mr_epochs': 0,
        'full_epochs': 1,
        'ablation_epochs': 0,

        # 设备配置
        'device': 1,  # 测试指定GPU 1
    }

    try:
        # 创建实验实例
        print(f"创建实验实例，指定设备: {config['device']}")
        experiment = MultiSignalExperiment(config)
        print(f"实验使用设备: {experiment.device}")

        # 加载数据
        print("加载数据...")
        experiment.load_data()
        print(f"训练数据加载完成，样本数: {len(experiment.train_dataset)}")

        # 创建模型
        print("创建模型...")
        model, optimizer = experiment.create_model_and_optimizer()

        # 检查模型设备
        model_device = next(model.parameters()).device
        print(f"模型位于设备: {model_device}")
        print(f"NLIModel内部设备: {experiment.NLImodel.device}")

        # 检查设备一致性
        if str(model_device) == str(experiment.device):
            print("✓ 设备一致性检查通过")
        else:
            print(f"✗ 设备不一致: 模型在{model_device}, 实验指定{experiment.device}")
            return False

        # 测试一个小的训练批次
        print("测试训练批次...")
        train_iter = iter(experiment.train_loader)
        batch = next(train_iter)

        # 检查批次数据设备
        print("移动批次数据到设备...")
        batch_device_before = {
            k: v.device if hasattr(v, 'device') else 'N/A'
            for k, v in batch.items() if hasattr(v, 'device')
        }
        print(f"批次数据设备 (移动前): {batch_device_before}")

        # 移动到设备
        batch_processed = {}
        for k, v in batch.items():
            if hasattr(v, 'to'):
                batch_processed[k] = v.to(experiment.device)
            else:
                batch_processed[k] = v

        batch_device_after = {
            k: v.device if hasattr(v, 'device') else 'N/A'
            for k, v in batch_processed.items() if hasattr(v, 'device')
        }
        print(f"批次数据设备 (移动后): {batch_device_after}")

        # 测试前向传播
        print("测试前向传播...")
        model.eval()
        with torch.no_grad():
            # 移除不需要的字段
            _ = batch_processed.pop("is_syn", None)
            _ = batch_processed.pop("pair_id", None)
            _ = batch_processed.pop("mr_type", None)

            outputs = model(**batch_processed)
            print(f"前向传播成功，输出logits形状: {outputs.logits.shape}")
            print(f"输出设备: {outputs.logits.device}")

        print("✓ 设备一致性修复测试通过！")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_device_consistency()
    exit(0 if success else 1)
