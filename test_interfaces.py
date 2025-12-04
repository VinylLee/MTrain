#!/usr/bin/env python3
"""
接口验证脚本 - 用于验证模块接口的正确性
"""

import os
import sys
import json
import torch
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_nli_dataset_interface():
    """测试NLIDataset接口"""
    print("测试NLIDataset接口...")

    try:
        from NLIdataset import NLIDataset

        # 测试基本初始化
        dataset = NLIDataset(
            dataset_name="snli",
            labeled_ratio=0.001,  # 使用很小的比例进行快速测试
            use_data_augmentation=False,
            seed=42)

        # 测试数据获取方法
        labeled_data = dataset.get_labeled_dataset()
        test_data = dataset.get_test_dataset()

        print(f"✓ NLIDataset初始化成功")
        print(f"✓ 有标签数据集大小: {len(labeled_data)}")
        print(f"✓ 测试数据集大小: {len(test_data)}")

        # 检查数据格式
        if len(labeled_data) > 0:
            sample = labeled_data[0]
            print(f"✓ 数据样本字段: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"✗ NLIDataset测试失败: {e}")
        return False


def test_nli_model_interface():
    """测试NLIModel接口"""
    print("\n测试NLIModel接口...")

    try:
        from NLImodel import NLIModel

        # 测试模型初始化
        model = NLIModel(
            model_name='bert-base-uncased',
            num_labels=3,
            device='cpu'  # 使用CPU避免CUDA问题
        )

        print(f"✓ NLIModel初始化成功")
        print(f"✓ 模型设备: {model.device}")
        print(f"✓ 分词器类型: {type(model.tokenizer)}")

        # 测试简单前向传播
        test_input = model.tokenizer("A man is walking",
                                     "A person is moving",
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True)

        with torch.no_grad():
            outputs = model.model(**test_input)
            print(f"✓ 前向传播成功，输出形状: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"✗ NLIModel测试失败: {e}")
        return False


def test_nli_test_interface():
    """测试NLItest接口"""
    print("\n测试NLItest接口...")

    try:
        from NLItest import test_nli
        from NLImodel import NLIModel
        from datasets import Dataset

        # 创建简单的测试数据
        test_data = [{
            "premise": "A man is walking",
            "hypothesis": "A person is moving",
            "label": 0
        }, {
            "premise": "It is raining",
            "hypothesis": "The weather is sunny",
            "label": 2
        }, {
            "premise": "The sky is blue",
            "hypothesis": "The sky has a color",
            "label": 0
        }]

        test_dataset = Dataset.from_list(test_data)

        # 创建模型
        model = NLIModel(model_name='bert-base-uncased',
                         num_labels=3,
                         device='cpu')

        # 测试评估函数
        results = test_nli(model.model,
                           model.tokenizer,
                           test_dataset,
                           batch_size=2)

        print(f"✓ test_nli执行成功")
        print(f"✓ 返回结果类型: {type(results)}")
        print(f"✓ 结果字段: {list(results.keys())}")
        print(f"✓ 准确率: {results.get('accuracy', 'N/A')}")

        return True

    except Exception as e:
        print(f"✗ NLItest测试失败: {e}")
        return False


def test_data_file_loading():
    """测试数据文件加载"""
    print("\n测试数据文件加载...")

    # 测试路径
    test_paths = [
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data_三合一filter0.88.json',
        '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/test.json'
    ]

    def load_data_file(path):
        """加载JSON或JSONL文件"""
        data = []
        try:
            # 先尝试作为标准JSON格式加载
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, "JSON"
        except json.JSONDecodeError:
            # 如果失败，尝试作为JSONL格式加载
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"    第{line_num}行JSON解析失败: {e}")
                                continue
                return data, "JSONL"
            except Exception as e:
                raise Exception(f"JSON和JSONL格式都加载失败: {e}")

    for path in test_paths:
        try:
            if os.path.exists(path):
                data, file_format = load_data_file(path)

                print(f"✓ 文件存在并可读取: {path}")
                print(f"  文件格式: {file_format}")
                print(f"  数据条数: {len(data)}")

                if len(data) > 0:
                    sample = data[0]
                    print(f"  样本字段: {list(sample.keys())}")

                    # 检查必需字段
                    required_fields = ['premise', 'hypothesis', 'label']
                    missing_fields = [
                        f for f in required_fields if f not in sample
                    ]
                    if missing_fields:
                        print(f"  ⚠ 缺少必需字段: {missing_fields}")
                    else:
                        print(f"  ✓ 包含所有必需字段")

                    # 显示样本数据
                    print(f"  样本示例:")
                    print(
                        f"    premise: {sample.get('premise', 'N/A')[:50]}...")
                    print(
                        f"    hypothesis: {sample.get('hypothesis', 'N/A')[:50]}..."
                    )
                    print(f"    label: {sample.get('label', 'N/A')}")
                    if 'pair_id' in sample:
                        print(f"    pair_id: {sample.get('pair_id', 'N/A')}")
                    if 'is_syn' in sample:
                        print(f"    is_syn: {sample.get('is_syn', 'N/A')}")
                    if 'mr_type' in sample:
                        print(f"    mr_type: {sample.get('mr_type', 'N/A')}")
            else:
                print(f"✗ 文件不存在: {path}")

        except Exception as e:
            print(f"✗ 文件读取失败 {path}: {e}")


def test_multi_signal_experiment():
    """测试多信号实验脚本的导入"""
    print("\n测试多信号实验脚本...")

    try:
        from new_run import MultiSignalExperiment

        # 创建简单配置
        config = {
            'use_augmented_data': False,
            'model_name': 'bert-base-uncased',
            'max_length': 128,
            'batch_size': 2,
            'baseline_epochs': 1,
            'mr_epochs': 1,
            'full_epochs': 1,
            'ablation_epochs': 1,
        }

        # 测试初始化
        experiment = MultiSignalExperiment(config)
        print(f"✓ MultiSignalExperiment初始化成功")
        print(f"✓ 设备: {experiment.device}")
        print(f"✓ 实验目录: {experiment.exp_dir}")

        return True

    except Exception as e:
        print(f"✗ MultiSignalExperiment测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("接口验证测试")
    print("=" * 60)

    tests = [
        ("NLIDataset接口", test_nli_dataset_interface),
        ("NLIModel接口", test_nli_model_interface),
        ("NLItest接口", test_nli_test_interface),
        ("数据文件加载", test_data_file_loading),
        ("MultiSignalExperiment", test_multi_signal_experiment),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} 测试出现异常: {e}")
            results[test_name] = False

    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！可以开始运行实验了。")
    else:
        print("⚠️  有测试失败，请检查相关接口或依赖。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
