#!/usr/bin/env python3
"""
测试 new_test.py 对 RTE 数据集的支持
"""

import json
import tempfile
import os
from new_test import NLIModelTester, create_test_config


def create_sample_rte_data():
    """创建示例 RTE 数据"""
    rte_samples = [
        {
            "premise": "The cat is on the mat.",
            "hypothesis": "There is a cat.",
            "label": 0
        },
        {
            "premise": "It is raining.",
            "hypothesis": "The weather is sunny.",
            "label": 1
        },
        {
            "premise": "John likes apples.",
            "hypothesis": "John enjoys fruit.",
            "label": 0
        },
        {
            "premise": "The book is red.",
            "hypothesis": "The book is blue.",
            "label": 1
        },
        {
            "premise": "She is reading.",
            "hypothesis": "Someone is reading.",
            "label": 0
        },
    ]
    return rte_samples


def create_sample_nli_data():
    """创建示例三分类 NLI 数据"""
    nli_samples = [
        {
            "premise": "The cat is on the mat.",
            "hypothesis": "There is a cat.",
            "label": 0
        },
        {
            "premise": "It is raining.",
            "hypothesis": "The weather is okay.",
            "label": 1
        },
        {
            "premise": "John likes apples.",
            "hypothesis": "John hates fruit.",
            "label": 2
        },
        {
            "premise": "The book is red.",
            "hypothesis": "The book is blue.",
            "label": 2
        },
        {
            "premise": "She is reading.",
            "hypothesis": "Someone is reading.",
            "label": 0
        },
    ]
    return nli_samples


def test_metric_calculation():
    """测试指标计算功能"""
    print("测试指标计算功能...")

    config = create_test_config()
    config['save_results'] = False  # 不保存测试结果

    tester = NLIModelTester(config)

    # 测试 RTE 数据（二分类）
    print("\n=== 测试 RTE 数据（二分类）===")
    rte_labels = [0, 1, 0, 1, 0, 1, 0, 0]
    rte_predictions = [0, 1, 1, 1, 0, 0, 0, 1]

    rte_metrics = tester._calculate_metrics(rte_labels, rte_predictions)

    print(f"数据集类型: {rte_metrics['dataset_info']['num_classes']}分类")
    print(f"类别名称: {rte_metrics['dataset_info']['class_names']}")
    print(f"准确率: {rte_metrics['overall']['accuracy']:.4f}")
    print(f"Macro F1: {rte_metrics['overall']['f1_score']:.4f}")

    # 测试标准 NLI 数据（三分类）
    print("\n=== 测试标准 NLI 数据（三分类）===")
    nli_labels = [0, 1, 2, 0, 1, 2, 0, 1]
    nli_predictions = [0, 1, 1, 2, 1, 2, 0, 0]

    nli_metrics = tester._calculate_metrics(nli_labels, nli_predictions)

    print(f"数据集类型: {nli_metrics['dataset_info']['num_classes']}分类")
    print(f"类别名称: {nli_metrics['dataset_info']['class_names']}")
    print(f"准确率: {nli_metrics['overall']['accuracy']:.4f}")
    print(f"Macro F1: {nli_metrics['overall']['f1_score']:.4f}")


def test_data_loading():
    """测试数据加载功能"""
    print("\n\n测试数据加载功能...")

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False) as f:
        rte_data = create_sample_rte_data()
        for item in rte_data:
            f.write(json.dumps(item) + '\n')
        rte_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False) as f:
        nli_data = create_sample_nli_data()
        for item in nli_data:
            f.write(json.dumps(item) + '\n')
        nli_file = f.name

    try:
        config = create_test_config()
        config['save_results'] = False

        tester = NLIModelTester(config)

        # 测试 RTE 数据加载
        print(f"\n加载 RTE 数据: {rte_file}")
        rte_dataset = tester.load_test_data(rte_file)
        print(f"RTE 数据条数: {len(rte_dataset)}")
        print(f"RTE 示例: {rte_dataset[0]}")

        # 测试 NLI 数据加载
        print(f"\n加载 NLI 数据: {nli_file}")
        nli_dataset = tester.load_test_data(nli_file)
        print(f"NLI 数据条数: {len(nli_dataset)}")
        print(f"NLI 示例: {nli_dataset[0]}")

    finally:
        # 清理临时文件
        os.unlink(rte_file)
        os.unlink(nli_file)


if __name__ == "__main__":
    print("开始测试 new_test.py 的 RTE 支持...")

    test_metric_calculation()
    test_data_loading()

    print("\n\n测试完成！")
    print("修改后的代码现在支持：")
    print("1. 自动检测数据集类型（二分类 RTE 或三分类 NLI）")
    print("2. 适应性的类别名称和混淆矩阵显示")
    print("3. 在测试报告中包含数据集类型信息")
    print("4. 对缺失类别的零除错误保护")
