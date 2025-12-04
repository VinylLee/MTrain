#!/usr/bin/env python3
"""
快速测试JSONL加载功能
"""

import json
import sys
from pathlib import Path


def test_jsonl_loading():
    """测试JSONL格式加载"""
    data_path = '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data_三合一filter0.88.json'

    print(f"测试文件: {data_path}")

    # 检查文件是否存在
    if not Path(data_path).exists():
        print("❌ 文件不存在!")
        return False

    # 尝试JSONL格式加载
    try:
        data = []
        line_count = 0
        error_count = 0

        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        data.append(item)
                        line_count += 1

                        # 显示前几个样本
                        if line_count <= 3:
                            print(f"\n样本 {line_count}:")
                            print(f"  字段: {list(item.keys())}")
                            print(
                                f"  premise: {item.get('premise', 'N/A')[:60]}..."
                            )
                            print(
                                f"  hypothesis: {item.get('hypothesis', 'N/A')[:60]}..."
                            )
                            print(f"  label: {item.get('label', 'N/A')}")
                            print(f"  pair_id: {item.get('pair_id', 'N/A')}")
                            print(f"  is_syn: {item.get('is_syn', 'N/A')}")
                            print(f"  mr_type: {item.get('mr_type', 'N/A')}")

                    except json.JSONDecodeError as e:
                        error_count += 1
                        if error_count <= 5:  # 只显示前5个错误
                            print(f"⚠️  第{line_num}行JSON解析失败: {e}")

        print(f"\n📊 加载统计:")
        print(f"  成功加载: {line_count} 条记录")
        print(f"  解析错误: {error_count} 条")
        print(f"  总计处理: {line_count + error_count} 条")

        if line_count > 0:
            print("✅ JSONL格式加载成功!")

            # 分析数据分布
            label_counts = {}
            mr_type_counts = {}
            pair_ids = set()

            for item in data:
                label = item.get('label', 'unknown')
                mr_type = item.get('mr_type', 'unknown')
                pair_id = item.get('pair_id')

                label_counts[label] = label_counts.get(label, 0) + 1
                mr_type_counts[mr_type] = mr_type_counts.get(mr_type, 0) + 1
                if pair_id is not None:
                    pair_ids.add(pair_id)

            print(f"\n📈 数据分布:")
            print(f"  标签分布: {label_counts}")
            print(f"  蜕变类型分布: {mr_type_counts}")
            print(f"  唯一pair_id数量: {len(pair_ids)}")

            return True
        else:
            print("❌ 没有成功加载任何数据!")
            return False

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


def test_with_multi_signal_loader():
    """测试使用multi_signal_experiment中的加载器"""
    print("\n" + "=" * 50)
    print("测试multi_signal_experiment的加载器")
    print("=" * 50)

    try:
        sys.path.insert(0, '/home/ubuntu/MTrain/MTrainNLI')
        from new_run import MultiSignalExperiment

        # 创建简单配置
        config = {
            'use_augmented_data': False,
            'model_name': 'bert-base-uncased',
            'max_length': 128,
            'batch_size': 4,
        }

        experiment = MultiSignalExperiment(config)

        # 测试数据加载
        data_path = '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/augmented_data_三合一filter0.88.json'
        dataset = experiment._load_json_dataset(data_path)

        print(f"✅ 使用实验脚本加载器成功!")
        print(f"  数据集大小: {len(dataset)}")
        print(f"  数据集类型: {type(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  样本字段: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"❌ 实验脚本加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("JSONL格式数据加载测试")
    print("=" * 50)

    # 测试1: 直接JSONL加载
    success1 = test_jsonl_loading()

    # 测试2: 使用实验脚本的加载器
    success2 = test_with_multi_signal_loader()

    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"直接JSONL加载: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"实验脚本加载器: {'✅ 成功' if success2 else '❌ 失败'}")

    if success1 and success2:
        print("\n🎉 所有测试通过! 可以开始运行完整实验了。")
        sys.exit(0)
    else:
        print("\n⚠️  有测试失败，请检查数据格式或文件路径。")
        sys.exit(1)
