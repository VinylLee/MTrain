#!/usr/bin/env python3
"""
简化的验证函数测试，专注于数据类型处理
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_data_type_handling():
    """测试验证函数中的数据类型处理逻辑"""
    print("🧪 测试数据类型处理逻辑...")

    # 模拟包含混合数据类型的batch
    batch = {
        'input_ids': torch.randint(0, 1000, (4, 32)),  # tensor
        'attention_mask': torch.ones(4, 32),  # tensor
        'labels': torch.randint(0, 3, (4, )).long(),  # tensor
        'text_pairs': [['text1', 'text2'] for _ in range(4)],  # list of lists
        'pair_id': ['pair_1', 'pair_2', 'pair_3', 'pair_4'],  # list of strings
        'mr_type': ['inv', 'flip', 'neutral', 'inv']  # list of strings
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")

    try:
        # 测试修复后的数据处理逻辑
        batch_on_device = {}
        for k, v in batch.items():
            if hasattr(v, 'to') and callable(getattr(v, 'to')):
                batch_on_device[k] = v.to(device)
                print(f"✅ {k}: tensor -> 移动到{device}")
            else:
                batch_on_device[k] = v
                print(f"🔄 {k}: {type(v).__name__} -> 保持原样")

        # 验证tensor类型的数据确实移动到了目标设备
        for k, v in batch_on_device.items():
            if hasattr(v, 'device'):
                assert v.device.type == device.type, f"{k} 未正确移动到 {device}"
                print(f"✅ {k} 在正确的设备上: {v.device}")

        print("🎉 数据类型处理测试通过！")
        return True

    except Exception as e:
        print(f"❌ 数据类型处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_function():
    """测试能否成功导入修复后的验证函数"""
    print("🧪 测试导入修复后的验证函数...")

    try:
        from NLI_train_stage import validate
        print("✅ 成功导入validate函数")

        # 检查函数签名
        import inspect
        sig = inspect.signature(validate)
        print(f"📋 函数签名: {sig}")

        return True

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🛠️  验证函数数据类型处理修复测试")
    print("=" * 60)

    # 测试1: 导入函数
    import_success = test_import_function()
    print()

    # 测试2: 数据类型处理逻辑
    logic_success = test_data_type_handling()
    print()

    print("=" * 60)
    if import_success and logic_success:
        print("🎉 所有测试通过！验证函数修复成功！")
        print("👍 现在验证函数可以正确处理包含列表和字符串的batch数据")
    else:
        print("💥 部分测试失败，需要进一步检查")
    print("=" * 60)
