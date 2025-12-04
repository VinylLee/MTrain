#!/usr/bin/env python3
"""
检查当前项目中的随机种子设置
"""

import sys
import os

sys.path.append('.')


def check_current_seed_setup():
    """检查当前的随机种子设置"""
    print("🔍 检查当前项目的随机种子设置")
    print("=" * 60)

    # 导入检测工具
    try:
        from seed_checker import print_seed_status, detect_seed_status

        print("1. 当前随机种子状态:")
        print_seed_status()

        # 检测状态
        status = detect_seed_status()

        print(f"\n2. 详细信息:")
        print(f"  - PyTorch初始种子: {status['timestamp']}")
        print(f"  - Python哈希种子: {status['python_hashseed']}")
        print(f"  - CUDNN确定性: {status['cudnn_deterministic']}")
        print(f"  - CUDNN基准: {status['cudnn_benchmark']}")
        print(f"  - 可能已设置种子: {status['likely_seeded']}")

        return status['likely_seeded']

    except ImportError:
        print("⚠️  seed_checker模块未找到")

        # 简单检查
        import torch
        import numpy as np
        import random

        print("进行简单的随机种子检查...")

        # 检查环境变量
        pythonhashseed = os.environ.get('PYTHONHASHSEED', 'Not set')
        cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING',
                                              'Not set')

        print(f"PYTHONHASHSEED: {pythonhashseed}")
        print(f"CUDA_LAUNCH_BLOCKING: {cuda_launch_blocking}")
        print(f"PyTorch初始种子: {torch.initial_seed()}")

        # 检查CUDNN设置
        if hasattr(torch.backends.cudnn, 'deterministic'):
            print(f"CUDNN确定性: {torch.backends.cudnn.deterministic}")
        if hasattr(torch.backends.cudnn, 'benchmark'):
            print(f"CUDNN基准: {torch.backends.cudnn.benchmark}")

        return False


def test_with_experiment():
    """测试实验脚本的随机种子设置"""
    print("\n" + "=" * 60)
    print("🧪 测试实验脚本的随机种子设置")
    print("=" * 60)

    try:
        # 导入实验类
        from new_run import MultiSignalExperiment

        # 创建测试配置
        test_config = {
            'device': 'cpu',  # 使用CPU避免GPU相关问题
            'random_seed': 123,
            'deterministic': True,
            'model_name': 'bert-base-uncased',
            'batch_size': 4,
            'learning_rate': 2e-5,
            'weight_decay': 1e-4,
            'baseline_epochs': 0,
            'mr_epochs': 0,
            'full_epochs': 1,
            'ablation_epochs': 0,
        }

        print("创建实验实例...")
        experiment = MultiSignalExperiment(test_config)

        print("✓ 实验实例创建成功，随机种子应该已设置")

        # 再次检查种子状态
        print("\n实验初始化后的种子状态:")
        check_current_seed_setup()

    except Exception as e:
        print(f"⚠️  测试实验脚本时出错: {e}")
        import traceback
        traceback.print_exc()


def provide_recommendations():
    """提供随机种子设置建议"""
    print("\n" + "=" * 60)
    print("📋 随机种子设置建议")
    print("=" * 60)

    print("1. 确保可重现性的步骤:")
    print("   ✓ 设置Python random种子: random.seed(42)")
    print("   ✓ 设置NumPy种子: np.random.seed(42)")
    print("   ✓ 设置PyTorch种子: torch.manual_seed(42)")
    print("   ✓ 设置CUDA种子: torch.cuda.manual_seed_all(42)")
    print("   ✓ 设置环境变量: PYTHONHASHSEED=42")

    print("\n2. 确定性模式设置:")
    print("   ✓ torch.backends.cudnn.deterministic = True")
    print("   ✓ torch.backends.cudnn.benchmark = False")
    print("   ✓ os.environ['CUDA_LAUNCH_BLOCKING'] = '1'")

    print("\n3. 在你的代码中:")
    print("   - 在 config 中添加 'random_seed': 42")
    print("   - 在 config 中添加 'deterministic': True")
    print("   - 确保在实验开始前调用种子设置函数")

    print("\n4. 验证方法:")
    print("   - 运行相同实验多次，结果应该完全一致")
    print("   - 使用 seed_checker.py 中的工具进行验证")
    print("   - 检查日志中的种子设置确认信息")


def main():
    # 检查当前状态
    is_seeded = check_current_seed_setup()

    # 测试实验脚本
    test_with_experiment()

    # 提供建议
    provide_recommendations()

    print(f"\n总结: 随机种子{'已正确设置' if is_seeded else '需要设置'}")


if __name__ == "__main__":
    main()
