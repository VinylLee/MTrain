#!/usr/bin/env python3
"""
随机种子检测演示脚本
"""

import sys
import os

sys.path.append('.')

from seed_checker import (set_random_seed, print_seed_status,
                          detect_seed_status, generate_test_outputs,
                          check_reproducibility)


def main():
    print("🔍 随机种子检测演示")
    print("=" * 60)

    # 1. 检查初始状态
    print("1. 检查初始随机种子状态:")
    print_seed_status()

    # 2. 设置随机种子
    print("\n2. 设置随机种子 (seed=42):")
    set_random_seed(42, deterministic=True)
    print("✓ 随机种子已设置")

    # 3. 再次检查状态
    print("\n3. 设置后的状态:")
    print_seed_status()

    # 4. 测试可重现性
    print("\n4. 测试可重现性:")

    # 生成第一组随机数
    print("生成第一组随机数...")
    outputs1 = generate_test_outputs()
    for key, value in outputs1.items():
        print(f"  {key}: {value}")

    # 重新设置种子
    print("\n重新设置相同种子...")
    set_random_seed(42, deterministic=True)

    # 生成第二组随机数
    print("生成第二组随机数...")
    outputs2 = generate_test_outputs()
    for key, value in outputs2.items():
        print(f"  {key}: {value}")

    # 检查是否一致
    print("\n5. 一致性检查:")
    all_same = True
    for key in outputs1:
        if key in outputs2:
            is_same = abs(outputs1[key] - outputs2[key]) < 1e-10
            status = "✓ 一致" if is_same else "✗ 不一致"
            print(f"  {key}: {status}")
            if not is_same:
                all_same = False

    if all_same:
        print("\n🎉 所有随机数生成器都正确设置了种子！")
    else:
        print("\n⚠️  某些随机数生成器的种子设置可能有问题")

    # 6. 使用内置检查函数
    print("\n6. 使用内置可重现性检查:")
    reproducibility = check_reproducibility(num_tests=3)
    for component, is_reproducible in reproducibility.items():
        status = "✓ 可重现" if is_reproducible else "✗ 不可重现"
        print(f"  {component}: {status}")


if __name__ == "__main__":
    main()
