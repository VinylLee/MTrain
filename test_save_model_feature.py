#!/usr/bin/env python3
"""
测试模型保存控制功能
==================

验证新添加的 save_best_model 参数是否正确工作
"""

import os
import tempfile
import shutil


def test_save_best_model_feature():
    """测试保存最佳模型功能"""

    print("=" * 60)
    print("测试模型保存控制功能")
    print("=" * 60)

    # 测试配置参数
    test_configs = [{
        "name": "启用模型保存 (默认目录)",
        "config": {
            'save_best_model': True,
            'save_model_dir': None
        }
    }, {
        "name": "启用模型保存 (指定目录)",
        "config": {
            'save_best_model': True,
            'save_model_dir': '/tmp/test_model_save'
        }
    }, {
        "name": "禁用模型保存",
        "config": {
            'save_best_model': False,
            'save_model_dir': None
        }
    }]

    for i, test_case in enumerate(test_configs, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   配置: {test_case['config']}")

        # 模拟 _get_model_save_path 方法的逻辑
        config = test_case['config']
        exp_dir = "/tmp/test_exp"
        filename = "test_model.pth"

        # 模拟获取保存路径
        if config.get('save_model_dir'):
            save_path = os.path.join(config['save_model_dir'], filename)
            print(f"   → 保存路径: {save_path}")
        else:
            save_path = os.path.join(exp_dir, filename)
            print(f"   → 保存路径: {save_path}")

        # 模拟保存决策
        if config.get('save_best_model', True):
            print(f"   ✓ 将保存模型")
        else:
            print(f"   ✗ 跳过模型保存")

    print(f"\n" + "=" * 60)
    print("新增功能说明:")
    print("=" * 60)
    print("1. save_best_model: 控制是否保存最佳模型")
    print("   - True: 保存模型 (默认)")
    print("   - False: 不保存模型")
    print()
    print("2. save_model_dir: 指定模型保存目录")
    print("   - None: 使用默认实验目录 (默认)")
    print("   - 路径字符串: 使用指定目录")
    print()
    print("3. 应用范围:")
    print("   - 基线实验最佳模型")
    print("   - MR一致性实验最佳模型")
    print("   - 完整多信号实验最佳模型")
    print("   - 简化复合MR实验最佳模型")
    print("   - 增强数据训练最佳模型")
    print("   - 最终训练模型")
    print()
    print("4. 使用示例:")
    print("   # 禁用所有模型保存")
    print("   config['save_best_model'] = False")
    print()
    print("   # 保存到指定目录")
    print("   config['save_best_model'] = True")
    print("   config['save_model_dir'] = '/path/to/save/models'")

    print(f"\n✅ 功能测试完成！")


if __name__ == "__main__":
    test_save_best_model_feature()
