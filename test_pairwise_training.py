#!/usr/bin/env python3
"""
测试新增的蜕变关系训练函数
"""

import sys
import os

sys.path.append('/home/ubuntu/MTrain/MTrainNLI')


def test_pairwise_training():
    """测试新的训练函数是否能正常导入"""
    try:
        from NLI_train_one_epoch import (
            train_one_epoch_pairwise_metamorphic,
            train_one_epoch_pairwise_dataloader,
            create_pairwise_dataloader_from_dataset)
        print("✅ 成功导入新的蜕变关系训练函数")

        # 检查函数签名
        print("\n📋 函数说明:")
        print(
            "1. train_one_epoch_pairwise_metamorphic: 直接使用dataset，按pair_id分组处理"
        )
        print(
            "2. train_one_epoch_pairwise_dataloader: 使用预处理的pairwise dataloader"
        )
        print("3. create_pairwise_dataloader_from_dataset: 创建按pair分组的数据加载器")

        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def describe_features():
    """描述新功能的特点"""
    print("\n🚀 新增功能特点:")
    print("=" * 50)

    print("\n1. 🎯 按pair_id分组训练:")
    print("   - 不再按固定batch_size训练")
    print("   - 将相同pair_id的源数据和蜕变数据放在一起处理")
    print("   - 确保一致性损失能正确计算")

    print("\n2. 🔄 蜕变关系处理:")
    print("   - inv-MR (标签不变): 使用KL散度确保预测一致")
    print("   - flip-MR (标签翻转): entailment ↔ contradiction")
    print("   - neutral-MR (标签变中性): 增加neutral类预测概率")

    print("\n3. 💡 损失函数设计:")
    print("   - 源数据: 标准交叉熵损失")
    print("   - 蜕变数据: 一致性损失 + 交叉熵损失")
    print("   - 支持不同蜕变类型的权重调节")
    print("   - 包含L2正则化和标签平滑")

    print("\n4. 📊 数据格式要求:")
    print("   - mr_id: 'none'表示源数据，其他表示蜕变数据")
    print("   - mr_type: 'inv', 'flip', 'neutral'")
    print("   - is_syn: True表示合成/蜕变数据")
    print("   - pair_id: 同一组的源数据和蜕变数据共享")


def usage_examples():
    """使用示例"""
    print("\n📝 使用示例:")
    print("=" * 50)

    print("\n方法1 - 直接使用dataset:")
    print("""
from NLI_train_one_epoch import train_one_epoch_pairwise_metamorphic

avg_loss = train_one_epoch_pairwise_metamorphic(
    model_instance=model,
    dataset=augmented_dataset,  # 包含源数据和蜕变数据
    optimizer=optimizer,
    device=device,
    epoch_idx=epoch,
    alpha_inv=1.0,      # 标签不变蜕变权重
    alpha_flip=1.0,     # 标签翻转蜕变权重  
    alpha_neutral=0.8,  # 标签变中性蜕变权重
    beta_consistency=0.5,  # 一致性损失权重
    gamma_ce=1.0        # 交叉熵损失权重
)
""")

    print("\n方法2 - 使用pairwise dataloader:")
    print("""
from NLI_train_one_epoch import (
    train_one_epoch_pairwise_dataloader,
    create_pairwise_dataloader_from_dataset
)

# 创建特殊的数据加载器
pairwise_loader = create_pairwise_dataloader_from_dataset(
    dataset=augmented_dataset,
    tokenizer=model.tokenizer,
    max_pairs_per_batch=4
)

# 训练
avg_loss = train_one_epoch_pairwise_dataloader(
    model_instance=model,
    pairwise_dataloader=pairwise_loader,
    optimizer=optimizer,
    device=device,
    epoch_idx=epoch,
    # ... 其他参数同上
)
""")


if __name__ == "__main__":
    print("🧪 测试蜕变关系训练函数")
    print("=" * 50)

    success = test_pairwise_training()

    if success:
        describe_features()
        usage_examples()

        print("\n✨ 推荐配置:")
        print("- alpha_inv=1.0: 同义词替换等保持标签的蜕变")
        print("- alpha_flip=1.0: 添加矛盾等翻转标签的蜕变")
        print("- alpha_neutral=0.8: 添加无关信息等变中性的蜕变")
        print("- beta_consistency=0.5: 平衡一致性损失和CE损失")
        print("- temperature=1.0: KL散度温度参数")

        print("\n🎉 代码已成功添加到 NLI_train_one_epoch.py!")
    else:
        print("\n❌ 测试失败，请检查代码")
