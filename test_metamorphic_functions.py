#!/usr/bin/env python3
"""
验证新添加的蜕变关系训练函数
"""


def test_function_availability():
    """测试函数是否可以正常导入和使用"""

    print("🧪 测试蜕变关系训练函数")
    print("=" * 50)

    try:
        # 测试导入
        import sys
        sys.path.append('/home/ubuntu/MTrain/MTrainNLI')

        from NLI_train_stage import train_with_metamorphic_relations, train_with_pairwise_dataloader
        print("✅ 成功导入训练函数")

        # 检查函数签名
        import inspect

        print("\n📋 train_with_metamorphic_relations 函数参数:")
        sig1 = inspect.signature(train_with_metamorphic_relations)
        for param_name, param in sig1.parameters.items():
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            print(f"  - {param_name}{default}")

        print("\n📋 train_with_pairwise_dataloader 函数参数:")
        sig2 = inspect.signature(train_with_pairwise_dataloader)
        key_params = [
            'model_instance', 'augmented_dataset', 'val_loader', 'optimizer',
            'device', 'epochs'
        ]
        for param_name in key_params:
            if param_name in sig2.parameters:
                param = sig2.parameters[param_name]
                default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
                print(f"  - {param_name}{default}")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def show_usage_example():
    """展示使用示例"""

    print("\n📖 完整使用示例:")
    print("=" * 50)

    example_code = '''
# 在你的主训练代码中使用蜕变关系训练

from NLI_train_stage import train_with_metamorphic_relations

def main_training_pipeline():
    # ... 初始化模型、数据等 ...
    
    # 阶段1: 基础训练 (使用有标签数据)
    model, _, _ = train_with_labeled_data(
        model_instance=model,
        loaders={"labeled": labeled_loader, "validation": val_loader},
        optimizer=optimizer,
        device=device,
        epochs=5,
        patience=3,
        alpha_init=0.5,
        gamma=1.0
    )
    
    # 阶段2: 蜕变关系增强训练 (新添加的函数)
    model, best_loss, best_acc, final_weights = train_with_metamorphic_relations(
        model_instance=model,
        augmented_dataset=augmented_dataset,  # 包含蜕变数据的数据集
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=10,
        patience=5,
        
        # 蜕变关系权重配置
        alpha_inv=1.0,      # 标签不变 (同义词替换等)
        alpha_flip=1.0,     # 标签翻转 (添加矛盾等)
        alpha_neutral=0.8,  # 标签变中性 (添加无关信息等)
        beta_consistency=0.5,  # 一致性损失权重
        gamma_ce=1.0,       # 交叉熵损失权重
        
        # 高级配置
        temperature=1.0,
        enable_dynamic_adjustment=True,  # 启用动态权重调整
        save_best_model=True
    )
    
    print(f"蜕变增强训练完成!")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"最终权重配置: {final_weights}")
    
    # 阶段3: 后续训练或评估
    # ...
    
    return model

# 运行训练
if __name__ == "__main__":
    model = main_training_pipeline()
'''

    print(example_code)


def show_key_features():
    """展示关键特性"""

    print("\n🚀 关键特性:")
    print("=" * 50)

    features = [
        "🎯 按pair_id分组: 确保蜕变关系正确建立", "🔄 多种蜕变类型: 支持inv/flip/neutral三种蜕变关系",
        "⚖️ 智能权重调整: 根据训练进展动态调整损失权重", "🛑 早停机制: 防止过拟合，自动停止训练",
        "💾 模型保存: 自动保存验证集上表现最好的模型", "📊 详细日志: 提供训练过程的详细信息", "🔧 易于集成: 兼容现有的训练管道",
        "⚡ 两种模式: dataset直接处理 + pairwise dataloader"
    ]

    for feature in features:
        print(f"  {feature}")


def show_data_requirements():
    """展示数据格式要求"""

    print("\n📊 数据格式要求:")
    print("=" * 50)

    print("增强数据集中每个样本需要包含:")
    print("  - premise: 前提文本")
    print("  - hypothesis: 假设文本")
    print("  - label: NLI标签 (0=entailment, 1=neutral, 2=contradiction)")
    print("  - pair_id: 数据对ID (同组数据共享)")
    print("  - is_syn: 是否为蜕变数据 (boolean)")
    print("  - mr_id: 蜕变关系ID ('none'表示源数据)")
    print("  - mr_type: 蜕变类型 ('inv', 'flip', 'neutral')")

    print("\n示例数据:")
    print('''
{
    "premise": "A group of people riding a yellow roller coaster.",
    "hypothesis": "A group of people are riding a roller coaster.",
    "label": 0,
    "pair_id": 0,
    "is_syn": false,
    "mr_id": "none",
    "mr_type": "inv"
}
''')


if __name__ == "__main__":
    success = test_function_availability()

    if success:
        show_usage_example()
        show_key_features()
        show_data_requirements()

        print("\n🎉 新的蜕变关系训练函数已成功添加!")
        print("你现在可以在主训练流程中使用 train_with_metamorphic_relations() 函数")

    else:
        print("\n❌ 函数验证失败，请检查代码")
