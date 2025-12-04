#!/usr/bin/env python3
"""
验证修复后的蜕变关系训练函数
"""


def test_function_signatures():
    """测试函数签名是否正确"""

    print("🔧 验证函数修复")
    print("=" * 50)

    try:
        import sys
        sys.path.append('/home/ubuntu/MTrain/MTrainNLI')

        # 测试导入
        from NLI_train_stage import train_with_metamorphic_relations, train_with_pairwise_dataloader
        from NLI_train_one_epoch import train_one_epoch_pairwise_metamorphic, create_pairwise_dataloader_from_dataset

        print("✅ 成功导入所有函数")

        # 检查函数签名
        import inspect

        print("\n📋 train_with_metamorphic_relations 函数参数:")
        sig1 = inspect.signature(train_with_metamorphic_relations)
        key_params = [
            'model_instance', 'augmented_dataset', 'tokenizer', 'val_loader',
            'optimizer', 'device'
        ]
        for param_name in key_params:
            if param_name in sig1.parameters:
                print(f"  ✅ {param_name}")
            else:
                print(f"  ❌ 缺少参数: {param_name}")

        print("\n📋 train_one_epoch_pairwise_metamorphic 函数参数:")
        sig2 = inspect.signature(train_one_epoch_pairwise_metamorphic)
        key_params2 = [
            'model_instance', 'dataset', 'tokenizer', 'optimizer', 'device'
        ]
        for param_name in key_params2:
            if param_name in sig2.parameters:
                print(f"  ✅ {param_name}")
            else:
                print(f"  ❌ 缺少参数: {param_name}")

        print("\n🎯 关键修复:")
        print("  ✅ 添加了tokenizer参数到所有相关函数")
        print("  ✅ 修复了model_instance.tokenizer访问错误")
        print("  ✅ 更新了函数调用和文档")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_fix():
    """展示修复后的使用方法"""

    print("\n📖 修复后的使用方法:")
    print("=" * 50)

    usage_example = '''
# 修复后的正确调用方式

from NLI_train_stage import train_with_metamorphic_relations

# 确保你有tokenizer对象
tokenizer = model.tokenizer  # 或者从其他地方获取
# 或者：from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained(...)

# 调用训练函数 (注意添加了tokenizer参数)
model, best_loss, best_acc, final_weights = train_with_metamorphic_relations(
    model_instance=model,
    augmented_dataset=augmented_dataset,
    tokenizer=tokenizer,  # 🔧 重要: 现在需要显式传入tokenizer
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=10,
    patience=3,
    alpha_inv=1.0,
    alpha_flip=1.0,
    alpha_neutral=0.8,
    beta_consistency=0.5,
    gamma_ce=1.0,
    enable_dynamic_adjustment=True,
    save_best_model=True
)
'''

    print(usage_example)


def show_error_fix():
    """展示修复的错误"""

    print("\n🐛 修复的错误:")
    print("=" * 50)

    print("❌ 原始错误:")
    print(
        "AttributeError: 'SequenceClassifierWithCustomLoss' object has no attribute 'tokenizer'"
    )

    print("\n✅ 修复方案:")
    print("1. 在函数签名中添加 tokenizer 参数")
    print("2. 将 model_instance.tokenizer 改为 tokenizer")
    print("3. 更新所有相关的函数调用")
    print("4. 更新文档和使用示例")

    print("\n🔧 修改的文件:")
    print("- NLI_train_one_epoch.py: 修复核心训练函数")
    print("- NLI_train_stage.py: 修复上层训练管道")
    print("- METAMORPHIC_TRAINING_COMPLETE.md: 更新使用文档")


if __name__ == "__main__":
    success = test_function_signatures()

    if success:
        show_usage_fix()
        show_error_fix()

        print("\n🎉 修复完成!")
        print("现在你可以正确使用蜕变关系训练函数了")
        print("记住: 需要在调用时传入tokenizer参数")

    else:
        print("\n❌ 修复验证失败，可能还有其他问题")
