#!/usr/bin/env python3
"""
蜕变关系增强训练的使用示例
展示如何在上层代码中调用新的训练函数
"""


def example_usage_metamorphic_training():
    """
    使用示例：如何在主训练流程中使用蜕变关系训练函数
    """

    print("🔧 蜕变关系增强训练使用示例")
    print("=" * 60)

    print("\n📋 主要函数:")
    print("1. train_with_metamorphic_relations() - 推荐使用")
    print("2. train_with_pairwise_dataloader() - 替代方案")

    print("\n💡 使用方法:")
    print("""
# 方法1: 使用推荐的蜕变关系训练函数
from NLI_train_stage import train_with_metamorphic_relations

# 在你的主训练流程中调用
model, best_loss, best_acc, final_weights = train_with_metamorphic_relations(
    model_instance=model,                # 你的NLI模型实例
    augmented_dataset=augmented_dataset, # 增强数据集 (包含mr_id, mr_type等字段)
    val_loader=val_loader,              # 验证数据加载器
    optimizer=optimizer,                 # 优化器
    device=device,                      # 计算设备
    epochs=10,                          # 训练轮数
    patience=3,                         # 早停容忍轮数
    
    # 蜕变关系权重配置
    alpha_inv=1.0,                      # 标签不变蜕变权重 (同义词替换等)
    alpha_flip=1.0,                     # 标签翻转蜕变权重 (添加矛盾等)  
    alpha_neutral=0.8,                  # 标签变中性蜕变权重 (添加无关信息等)
    beta_consistency=0.5,               # 一致性损失权重
    gamma_ce=1.0,                       # 交叉熵损失权重
    
    # 其他参数
    temperature=1.0,                    # KL散度温度参数
    label_smoothing=0.1,                # 标签平滑
    l2_reg_weight=1e-4,                 # L2正则化权重
    enable_dynamic_adjustment=True,      # 启用动态权重调整
    save_best_model=True                # 保存最佳模型
)

print(f"训练完成!")
print(f"最佳验证准确率: {best_acc:.4f}")
print(f"最佳验证损失: {best_loss:.4f}")
print(f"最终权重配置: {final_weights}")
""")

    print("\n📊 数据格式要求:")
    print("""
增强数据集中每个样本应包含以下字段:
- premise: 前提文本
- hypothesis: 假设文本  
- label: NLI标签 (0=entailment, 1=neutral, 2=contradiction)
- pair_id: 数据对ID (同一组的源数据和蜕变数据共享)
- is_syn: 是否为合成/蜕变数据 (bool)
- mr_id: 蜕变关系ID ('none'表示源数据)
- mr_type: 蜕变关系类型 ('inv', 'flip', 'neutral')

示例数据:
{
    "premise": "A group of people riding a yellow roller coaster.",
    "hypothesis": "A group of people are riding a roller coaster.",
    "label": 0,
    "pair_id": 0,
    "is_syn": false,
    "mr_id": "none", 
    "mr_type": "inv"
}
""")

    print("\n⚙️ 权重配置建议:")
    print("""
推荐配置:
- alpha_inv=1.0     : 标签保持不变的蜕变应该与源数据预测一致
- alpha_flip=1.0    : 标签翻转的蜕变应该翻转预测 (entailment ↔ contradiction)
- alpha_neutral=0.8 : 标签变中性的蜕变应该增加neutral预测
- beta_consistency=0.5 : 一致性损失与CE损失平衡
- temperature=1.0   : KL散度温度，控制软化程度

高级配置:
- enable_dynamic_adjustment=True : 启用动态权重调整，根据训练进展自动优化
- save_best_model=True : 自动保存验证集上表现最好的模型
- patience=3-5 : 早停容忍轮数，防止过拟合
""")

    print("\n🔄 训练流程集成:")
    print("""
在你的主训练管道中的建议位置:

1. 基础训练阶段 (有标签数据)
   ↓
2. 蜕变关系增强训练阶段 ← 在这里使用新函数
   ↓  
3. 精调阶段 (可选)
   ↓
4. 最终评估

或者替换现有的增强数据训练阶段:
- 将 train_with_augment() 替换为 train_with_metamorphic_relations()
- 获得更好的蜕变关系利用效果
""")

    print("\n🚀 优势特点:")
    print("""
✅ 按pair_id分组处理，确保蜕变关系正确建立
✅ 针对不同蜕变类型设计专门的损失函数
✅ 支持动态权重调整，自动优化训练过程  
✅ 内置早停机制，防止过拟合
✅ 详细的训练日志，便于调试和监控
✅ 自动保存最佳模型状态
✅ 兼容现有的训练管道
""")


if __name__ == "__main__":
    example_usage_metamorphic_training()
