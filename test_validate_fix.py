#!/usr/bin/env python3
"""
测试验证函数修复的脚本
"""

import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 模拟一个包含混合数据类型的数据集
class MockDataset(Dataset):

    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000,
                                       (32, )),  # tensor - 应该移动到device
            'attention_mask': torch.ones(32),  # tensor - 应该移动到device
            'labels':
            torch.randint(0, 3, (1, )).squeeze().long(),  # tensor - 确保是long类型
            'text_pairs': ['premise text',
                           'hypothesis text'],  # list - 不应该移动到device
            'pair_id': f'pair_{idx}',  # string - 不应该移动到device
            'mr_type': 'inv'  # string - 不应该移动到device
        }


# 模拟一个简单的模型
class MockModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 3)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # 忽略其他非tensor参数
        logits = self.linear(input_ids.float()).mean(dim=1)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # 返回类似transformers模型的输出
        class ModelOutput:

            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

        return ModelOutput(logits, loss)


def test_validate_function():
    print("🧪 测试验证函数修复...")

    try:
        # 导入修复后的验证函数
        from NLI_train_stage import validate
        print("✅ 成功导入validate函数")

        # 创建测试数据
        dataset = MockDataset(20)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 创建模型
        model = MockModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"📱 使用设备: {device}")

        # 测试验证函数
        print("🔍 测试验证函数...")
        val_loss, val_acc = validate(device=device,
                                     model_instance=model,
                                     val_loader=dataloader,
                                     prefix="测试")

        print(f"✅ 验证函数运行成功!")
        print(f"📊 验证损失: {val_loss:.4f}")
        print(f"📊 验证准确率: {val_acc:.4f}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("🛠️  测试验证函数修复")
    print("=" * 50)

    success = test_validate_function()

    print("=" * 50)
    if success:
        print("🎉 所有测试通过！验证函数修复成功！")
    else:
        print("💥 测试失败，需要进一步检查")
    print("=" * 50)
