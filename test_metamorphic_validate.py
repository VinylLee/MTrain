#!/usr/bin/env python3
"""
测试验证函数修复 - 模拟真实的蜕变关系数据格式
"""

import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MetamorphicDataset(Dataset):
    """模拟蜕变关系数据集格式"""

    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟蜕变关系数据的完整格式
        return {
            'input_ids': torch.randint(0, 1000, (32, )),  # tensor - 模型需要
            'attention_mask': torch.ones(32),  # tensor - 模型需要
            'labels': torch.randint(0, 3,
                                    (1, )).squeeze().long(),  # tensor - 模型需要
            'pair_id': f'pair_{idx // 2}',  # string - 模型不需要
            'is_syn': torch.tensor(idx % 2 == 0,
                                   dtype=torch.bool),  # tensor - 模型不需要
            'mr_type': ['inv', 'flip', 'neutral'][idx % 3],  # string - 模型不需要
            'mr_id': f'mr_{idx}',  # string - 模型不需要
        }


class MockNLIModel(torch.nn.Module):
    """模拟NLI模型，只接受标准transformers参数"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 3)

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                token_type_ids=None,
                **kwargs):
        """
        模拟transformers模型的forward方法
        只接受标准参数，拒绝额外的关键字参数
        """
        # 检查是否有不支持的参数
        supported_params = {
            'input_ids', 'attention_mask', 'labels', 'token_type_ids'
        }
        unsupported = set(kwargs.keys()) - supported_params
        if unsupported:
            raise TypeError(
                f"forward() got an unexpected keyword argument '{list(unsupported)[0]}'"
            )

        # 简单的前向计算
        logits = self.linear(input_ids.float()).mean(dim=1)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            # 确保labels是正确的类型
            labels = labels.long()
            loss = loss_fn(logits, labels)  # 返回类似transformers模型的输出

        class ModelOutput:

            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

        return ModelOutput(logits, loss)


def test_validate_with_metamorphic_data():
    """测试验证函数是否能正确处理蜕变关系数据"""
    print("🧪 测试验证函数处理蜕变关系数据...")

    try:
        # 导入修复后的验证函数
        from NLI_train_stage import validate
        print("✅ 成功导入validate函数")

        # 创建测试数据
        dataset = MetamorphicDataset(20)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 创建模型
        model = MockNLIModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"📱 使用设备: {device}")

        # 测试一个batch的数据格式
        sample_batch = next(iter(dataloader))
        print(f"📊 batch包含的键: {list(sample_batch.keys())}")
        for k, v in sample_batch.items():
            if hasattr(v, 'shape'):
                print(f"  - {k}: {type(v).__name__} {v.shape}")
            else:
                print(
                    f"  - {k}: {type(v).__name__} (例: {v[0] if isinstance(v, list) else v})"
                )

        # 测试验证函数
        print("🔍 运行验证函数...")
        val_loss, val_acc = validate(device=device,
                                     model_instance=model,
                                     val_loader=dataloader,
                                     prefix="蜕变数据测试")

        print(f"✅ 验证函数运行成功!")
        print(f"📊 验证损失: {val_loss:.4f}")
        print(f"📊 验证准确率: {val_acc:.4f}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward_directly():
    """直接测试模型forward方法，确认参数过滤的必要性"""
    print("🧪 测试模型forward方法参数敏感性...")

    try:
        model = MockNLIModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 创建测试数据
        batch_with_extra_params = {
            'input_ids': torch.randint(0, 1000, (2, 32)).to(device),
            'attention_mask': torch.ones(2, 32).to(device),
            'labels': torch.randint(0, 3, (2, )).long().to(device),
            'pair_id': ['pair_1', 'pair_2'],  # 额外参数
            'is_syn': torch.tensor([True, False]).to(device),  # 额外参数
            'mr_type': ['inv', 'flip'],  # 额外参数
        }

        print("🔍 测试带有额外参数的forward调用...")
        try:
            # 这应该失败
            outputs = model(**batch_with_extra_params)
            print("❌ 意外成功：模型接受了额外参数（这不应该发生）")
            return False
        except TypeError as e:
            print(f"✅ 预期的错误：{e}")

        # 测试只传递模型需要的参数
        print("🔍 测试只传递标准参数的forward调用...")
        filtered_batch = {
            'input_ids': batch_with_extra_params['input_ids'],
            'attention_mask': batch_with_extra_params['attention_mask'],
            'labels': batch_with_extra_params['labels'],
        }

        outputs = model(**filtered_batch)
        print("✅ 成功：过滤后的参数正常工作")
        print(
            f"📊 输出形状: logits={outputs.logits.shape}, loss={outputs.loss.item():.4f}"
        )

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("🛠️  验证函数蜕变关系数据处理测试")
    print("=" * 70)

    # 测试1: 直接测试模型参数敏感性
    print("\n1️⃣ 测试模型forward方法参数处理")
    print("-" * 50)
    direct_test_success = test_model_forward_directly()

    print("\n2️⃣ 测试修复后的验证函数")
    print("-" * 50)
    # 测试2: 完整的验证函数测试
    validate_test_success = test_validate_with_metamorphic_data()

    print("\n" + "=" * 70)
    if direct_test_success and validate_test_success:
        print("🎉 所有测试通过！验证函数现在可以正确处理蜕变关系数据！")
        print("👍 验证函数会自动过滤掉模型不需要的额外参数")
        print("📋 支持的参数: input_ids, attention_mask, labels, token_type_ids")
        print("🚫 过滤的参数: pair_id, is_syn, mr_type, mr_id 等")
    else:
        print("💥 部分测试失败，需要进一步检查")
    print("=" * 70)
