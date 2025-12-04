#!/usr/bin/env python3
"""
测试 train_nli_unlabeled_labeled 函数的简单脚本
"""

from NLIdataset import NLIDataset
from NLImodel import NLIModel
from NLItrain import train_nli_unlabeled_labeled
from mylogging import logger


def test_train_nli_unlabeled_labeled():
    """
    测试新的训练方法 train_nli_unlabeled_labeled
    """
    try:
        logger.info("========== 开始测试 train_nli_unlabeled_labeled ==========")

        # 初始化数据集 (使用较小的数据集以便快速测试)
        logger.info("初始化数据集...")
        dataset_handler = NLIDataset(
            dataset_name="snli",
            labeled_ratio=0.0001,  # 极小的有标签数据
            unlabeled_ratio=0.0005,  # 少量无标签数据
            use_original_val=True,
            use_data_augmentation=True,
            llm_model_name="llama-3.2-3b-instruct")

        labeled_dataset = dataset_handler.get_labeled_dataset()
        unlabeled_dataset = dataset_handler.get_unlabeled_dataset()
        validation_dataset = dataset_handler.get_validation_dataset()
        augmented_dataset = dataset_handler.get_augmented_dataset()

        logger.info(
            f"数据集大小 - 有标签: {len(labeled_dataset)}, 无标签: {len(unlabeled_dataset)}, 验证: {len(validation_dataset)}"
        )
        if augmented_dataset:
            logger.info(f"增强数据集大小: {len(augmented_dataset)}")

        # 初始化模型
        logger.info("初始化模型...")
        model = NLIModel(model_name="bert-base-uncased")

        # 使用新的训练方法进行训练
        logger.info("开始使用 train_nli_unlabeled_labeled 进行训练...")
        trained_model = train_nli_unlabeled_labeled(
            model=model,
            labeled_dataset=labeled_dataset,
            augmented_dataset=augmented_dataset,
            unlabeled_dataset=unlabeled_dataset,
            validation_dataset=validation_dataset,
            learning_rate=2e-5,
            batch_size=4,  # 使用较小的批次大小
            unlabeled_epochs=3,  # 减少轮数以便快速测试
            labeled_epochs=2,
            augmented_epochs=2,
            patience=3)

        logger.info("========== train_nli_unlabeled_labeled 测试完成 ==========")
        logger.info("新的训练方法工作正常！")
        return True

    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_train_nli_unlabeled_labeled()
    if success:
        logger.info("✅ 测试成功！")
    else:
        logger.error("❌ 测试失败！")
