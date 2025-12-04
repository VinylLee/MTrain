# main.py
"""
主入口文件
---------
使用argparse获取命令行参数，执行训练和测试流程。
根据需求，可在该脚本中灵活地引入数据增强、模型选择、数据集选择等逻辑。
"""

import argparse
import os
import signal
import sys
import gc
import torch

from NLIdataset import NLIDataset
from NLImodel import NLIModel
from NLItest import test_nli
from NLItrain import train_nli, train_nli_progressive, train_nli_unlabeled_labeled
from mylogging import logger


def cleanup_gpu():
    """
    清理GPU内存
    """
    logger.info("已被注释掉的GPU清理函数")
    # try:
    #     if torch.cuda.is_available():
    #         logger.info("正在清理GPU内存...")
    #         # 清空CUDA缓存
    #         torch.cuda.empty_cache()
    #         # 收集CUDA IPC资源
    #         torch.cuda.ipc_collect()
    #         # 同步所有CUDA设备
    #         torch.cuda.synchronize()
    #         logger.info("GPU内存清理完成")
    #     # 强制垃圾回收
    #     gc.collect()
    # except Exception as e:
    #     logger.warning(f"GPU清理过程中出现错误: {e}")


def signal_handler(sig, frame):
    """
    信号处理函数，用于优雅退出
    """
    # 判断是否是键盘退出
    if sig == signal.SIGINT:
        logger.info("接收到键盘中断信号 (Ctrl+C)，正在清理资源...")
        cleanup_gpu()
        logger.info('资源清理完成，程序退出')
        sys.exit(0)
    else:
        logger.info(f"接收到信号 {sig}，不做处理，程序继续运行")


def main():
    """
    main函数
    ---------
    中文文档注释：
    解析命令行参数，选择数据集、模型，设定训练和测试超参数，并调用相应的函数进行训练和测试。
    """

    # 注册信号处理器，用于优雅退出
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    # 初始清理GPU内存
    # cleanup_gpu()

    parser = argparse.ArgumentParser(
        description="NLI Training and Testing Pipeline")
    # 数据集和模型相关
    parser.add_argument("--dataset_name",
                        type=str,
                        default="snli",
                        help="数据集名称，如snli或mnli等")
    parser.add_argument("--model_name",
                        type=str,
                        default="bert-base-uncased",
                        help="预训练模型名称")

    # 训练相关超参数
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-5,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="batch大小")

    # 数据划分相关参数
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--labeled_ratio",
                        type=float,
                        default=0.005,
                        help="有标签训练集比例")
    parser.add_argument("--unlabeled_ratio",
                        type=float,
                        default=0.095,
                        help="无标签训练集比例")

    # 是否使用原始验证集
    parser.add_argument("--used_original_val",
                        action="store_true",
                        help="是否使用原始验证集（默认使用）（已废弃）")
    parser.add_argument("--no_used_original_val",
                        dest="used_original_val",
                        action="store_false",
                        help="不使用原始验证集（使用训练集划分）（已废弃）")
    parser.set_defaults(used_original_val=True)

    # 是否使用数据增强
    parser.add_argument("--use_data_augmentation",
                        action="store_true",
                        help="是否使用数据增强（默认不使用）")
    parser.add_argument("--no_use_data_augmentation",
                        dest="use_data_augmentation",
                        action="store_false",
                        help="不使用数据增强（使用原始数据集）")
    parser.set_defaults(use_data_augmentation=False)

    # 是否只测试
    parser.add_argument("--only_test",
                        action="store_true",
                        help="是否只测试模型（默认训练和测试）")
    parser.add_argument("--no_only_test",
                        dest="only_test",
                        action="store_false",
                        help="训练和测试模型（默认）")
    parser.set_defaults(only_test=False)

    # LLM模型名称
    from LMstudio import MODEL
    parser.add_argument("--llm_model_name",
                        type=str,
                        default=MODEL,
                        help="LLM模型名称")

    # 模型保存路径
    parser.add_argument("--model_save_path",
                        type=str,
                        default=None,
                        help="模型保存路径")

    # 模型加载路径
    parser.add_argument("--model_load_path",
                        type=str,
                        default=None,
                        help="模型加载路径")

    # 是否只使用未标注数据
    parser.add_argument("--only_unlabeled",
                        action="store_true",
                        help="是否只使用未标注数据（默认使用有标签和无标签数据）")
    parser.add_argument("--no_only_unlabeled",
                        dest="only_unlabeled",
                        action="store_false",
                        help="使用有标签和无标签数据（默认）")
    parser.set_defaults(only_unlabeled=False)

    # 是否使用渐进式训练
    parser.add_argument("--progressive_training",
                        action="store_true",
                        help="是否使用渐进式训练（默认不使用）")
    parser.add_argument("--no_progressive_training",
                        dest="progressive_training",
                        action="store_false",
                        help="不使用渐进式训练（默认）")
    parser.set_defaults(progressive_training=False)

    # 是否使用先无标签后有标签的训练方法
    parser.add_argument("--unlabeled_first_training",
                        action="store_true",
                        help="是否使用先无标签后有标签的训练方法（默认不使用）")
    parser.add_argument("--no_unlabeled_first_training",
                        dest="unlabeled_first_training",
                        action="store_false",
                        help="不使用先无标签后有标签的训练方法（默认）")
    parser.set_defaults(unlabeled_first_training=False)

    # GPU设备选择
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="指定使用的GPU设备（例如 'cuda:0', 'cuda:1', 'cpu', 或 'auto' 自动选择）")
    parser.add_argument("--gpu",
                        type=int,
                        default=None,
                        help="指定使用的GPU编号（例如 0 表示 cuda:0, 1 表示 cuda:1）")

    args = parser.parse_args()

    # 处理GPU设备选择
    import torch
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu >= 0:
            args.device = f"cuda:{args.gpu}"
        else:
            logger.warning(f"指定的GPU {args.gpu} 不可用，回退到自动选择")
            args.device = "auto"

    # 清理GPU内存，确保干净的开始状态
    cleanup_gpu()

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device.startswith("cuda"):
        if torch.cuda.is_available():
            # 验证指定的GPU是否存在
            try:
                gpu_id = int(
                    args.device.split(":")[1]) if ":" in args.device else 0
                if gpu_id < torch.cuda.device_count():
                    device = args.device
                else:
                    logger.warning(f"指定的GPU {args.device} 不存在，回退到默认CUDA设备")
                    device = "cuda"
            except (ValueError, IndexError):
                logger.warning(f"无效的设备格式 {args.device}，回退到默认CUDA设备")
                device = "cuda"
        else:
            logger.warning("CUDA不可用，回退到CPU")
            device = "cpu"
    else:
        device = args.device

    # 模型保存路径
    if args.model_save_path is None:
        args.model_save_path = f"nli_saved_model/{args.dataset_name}/{args.model_name}"

    # 模型加载路径
    if args.only_test:
        if args.model_load_path is None:
            raise ValueError("如果只测试模型，则必须提供模型加载路径 --model_load_path")
        args.model_load_path = os.path.join(args.model_save_path,
                                            args.model_load_path)
        if not os.path.exists(args.model_load_path):
            raise ValueError(f"模型加载路径不存在: {args.model_load_path}")

    logger.info(f"数据集名称: {args.dataset_name}")
    logger.info(f"验证集比例: {args.val_ratio}")
    logger.info(f"有标签训练集比例: {args.labeled_ratio}")
    logger.info(f"无标签训练集比例: {args.unlabeled_ratio}")
    logger.info(f"是否使用原始验证集: {args.used_original_val}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"batch大小: {args.batch_size}")
    logger.info(f"模型保存路径: {args.model_save_path}")
    logger.info(f"模型加载路径: {args.model_load_path}")
    logger.info(f"是否只测试模型: {args.only_test}")
    logger.info(f"是否使用数据增强: {args.use_data_augmentation}")
    logger.info(f"LLM模型名称: {args.llm_model_name}")
    logger.info(f"是否只使用未标注数据: {args.only_unlabeled}")
    logger.info(f"是否使用渐进式训练: {args.progressive_training}")
    logger.info(f"是否使用先无标签后有标签的训练方法: {args.unlabeled_first_training}")
    logger.info(f"使用设备: {device}")

    # 设置PyTorch的当前设备（如果是CUDA设备）
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")

            # 提取GPU ID并设置为当前设备
            if ":" in device:
                gpu_id = int(device.split(":")[1])
            else:
                gpu_id = 0

            # 设置当前CUDA设备
            torch.cuda.set_device(gpu_id)
            logger.info(f"已设置CUDA当前设备为: {gpu_id}")
            logger.info(f"PyTorch当前设备: {torch.cuda.current_device()}")
            logger.info(f"设备名称: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(
                f"设备内存总量: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB"
            )
        else:
            logger.warning("CUDA不可用，但指定了CUDA设备")
    elif device == "cpu":
        logger.info("使用CPU设备进行计算")

    # ----------------------------------------------------------------------------------

    # 1. 加载数据集
    logger.info("加载数据集")
    nli_dataset = NLIDataset(dataset_name=args.dataset_name,
                             val_ratio=args.val_ratio,
                             labeled_ratio=args.labeled_ratio,
                             unlabeled_ratio=args.unlabeled_ratio,
                             use_original_val=args.used_original_val,
                             use_data_augmentation=args.use_data_augmentation,
                             llm_model_name=args.llm_model_name)
    labeled_dataset = nli_dataset.get_labeled_dataset()
    augmented_dataset = nli_dataset.augmented_dataset
    unlabeled_dataset = nli_dataset.get_unlabeled_dataset()
    val_dataset = nli_dataset.get_validation_dataset()
    test_dataset = nli_dataset.get_test_dataset()

    try:
        if not args.only_test:  # 训练和测试
            # 2. 初始化模型
            logger.info("初始化模型")
            nli_model = NLIModel(model_name=args.model_name, device=device)

            # 2.5. 测试模型
            logger.info("测试模型 -- 在训练前测试模型性能")
            test_metric = test_nli(model=nli_model.model,
                                   tokenizer=nli_model.tokenizer,
                                   test_dataset=test_dataset,
                                   batch_size=args.batch_size)
            logger.info(f"初始模型测试结果: {test_metric}")

            # 3. 训练模型
            logger.info("训练模型")
            if args.unlabeled_first_training:
                # 使用先无标签后有标签的训练方法
                nli_model.model = train_nli_unlabeled_labeled(
                    model=nli_model,
                    labeled_dataset=labeled_dataset,
                    augmented_dataset=augmented_dataset,
                    unlabeled_dataset=unlabeled_dataset,
                    validation_dataset=val_dataset,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    ckpt_name=nli_dataset.get_subdir() + "_ul_first")
            elif args.progressive_training:
                # 使用渐进式训练方法
                nli_model.model = train_nli_progressive(
                    model=nli_model,
                    labeled_dataset=labeled_dataset,
                    augmented_dataset=augmented_dataset,
                    unlabeled_dataset=unlabeled_dataset,
                    validation_dataset=val_dataset,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                )
            else:
                # 使用标准训练方法
                nli_model.model = train_nli(
                    model=nli_model,
                    labeled_dataset=labeled_dataset,
                    augmented_dataset=augmented_dataset,
                    unlabeled_dataset=unlabeled_dataset,
                    validation_dataset=val_dataset,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    use_data_augmentation=args.use_data_augmentation,
                    ckpt_name=nli_dataset.get_subdir(),
                    only_unlabeled=args.only_unlabeled)

        else:  # 只测试
            # 2. 加载模型
            logger.info("加载模型")
            nli_model = NLIModel.load_model(args.model_load_path,
                                            device=device)

        # 4. 测试模型
        logger.info("测试模型")
        test_metric = test_nli(model=nli_model.model,
                               tokenizer=nli_model.tokenizer,
                               test_dataset=test_dataset,
                               batch_size=args.batch_size)

        # 当前时间
        import datetime
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        if not args.only_test:  # 训练和测试
            # 5. 保存模型
            logger.info("保存训练后的模型")
            model_save_name = f"{args.labeled_ratio}_{args.use_data_augmentation}_{args.llm_model_name}_{current_time}_{test_metric['accuracy']:.4f}_{test_metric['f1']:.4f}"
            nli_model.save_model(
                os.path.join(args.model_save_path, model_save_name))
            return test_metric

    except KeyboardInterrupt:
        logger.info("接收到键盘中断信号 (Ctrl+C)")
        cleanup_gpu()
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        cleanup_gpu()
        raise
    finally:
        # 确保在程序结束时清理GPU内存
        logger.info("程序执行完成，正在进行最终清理...")
        cleanup_gpu()


if __name__ == "__main__":
    logger.info("开始运行主程序")
    test_metric = main()
