#!/usr/bin/env python3
"""
测试GPU设备选择功能的脚本
"""

import torch
import sys
import os

# 添加当前路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NLImodel import NLIModel
from mylogging import logger


def test_device_selection():
    """测试不同设备选择选项"""

    logger.info("开始测试设备选择功能")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"可用的CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA设备 {i}: {torch.cuda.get_device_name(i)}")

    # 测试自动设备选择
    logger.info("\n=== 测试自动设备选择 ===")
    try:
        model_auto = NLIModel(model_name="bert-base-uncased")
        logger.info(f"自动选择的设备: {model_auto.device}")
        del model_auto
        logger.info("自动设备选择测试成功")
    except Exception as e:
        logger.error(f"自动设备选择测试失败: {e}")

    # 测试显式CPU设备选择
    logger.info("\n=== 测试CPU设备选择 ===")
    try:
        model_cpu = NLIModel(model_name="bert-base-uncased", device="cpu")
        logger.info(f"指定CPU设备: {model_cpu.device}")
        del model_cpu
        logger.info("CPU设备选择测试成功")
    except Exception as e:
        logger.error(f"CPU设备选择测试失败: {e}")

    # 测试CUDA设备选择（如果可用）
    if torch.cuda.is_available():
        logger.info("\n=== 测试CUDA设备选择 ===")
        try:
            model_cuda = NLIModel(model_name="bert-base-uncased",
                                  device="cuda")
            logger.info(f"指定CUDA设备: {model_cuda.device}")
            del model_cuda
            logger.info("CUDA设备选择测试成功")
        except Exception as e:
            logger.error(f"CUDA设备选择测试失败: {e}")

        # 测试特定GPU设备选择
        if torch.cuda.device_count() > 1:
            logger.info("\n=== 测试特定GPU设备选择 ===")
            try:
                model_cuda1 = NLIModel(model_name="bert-base-uncased",
                                       device="cuda:1")
                logger.info(f"指定CUDA:1设备: {model_cuda1.device}")
                del model_cuda1
                logger.info("特定GPU设备选择测试成功")
            except Exception as e:
                logger.error(f"特定GPU设备选择测试失败: {e}")
        else:
            logger.info("只有一个GPU设备，跳过多GPU测试")
    else:
        logger.info("CUDA不可用，跳过CUDA设备测试")

    logger.info("\n设备选择功能测试完成")


def test_command_line_args():
    """测试命令行参数解析逻辑"""
    logger.info("\n=== 测试命令行参数解析逻辑 ===")

    # 模拟不同的命令行参数组合
    test_cases = [
        {
            "device": "auto",
            "gpu": None
        },
        {
            "device": "cpu",
            "gpu": None
        },
        {
            "device": "cuda",
            "gpu": None
        },
        {
            "device": "cuda:0",
            "gpu": None
        },
        {
            "device": "cuda:1",
            "gpu": None
        },
        {
            "device": "auto",
            "gpu": 0
        },
        {
            "device": "auto",
            "gpu": 1
        },
        {
            "device": "invalid",
            "gpu": None
        },
    ]

    for i, args in enumerate(test_cases):
        logger.info(
            f"\n测试案例 {i+1}: device={args['device']}, gpu={args['gpu']}")

        # 模拟参数处理逻辑
        device_arg = args["device"]
        gpu_arg = args["gpu"]

        # 处理GPU设备选择（复制自NLImain.py的逻辑）
        if gpu_arg is not None:
            if torch.cuda.is_available() and gpu_arg >= 0:
                device = f"cuda:{gpu_arg}"
            else:
                logger.warning(f"指定的GPU {gpu_arg} 不可用，回退到自动选择")
                device = "auto"
        else:
            device = device_arg

        # 设置设备
        if device == "auto":
            final_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda"):
            if torch.cuda.is_available():
                # 验证指定的GPU是否存在
                try:
                    gpu_id = int(device.split(":")[1]) if ":" in device else 0
                    if gpu_id < torch.cuda.device_count():
                        final_device = device
                    else:
                        logger.warning(f"指定的GPU {device} 不存在，回退到默认CUDA设备")
                        final_device = "cuda"
                except (ValueError, IndexError):
                    logger.warning(f"无效的设备格式 {device}，回退到默认CUDA设备")
                    final_device = "cuda"
            else:
                logger.warning("CUDA不可用，回退到CPU")
                final_device = "cpu"
        else:
            final_device = device

        logger.info(f"最终设备: {final_device}")


if __name__ == "__main__":
    test_device_selection()
    test_command_line_args()
