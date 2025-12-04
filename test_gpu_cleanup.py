#!/usr/bin/env python3
"""
GPU清理功能测试脚本
用于测试GPU内存分配和清理功能
"""

import torch
import gc
import signal
import sys
import time
from mylogging import logger


def cleanup_gpu():
    """
    清理GPU内存
    """
    try:
        if torch.cuda.is_available():
            logger.info("正在清理GPU内存...")
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            # 收集CUDA IPC资源
            torch.cuda.ipc_collect()
            # 同步所有CUDA设备
            torch.cuda.synchronize()
            logger.info("GPU内存清理完成")
        # 强制垃圾回收
        gc.collect()
    except Exception as e:
        logger.warning(f"GPU清理过程中出现错误: {e}")


def signal_handler(sig, frame):
    """
    信号处理函数，用于优雅退出
    """
    logger.info(f'接收到信号 {sig}，正在清理资源...')
    cleanup_gpu()
    logger.info('资源清理完成，程序退出')
    sys.exit(0)


def test_gpu_memory():
    """
    测试GPU内存分配和清理
    """
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 初始清理
    cleanup_gpu()

    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法测试GPU内存")
        return

    device = torch.cuda.current_device()
    logger.info(f"使用GPU设备: {device}")
    logger.info(f"GPU名称: {torch.cuda.get_device_name(device)}")

    try:
        logger.info("开始分配GPU内存...")

        # 分配一些GPU内存
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            logger.info(f"第{i+1}次分配后GPU内存使用: {memory_allocated:.2f} MB")
            time.sleep(1)

        logger.info("模拟长时间运行... (按Ctrl+C中断测试)")

        # 模拟长时间运行
        for i in range(60):
            time.sleep(1)
            if i % 10 == 0:
                memory_allocated = torch.cuda.memory_allocated(
                    device) / 1024**2
                logger.info(f"运行{i}秒，GPU内存使用: {memory_allocated:.2f} MB")

        logger.info("测试完成，正在清理...")
        del tensors
        cleanup_gpu()

    except KeyboardInterrupt:
        logger.info("接收到键盘中断信号")
        cleanup_gpu()
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        cleanup_gpu()
        raise
    finally:
        logger.info("最终清理...")
        cleanup_gpu()


if __name__ == "__main__":
    test_gpu_memory()
