#!/usr/bin/env python3
"""
简单测试GPU设备选择功能
"""

import torch
import sys
import os

# 添加当前路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NLImodel import NLIModel
from mylogging import logger


def test_gpu_selection():
    """测试GPU设备选择"""
    
    logger.info("=== GPU设备选择测试 ===")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法测试GPU选择")
        return
    
    device_count = torch.cuda.device_count()
    logger.info(f"可用的CUDA设备数量: {device_count}")
    
    for i in range(device_count):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试自动选择
    logger.info("\n--- 测试自动选择 ---")
    logger.info(f"初始PyTorch当前设备: {torch.cuda.current_device()}")
    
    model_auto = NLIModel(model_name="bert-base-uncased")
    logger.info(f"自动选择后，模型设备: {model_auto.device}")
    logger.info(f"PyTorch当前设备: {torch.cuda.current_device()}")
    del model_auto
    
    # 测试GPU 0
    logger.info("\n--- 测试指定GPU 0 ---")
    model_cuda0 = NLIModel(model_name="bert-base-uncased", device="cuda:0")
    logger.info(f"指定cuda:0后，模型设备: {model_cuda0.device}")
    logger.info(f"PyTorch当前设备: {torch.cuda.current_device()}")
    del model_cuda0
    
    # 测试GPU 1（如果有的话）
    if device_count > 1:
        logger.info("\n--- 测试指定GPU 1 ---")
        model_cuda1 = NLIModel(model_name="bert-base-uncased", device="cuda:1")
        logger.info(f"指定cuda:1后，模型设备: {model_cuda1.device}")
        logger.info(f"PyTorch当前设备: {torch.cuda.current_device()}")
        
        # 创建一个测试张量来验证设备
        test_tensor = torch.randn(2, 3).to(model_cuda1.device)
        logger.info(f"测试张量的设备: {test_tensor.device}")
        
        del model_cuda1, test_tensor
    else:
        logger.info("\n只有一个GPU，跳过GPU 1测试")
    
    logger.info("\n测试完成")


if __name__ == "__main__":
    test_gpu_selection()
