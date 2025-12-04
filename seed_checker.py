#!/usr/bin/env python3
"""
随机种子检测和设置工具

提供检测当前随机种子状态和设置随机种子的功能
"""

import torch
import numpy as np
import random
import os
from typing import Optional, Dict, Any


def get_current_random_states() -> Dict[str, Any]:
    """
    获取当前所有随机数生成器的状态
    
    Returns:
        包含各种随机状态的字典
    """
    states = {}

    # Python内置random模块状态
    states['python_random'] = random.getstate()

    # NumPy随机状态
    states['numpy_random'] = np.random.get_state()

    # PyTorch CPU随机状态
    states['torch_cpu'] = torch.get_rng_state()

    # PyTorch CUDA随机状态（如果有GPU）
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()

    return states


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置所有随机种子以确保可重现性
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式（可能影响性能）
    """
    # 设置Python内置random模块种子
    random.seed(seed)

    # 设置NumPy随机种子
    np.random.seed(seed)

    # 设置PyTorch随机种子
    torch.manual_seed(seed)

    # 如果有CUDA，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对所有GPU设备

    # 设置环境变量（影响某些库的随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        # 启用确定性模式（注意：可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 设置环境变量确保确定性
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def check_reproducibility(num_tests: int = 3) -> Dict[str, bool]:
    """
    检查当前设置下的可重现性
    
    Args:
        num_tests: 测试次数
        
    Returns:
        各个组件的可重现性检查结果
    """
    results = {
        'python_random': True,
        'numpy_random': True,
        'torch_cpu': True,
        'torch_cuda': True
    }

    # 保存当前状态
    initial_states = get_current_random_states()

    # 测试Python random
    python_values = []
    for _ in range(num_tests):
        random.setstate(initial_states['python_random'])
        python_values.append(random.random())
    results['python_random'] = len(set(python_values)) == 1

    # 测试NumPy random
    numpy_values = []
    for _ in range(num_tests):
        np.random.set_state(initial_states['numpy_random'])
        numpy_values.append(np.random.random())
    results['numpy_random'] = len(set(numpy_values)) == 1

    # 测试PyTorch CPU
    torch_cpu_values = []
    for _ in range(num_tests):
        torch.set_rng_state(initial_states['torch_cpu'])
        torch_cpu_values.append(torch.rand(1).item())
    results['torch_cpu'] = len(set(torch_cpu_values)) == 1

    # 测试PyTorch CUDA（如果可用）
    if torch.cuda.is_available() and 'torch_cuda' in initial_states:
        torch_cuda_values = []
        for _ in range(num_tests):
            torch.cuda.set_rng_state_all(initial_states['torch_cuda'])
            torch_cuda_values.append(
                torch.cuda.FloatTensor(1).uniform_().item())
        results['torch_cuda'] = len(set(torch_cuda_values)) == 1

    return results


def generate_test_outputs(device: str = 'cpu') -> Dict[str, float]:
    """
    生成测试输出用于检验随机性
    
    Args:
        device: 计算设备
        
    Returns:
        各种随机数生成器的测试输出
    """
    outputs = {}

    # Python random
    outputs['python_random'] = random.random()

    # NumPy random
    outputs['numpy_random'] = np.random.random()

    # PyTorch CPU
    outputs['torch_cpu'] = torch.rand(1).item()

    # PyTorch指定设备
    device_obj = torch.device(device)
    outputs['torch_device'] = torch.rand(1, device=device_obj).item()

    # 如果是CUDA设备，额外测试CUDA随机性
    if device_obj.type == 'cuda':
        outputs['torch_cuda_explicit'] = torch.cuda.FloatTensor(
            1).uniform_().item()

    return outputs


def compare_random_outputs(outputs1: Dict[str, float],
                           outputs2: Dict[str, float]) -> Dict[str, bool]:
    """
    比较两次随机输出是否相同
    
    Args:
        outputs1: 第一次的输出
        outputs2: 第二次的输出
        
    Returns:
        各个组件的一致性检查结果
    """
    comparison = {}

    for key in outputs1:
        if key in outputs2:
            comparison[key] = abs(outputs1[key] - outputs2[key]) < 1e-10
        else:
            comparison[key] = False

    return comparison


def detect_seed_status() -> Dict[str, Any]:
    """
    检测当前随机种子的设置状态
    
    Returns:
        包含种子状态信息的字典
    """
    status = {
        'timestamp':
        torch.initial_seed(),  # PyTorch初始种子
        'python_hashseed':
        os.environ.get('PYTHONHASHSEED', 'Not set'),
        'cudnn_deterministic':
        getattr(torch.backends.cudnn, 'deterministic', False),
        'cudnn_benchmark':
        getattr(torch.backends.cudnn, 'benchmark', True),
        'cuda_launch_blocking':
        os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set'),
    }

    # 检查是否设置了种子（通过生成重复的随机数来检测）
    # 保存当前状态
    current_states = get_current_random_states()

    # 生成第一组随机数
    outputs1 = generate_test_outputs()

    # 恢复状态并生成第二组随机数
    random.setstate(current_states['python_random'])
    np.random.set_state(current_states['numpy_random'])
    torch.set_rng_state(current_states['torch_cpu'])
    if torch.cuda.is_available() and 'torch_cuda' in current_states:
        torch.cuda.set_rng_state_all(current_states['torch_cuda'])

    outputs2 = generate_test_outputs()

    # 比较结果
    reproducibility = compare_random_outputs(outputs1, outputs2)
    status['reproducibility_check'] = reproducibility
    status['likely_seeded'] = all(reproducibility.values())

    return status


def print_seed_status():
    """
    打印当前随机种子状态的详细信息
    """
    print("=" * 60)
    print("随机种子状态检查")
    print("=" * 60)

    status = detect_seed_status()

    print(f"PyTorch初始种子: {status['timestamp']}")
    print(f"PYTHONHASHSEED: {status['python_hashseed']}")
    print(f"CUDNN确定性模式: {status['cudnn_deterministic']}")
    print(f"CUDNN基准模式: {status['cudnn_benchmark']}")
    print(f"CUDA同步启动: {status['cuda_launch_blocking']}")
    print()

    print("可重现性检查:")
    for component, is_reproducible in status['reproducibility_check'].items():
        status_str = "✓ 可重现" if is_reproducible else "✗ 不可重现"
        print(f"  {component}: {status_str}")

    print()
    overall_status = "已正确设置" if status['likely_seeded'] else "未设置或设置不完整"
    print(f"整体评估: 随机种子 {overall_status}")

    if not status['likely_seeded']:
        print("\n建议:")
        print("- 调用 set_random_seed() 函数设置随机种子")
        print("- 确保所有随机数生成器都使用相同的种子")
        print("- 考虑启用确定性模式以获得完全可重现的结果")


if __name__ == "__main__":
    # 演示用法
    print("检查当前随机种子状态:")
    print_seed_status()

    print("\n" + "=" * 60)
    print("设置随机种子后:")
    set_random_seed(42, deterministic=True)
    print_seed_status()

    print("\n" + "=" * 60)
    print("测试可重现性:")

    # 生成第一组输出
    outputs1 = generate_test_outputs()
    print(f"第一次生成: {outputs1}")

    # 重新设置种子
    set_random_seed(42, deterministic=True)

    # 生成第二组输出
    outputs2 = generate_test_outputs()
    print(f"第二次生成: {outputs2}")

    # 比较结果
    comparison = compare_random_outputs(outputs1, outputs2)
    print(f"一致性检查: {comparison}")
