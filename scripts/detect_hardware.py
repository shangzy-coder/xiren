#!/usr/bin/env python3
"""
硬件检测脚本
根据系统环境自动选择CPU或GPU版本的依赖
"""
import subprocess
import sys
import os

def check_nvidia_gpu():
    """检查是否有可用的NVIDIA GPU"""
    try:
        # 检查nvidia-smi命令
        result = subprocess.run(['nvidia-smi'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        # 检查nvcc命令
        result = subprocess.run(['nvcc', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_requirements_file():
    """根据硬件环境返回合适的requirements文件"""
    has_gpu = check_nvidia_gpu()
    has_cuda = check_cuda_available()
    
    if has_gpu and has_cuda:
        print("检测到NVIDIA GPU和CUDA环境", file=sys.stderr)
        return "requirements-gpu.txt"
    else:
        print("使用CPU环境", file=sys.stderr)
        return "requirements-cpu.txt"

if __name__ == "__main__":
    requirements_file = get_requirements_file()
    print(requirements_file)
