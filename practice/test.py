"""
这是一个测试脚本，用于检查当前设备是否支持MPS（Metal Performance Shaders）。
"""

import torch

def test():
    """
    测试当前设备是否支持MPS（Metal Performance Shaders）。
    Returns:
        str: 返回当前设备名称，如果支持MPS则返回"mps"，否则返回"cpu"。
    """
    _device = "mps" if torch.backends.mps.is_available() else "cpu" # 默认使用MPS设备

    return _device

if __name__ == "__main__":
    device=test()
    print(f"当前设备: {device}")
