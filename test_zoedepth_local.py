#!/usr/bin/env python3
"""
测试本地加载 ZoeDepth 模型
"""
import sys
import os

# 添加 src 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'iPlanner'))

import numpy as np
from iplanner.zoe_depth_wrapper import ZoeDepthConverter

def test_zoedepth_local():
    """测试本地模型加载"""
    print("=" * 60)
    print("测试 ZoeDepth 本地模型加载")
    print("=" * 60)
    
    # 测试 zoedepth 模型
    print("\n[1/3] 测试加载 ZoeD_M12_N 模型 (model_name='zoedepth')...")
    try:
        converter_n = ZoeDepthConverter(model_name="zoedepth")
        print("✓ 成功加载 ZoeD_M12_N 模型")
        
        # 创建虚拟输入测试推理
        dummy_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print("  - 运行推理测试...")
        depth = converter_n.rgb_to_depth(dummy_rgb)
        print(f"  - 深度图输出形状: {depth.shape}")
        print(f"  - 深度图数值范围: [{depth.min():.4f}, {depth.max():.4f}]")
        print("✓ ZoeD_M12_N 模型推理正常")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 测试 zoedepth_nk 模型
    print("\n[2/3] 测试加载 ZoeD_M12_NK 模型 (model_name='zoedepth_nk')...")
    try:
        converter_nk = ZoeDepthConverter(model_name="zoedepth_nk")
        print("✓ 成功加载 ZoeD_M12_NK 模型")
        
        # 创建虚拟输入测试推理
        dummy_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print("  - 运行推理测试...")
        depth_tensor = converter_nk.rgb_to_depth_tensor(dummy_rgb)
        print(f"  - 深度图输出类型: {type(depth_tensor)}")
        print(f"  - 深度图输出形状: {depth_tensor.shape}")
        print(f"  - 深度图数值范围: [{depth_tensor.min():.4f}, {depth_tensor.max():.4f}]")
        print("✓ ZoeD_M12_NK 模型推理正常")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 测试输入方式
    print("\n[3/3] 测试不同输入方式...")
    try:
        converter = ZoeDepthConverter(model_name="zoedepth")
        
        # 测试 numpy 数组输入
        print("  - 测试 numpy 数组输入...")
        rgb_np = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth1 = converter.rgb_to_depth(rgb_np)
        print(f"    ✓ numpy 数组输入成功, 输出形状: {depth1.shape}")
        
        # 测试浮点数 numpy 数组输入
        print("  - 测试浮点数 numpy 数组输入...")
        rgb_float = np.random.rand(480, 640, 3).astype(np.float32)
        depth2 = converter.rgb_to_depth(rgb_float)
        print(f"    ✓ 浮点数数组输入成功, 输出形状: {depth2.shape}")
        
        print("✓ 所有输入方式测试正常")
    except Exception as e:
        print(f"✗ 输入测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！模型本地加载正常")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_zoedepth_local()
    sys.exit(0 if success else 1)
