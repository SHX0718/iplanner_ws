#!/usr/bin/env python3
import sys
import os
import numpy as np
import cv2
import time

# 优化PyTorch加载速度
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

# 添加 iPlanner 路径
workspace_root = os.path.dirname(os.path.abspath(__file__))
iplanner_path = os.path.join(workspace_root, 'src', 'iPlanner')
sys.path.insert(0, iplanner_path)

from iplanner.zoe_depth_wrapper import ZoeDepthConverter

def main():
    # 输入图像路径
    input_image_path = "/home/shx/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData/campus/camera/0.png"
    
    # 输出深度图路径
    output_depth_path = "/home/shx/Developments/iplanner_ws/depth_output_0.png"
    output_colored_path = "/home/shx/Developments/iplanner_ws/depth_colored_0.png"
    
    print(f"[INFO] 输入图像: {input_image_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_image_path):
        print(f"[ERROR] 输入图像不存在: {input_image_path}")
        return
    
    # 初始化 ZoeDepth 转换器
    print("[INFO] 正在初始化 ZoeDepth 模型...")
    init_start = time.time()
    
    # 尝试使用 NK 模型（更稳定）
    try:
        converter = ZoeDepthConverter(model_name="zoedepth_nk")
        init_time = time.time() - init_start
        print(f"[INFO] 使用 ZoeDepth_NK 模型初始化完成 (耗时: {init_time:.2f}秒)")
    except Exception as e:
        print(f"[WARN] NK模型加载失败: {e}")
        print("[INFO] 尝试使用默认 ZoeDepth 模型...")
        converter = ZoeDepthConverter(model_name="zoedepth")
        init_time = time.time() - init_start
        print(f"[INFO] 模型初始化完成 (耗时: {init_time:.2f}秒)")
    
    # 读取 RGB 图像
    print("[INFO] 正在读取图像...")
    rgb_image = cv2.imread(input_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是 BGR，转换为 RGB
    print(f"[INFO] 图像尺寸: {rgb_image.shape}")
    
    # 转换为深度图
    print("[INFO] 正在转换为深度图...")
    infer_start = time.time()
    depth_map = converter.rgb_to_depth(rgb_image)
    infer_time = time.time() - infer_start
    print(f"[INFO] 深度图尺寸: {depth_map.shape}")
    print(f"[INFO] 深度值范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"[INFO] 推理耗时: {infer_time:.2f}秒")
    
    # 保存原始深度图（归一化到 0-255）
    depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    cv2.imwrite(output_depth_path, depth_normalized)
    print(f"[INFO] 深度图已保存到: {output_depth_path}")
    
    # 保存彩色深度图（用于可视化）
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(output_colored_path, depth_colored)
    print(f"[INFO] 彩色深度图已保存到: {output_colored_path}")
    
    print("\n[SUCCESS] 转换完成！")
    print(f"  - 原始深度图: {output_depth_path}")
    print(f"  - 彩色深度图: {output_colored_path}")

if __name__ == "__main__":
    main()
