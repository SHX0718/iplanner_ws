#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import cv2

# 添加 ZoeDepth 和 iPlanner 路径
workspace_root = "/home/shx/Developments/iplanner_ws"
zoedepth_path = os.path.join(workspace_root, 'ZoeDepth')
iplanner_path = os.path.join(workspace_root, 'src', 'iPlanner')

if zoedepth_path not in sys.path:
    sys.path.insert(0, zoedepth_path)
if iplanner_path not in sys.path:
    sys.path.insert(0, iplanner_path)

from iplanner.zoe_depth_wrapper import ZoeDepthConverter

def main():
    # 输入图像路径
    input_image_path = "/home/shx/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData/campus/camera/100.png"
    
    # 输出深度图路径
    output_depth_path = "/home/shx/Developments/iplanner_ws/depth_output_0.png"
    output_colored_path = "/home/shx/Developments/iplanner_ws/depth_colored_0.png"
    
    print(f"[INFO] 输入图像: {input_image_path}")
    
    # 使用本地模型文件加载 ZoeDepth
    print("[INFO] 正在从本地加载 ZoeDepth_NK 模型...")
    print("[INFO] 模型路径: /home/shx/Developments/iplanner_ws/ZoeDepth/zoedepth/models/ZoeD_M12_NK.pt")
    
    # 使用 ZoeDepthConverter 加载本地模型，强制使用 CPU
    converter = ZoeDepthConverter(model_name="zoedepth_nk", force_cpu=True)
    print("[INFO] 模型加载完成，使用设备: CPU")
    
    # 读取 RGB 图像
    print("[INFO] 正在读取图像...")
    rgb_image = cv2.imread(input_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"[INFO] 图像尺寸: {rgb_image.shape}")
    
    # 转换为深度图
    print("[INFO] 正在转换为深度图...")
    
    # 使用 ZoeDepthConverter 进行转换
    depth_map = converter.rgb_to_depth(rgb_image)
    
    print(f"[INFO] 深度图尺寸: {depth_map.shape}")
    print(f"[INFO] 深度值范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
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
