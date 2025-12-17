#!/usr/bin/env python3
"""
批量处理RGB图片并与真实深度图对比
功能:
1. 使用ZoeDepth将RGB图片转换成预测深度图
2. 将RGB图、真实深度图和预测深度图拼接成一张对比图

数据路径: /home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData
场景: indoor, campus, forest, garage, tunnel
"""

import sys
import os
import time
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加ZoeDepth到Python路径
sys.path.insert(0, '/home/tms01/Developments/iplanner_ws/ZoeDepth')

# 导入必要的ZoeDepth组件
from zoedepth.utils.misc import colorize, save_raw_16bit
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


# 配置路径
DATA_BASE_DIR = "/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData"
MODEL_PATH = "/home/tms01/Developments/iplanner_ws/ZoeDepth/zoedepth/models/ZoeDepthNKv1_14-Dec_21-34-fa7c108ac8c1_best.pt"
# 输出目录将在各场景目录下创建
# - depth_prediction: 预测深度图
# - comparison: 三图拼接对比图

# 场景列表
SCENES = ["indoor", "campus", "forest", "garage", "tunnel"]


def load_model(device):
    """加载ZoeDepth模型"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    
    print(f"[模型加载] 从 {MODEL_PATH} 加载模型...")
    
    # 使用local::前缀让ZoeDepth自己加载模型
    pretrained_resource = f"local::{MODEL_PATH}"
    conf = get_config("zoedepth_nk", "infer", pretrained_resource=pretrained_resource)
    conf['force_reload'] = False  # 不重新下MiDaS
    conf['use_pretrained_midas'] = False  # 不使用预训练MiDaS
    
    model = build_model(conf)
    model = model.to(device)
    model.eval()
    
    # 修复MiDaS Block中缺失的drop_path属性
    for name, module in model.named_modules():
        if hasattr(module, '__class__') and 'Block' in module.__class__.__name__:
            if not hasattr(module, 'drop_path'):
                module.drop_path = torch.nn.Identity()
    
    print(f"[模型加载] 完成，使用设备: {device}")
    return model


def colorize_depth(depth, cmap='magma_r'):
    """将深度图转换为彩色图像"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    
    # 归一化深度值
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # 应用colormap
    colormap = cm.get_cmap(cmap)
    colored = colormap(depth_normalized)
    
    # 转换为uint8
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    return colored_uint8


def load_gt_depth(depth_path):
    """加载真实深度图并转换为彩色"""
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img)
    
    # 如果是16位深度图，转换为float
    if depth_array.dtype == np.uint16:
        depth_array = depth_array.astype(np.float32) / 1000.0  # 假设单位为mm
    elif depth_array.dtype == np.uint8:
        depth_array = depth_array.astype(np.float32)
    
    return depth_array


def process_scene(model, scene_name, device):
    """处理单个场景的所有图片"""
    camera_dir = os.path.join(DATA_BASE_DIR, scene_name, "camera")
    depth_dir = os.path.join(DATA_BASE_DIR, scene_name, "depth")
    
    if not os.path.exists(camera_dir):
        print(f"[警告] 场景 {scene_name} 的camera目录不存在: {camera_dir}")
        return 0
    
    # 获取所有RGB图片
    rgb_files = sorted(glob.glob(os.path.join(camera_dir, "*.png")))
    
    if len(rgb_files) == 0:
        print(f"[警告] 场景 {scene_name} 没有找到RGB图片")
        return 0
    
    # 在场景目录下创建输出子目录
    scene_depth_output = os.path.join(DATA_BASE_DIR, scene_name, "depth_prediction")
    scene_comparison_output = os.path.join(DATA_BASE_DIR, scene_name, "comparison")
    os.makedirs(scene_depth_output, exist_ok=True)
    os.makedirs(scene_comparison_output, exist_ok=True)
    
    processed_count = 0
    total_inference_time = 0
    
    print(f"\n[场景: {scene_name}] 共 {len(rgb_files)} 张图片")
    
    for i, rgb_path in enumerate(tqdm(rgb_files, desc=f"{scene_name}", unit=" img", ncols=80)):
        filename = os.path.basename(rgb_path)
        
        # 对应的真实深度图路径
        gt_depth_path = os.path.join(depth_dir, filename)
        
        try:
            # 1. 读取RGB图片
            rgb_image = Image.open(rgb_path).convert("RGB")
            orig_size = rgb_image.size  # (width, height)
            
            # 2. 使用ZoeDepth进行深度预测
            inference_start = time.time()
            with torch.no_grad():
                pred_depth = model.infer_pil(rgb_image)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # 3. 保存预测深度图（16位原始数据，用于误差分析）
            pred_depth_array = np.array(pred_depth)  # 转为numpy数组
            # 调整为与GT相同的尺寸
            from PIL import Image as PILImage
            pred_depth_resized = np.array(PILImage.fromarray(pred_depth_array).resize(orig_size, PILImage.Resampling.LANCZOS))
            # 转换为毫米并保存为16位图像（与GT格式一致）
            pred_depth_mm = (pred_depth_resized * 1000).astype(np.uint16)
            pred_depth_save_path = os.path.join(scene_depth_output, filename)
            Image.fromarray(pred_depth_mm).save(pred_depth_save_path)
            
            # 4. 创建拼接对比图（RGB + 真实深度 + 预测深度）
            # 将预测深度图调整为原图尺寸（使用与真实深度图相同的彩色风格）
            pred_depth_colored_same = colorize_depth(pred_depth_array)  # 使用相同的colorize_depth函数
            pred_depth_pil = Image.fromarray(pred_depth_colored_same).resize(orig_size, Image.Resampling.LANCZOS)
            
            # 加载真实深度图
            if os.path.exists(gt_depth_path):
                gt_depth_array = load_gt_depth(gt_depth_path)
                gt_depth_colored = colorize_depth(gt_depth_array)
                gt_depth_pil = Image.fromarray(gt_depth_colored).resize(orig_size, Image.Resampling.LANCZOS)
            else:
                # 如果没有真实深度图，用灰色图代替
                gt_depth_pil = Image.new("RGB", orig_size, (128, 128, 128))
            
            # 创建拼接图（水平排列：RGB | GT Depth | Pred Depth）
            comparison = Image.new("RGB", (orig_size[0] * 3, orig_size[1]))
            comparison.paste(rgb_image, (0, 0))
            comparison.paste(gt_depth_pil, (orig_size[0], 0))
            comparison.paste(pred_depth_pil, (orig_size[0] * 2, 0))
            
            # 保存对比图
            comparison_save_path = os.path.join(scene_comparison_output, filename)
            comparison.save(comparison_save_path)
            
            processed_count += 1
        
        except Exception as e:
            print(f"    [错误] 处理 {filename} 失败: {str(e)}")
            continue
    
    avg_inference_time = total_inference_time / max(processed_count, 1)
    print(f"    场景完成: 处理 {processed_count} 张, 平均推理耗时: {avg_inference_time:.3f}s")
    
    return processed_count


def main():
    print("="*70)
    print("RGB图片批量深度预测与对比工具")
    print("="*70)
    
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] 使用: {device}")
    
    # 输出目录说明
    print(f"[输出目录]")
    print(f"    预测深度图: <场景目录>/depth_prediction/")
    print(f"    对比图: <场景目录>/comparison/")
    
    # 加载模型
    total_start = time.time()
    model_start = time.time()
    model = load_model(device)
    model_load_time = time.time() - model_start
    print(f"[模型加载耗时] {model_load_time:.2f}s")
    
    # 处理每个场景
    total_processed = 0
    for scene in SCENES:
        processed = process_scene(model, scene, device)
        total_processed += processed
    
    # 总结
    total_time = time.time() - total_start
    print("\n" + "="*70)
    print("处理完成总结")
    print("="*70)
    print(f"总共处理图片: {total_processed} 张")
    print(f"总耗时: {total_time:.2f}s")
    print(f"输出位置: {DATA_BASE_DIR}/<场景>/depth_prediction/ 和 comparison/")
    print("="*70)


if __name__ == "__main__":
    main()
