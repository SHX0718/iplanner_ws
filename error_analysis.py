#!/usr/bin/env python3
"""
深度预测误差分析工具 (多进程加速版)
功能：利用多核CPU并行分析ZoeDepth预测深度图与真实深度图的误差
优势：比单核版快 10-20 倍
环境：需在iplanner环境下运行
依赖：pip install tqdm matplotlib opencv-python numpy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import matplotlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 设置matplotlib后端
matplotlib.use('Agg')

# ================= 配置区域 =================
DATA_ROOT = "/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData"
DEPTH_SCALE = 1000.0 
ERROR_THRESHOLD = 0.15 
# ===========================================

def apply_color_map(depth_map):
    """生成伪彩色深度图"""
    norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    norm_depth = np.uint8(norm_depth)
    return cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)

def process_single_image(args):
    """
    单个图像处理函数 (必须是顶层函数以便多进程pickle)
    args: (base_path, file_id, output_dir, error_threshold)
    """
    base_path, file_id, output_dir, error_threshold = args
    
    # 路径构建
    rgb_path = os.path.join(base_path, "camera", file_id)
    gt_path = os.path.join(base_path, "depth", file_id)
    pred_raw_path = os.path.join(base_path, "depth_prediction", file_id)
    pred_colored_path = os.path.join(base_path, "depth_prediction_colored", file_id)
    output_path = os.path.join(output_dir, file_id)

    # 如果目标文件已存在，跳过（断点续传）
    # if os.path.exists(output_path):
    #     return True

    if not all(os.path.exists(p) for p in [rgb_path, gt_path, pred_raw_path]):
        return False

    try:
        # --- 1. 读取 ---
        img_rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        img_gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        img_pred_raw = cv2.imread(pred_raw_path, cv2.IMREAD_UNCHANGED)
        
        # 深度图预处理
        depth_gt = img_gt_raw.astype(np.float32) / DEPTH_SCALE
        depth_pred = img_pred_raw.astype(np.float32) / DEPTH_SCALE

        # 彩色图处理
        if os.path.exists(pred_colored_path):
            img_pred_colored = cv2.cvtColor(cv2.imread(pred_colored_path), cv2.COLOR_BGR2RGB)
            if img_pred_colored.shape[:2] != img_rgb.shape[:2]:
                img_pred_colored = cv2.resize(img_pred_colored, (img_rgb.shape[1], img_rgb.shape[0]))
        else:
            img_pred_colored = cv2.cvtColor(apply_color_map(img_pred_raw), cv2.COLOR_BGR2RGB)

        # --- 2. 计算 ---
        valid_mask = depth_gt > 0.001
        
        abs_diff = np.abs(depth_pred - depth_gt)
        rel_diff = abs_diff / (depth_gt + 1e-6)
        
        abs_diff[~valid_mask] = 0
        rel_diff[~valid_mask] = 0
        
        signed_diff = depth_pred - depth_gt
        signed_diff[~valid_mask] = 0

        bad_pixels_mask = (rel_diff > error_threshold) & valid_mask

        # 无人机安全逻辑
        fatal_mask = (depth_gt < 3.0) & (depth_pred > depth_gt * 1.25) & valid_mask
        false_alarm_mask = (depth_gt > 3.0) & (depth_pred < depth_gt * 0.7) & valid_mask

        # 统计
        valid_count = np.sum(valid_mask)
        mae = np.mean(abs_diff[valid_mask]) if valid_count > 0 else 0
        mre = np.mean(rel_diff[valid_mask]) if valid_count > 0 else 0
        error_ratio = np.sum(bad_pixels_mask) / valid_count * 100 if valid_count > 0 else 0
        fatal_ratio = np.sum(fatal_mask) / valid_count * 100 if valid_count > 0 else 0

        # --- 3. 叠加图 ---
        overlay_img = img_rgb.copy()
        red_layer = np.zeros_like(img_rgb); red_layer[:] = [255, 0, 0]
        yellow_layer = np.zeros_like(img_rgb); yellow_layer[:] = [255, 255, 0]

        if np.any(false_alarm_mask):
            overlay_img[false_alarm_mask] = cv2.addWeighted(img_rgb[false_alarm_mask], 0.6, yellow_layer[false_alarm_mask], 0.4, 0)
        if np.any(fatal_mask):
            overlay_img[fatal_mask] = cv2.addWeighted(img_rgb[fatal_mask], 0.6, red_layer[fatal_mask], 0.4, 0)

        # --- 4. 绘图 ---
        # 技巧：降低 dpi 和 figure size 可以显著提升速度
        fig, axes = plt.subplots(2, 4, figsize=(20, 9)) 
        plt.subplots_adjust(wspace=0.1, hspace=0.15)
        ax = axes.flatten()

        # [0] RGB
        ax[0].imshow(img_rgb)
        ax[0].set_title(f"1. RGB ({file_id})")
        ax[0].axis('off')

        # [1] GT
        max_d = np.percentile(depth_gt[valid_mask], 98) if valid_count > 0 else 10.0
        im1 = ax[1].imshow(depth_gt, cmap='magma', vmin=0, vmax=max_d)
        ax[1].set_title("2. GT Depth")
        ax[1].axis('off')
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        # [2] Pred
        ax[2].imshow(img_pred_colored)
        ax[2].set_title("3. Pred (Vis)")
        ax[2].axis('off')

        # [3] Abs Error
        im3 = ax[3].imshow(abs_diff, cmap='jet', vmin=0, vmax=2.0)
        ax[3].set_title("4. Abs Error")
        ax[3].axis('off')
        plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        # [4] Rel Error
        im4 = ax[4].imshow(rel_diff, cmap='inferno', vmin=0, vmax=0.5)
        ax[4].set_title("5. Rel Error")
        ax[4].axis('off')
        plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

        # [5] Bias
        im5 = ax[5].imshow(signed_diff, cmap='bwr', vmin=-1.0, vmax=1.0)
        ax[5].set_title("6. Bias")
        ax[5].axis('off')
        plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

        # [6] Mask
        ax[6].imshow(bad_pixels_mask, cmap='gray')
        ax[6].set_title(f"7. Mask (>{error_threshold:.0%})")
        ax[6].axis('off')

        # [7] Overlay
        ax[7].imshow(overlay_img)
        ax[7].set_title("8. Safety Overlay")
        ax[7].axis('off')

        scene_name = os.path.basename(base_path)
        plt.suptitle(f"{scene_name} / {file_id} | MAE:{mae:.2f}m MRE:{mre:.1%} Fatal:{fatal_ratio:.1f}%", fontsize=14)
        
        plt.savefig(output_path, dpi=100, bbox_inches='tight') # dpi 100 足够浏览，速度更快
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"Error processing {file_id}: {e}")
        return False

def get_all_images(scene_path):
    camera_dir = scene_path / "camera"
    if not camera_dir.exists(): return []
    images = list(camera_dir.glob("*.png"))
    try:
        images.sort(key=lambda x: int(x.stem))
    except:
        images.sort()
    return [img.name for img in images]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--scenes', type=str, nargs='+', default=None)
    # 默认使用 CPU 核心数 - 2，防止卡死系统
    default_workers = max(1, multiprocessing.cpu_count() - 2)
    parser.add_argument('--workers', type=int, default=default_workers, help='多进程并行数量')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Path not found: {data_root}")
        return
    
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([p.name for p in data_root.iterdir() if p.is_dir() and (p/"camera").exists()])
    
    print(f"========================================")
    print(f" 🚀 多进程加速启动 | Workers: {args.workers}")
    print(f" 📂 数据集: {data_root}")
    print(f" 🎬 待处理场景: {scenes}")
    print(f"========================================")

    total_count = 0

    for scene_name in scenes:
        scene_path = data_root / scene_name
        output_dir = scene_path / "error_analysis"
        output_dir.mkdir(exist_ok=True)
        
        all_images = get_all_images(scene_path)
        if not all_images: continue
        
        print(f"\n正在处理: {scene_name} ({len(all_images)} 张)")
        
        # 准备参数列表
        task_args = [(str(scene_path), img_name, str(output_dir), ERROR_THRESHOLD) for img_name in all_images]
        
        # 启动进程池
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # 使用 tqdm 监控进度
            results = list(tqdm(
                executor.map(process_single_image, task_args), 
                total=len(task_args),
                unit="img"
            ))
            
        success = sum(results)
        total_count += success
        print(f"完成: {success}/{len(all_images)}")

    print(f"\n🎉 全部处理完成! 总计生成: {total_count} 张分析图")

if __name__ == "__main__":
    main()