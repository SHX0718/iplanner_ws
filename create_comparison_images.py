#!/usr/bin/env python3
"""
深度图对比可视化工具
功能：将RGB图、真实深度图(depth)和预测深度图(depth_prediction_colored)合并到一张图中
输出：保存到各场景的comparison目录
环境：需在iplanner环境下运行
"""

import os
import time
from PIL import Image
from pathlib import Path
import argparse
import numpy as np


def colorize_depth(depth_image):
    """将灰度深度图转换为彩色深度图（用于真实深度图可视化）"""
    # 如果已经是RGB，直接返回
    if depth_image.mode == 'RGB':
        return depth_image
    
    # 转换为numpy数组
    depth_array = np.array(depth_image)
    
    # 归一化到0-255
    if depth_array.max() > 0:
        depth_normalized = ((depth_array - depth_array.min()) / 
                           (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    else:
        depth_normalized = depth_array.astype(np.uint8)
    
    # 应用颜色映射（使用matplotlib的jet colormap）
    # 这里简化处理，使用灰度图
    # 如果需要彩色，可以使用cv2.applyColorMap
    return Image.fromarray(depth_normalized).convert('RGB')


def get_all_scenes(data_root):
    """获取所有场景目录"""
    scenes = []
    for item in Path(data_root).iterdir():
        if item.is_dir() and (item / "camera").exists():
            scenes.append(item.name)
    return sorted(scenes)


def get_image_files(scene_path):
    """获取场景中所有图像文件"""
    camera_dir = scene_path / "camera"
    if not camera_dir.exists():
        return []
    
    # 获取所有png文件并按数字排序
    images = list(camera_dir.glob("*.png"))
    images.sort(key=lambda x: int(x.stem))
    return images


def create_comparison(rgb_path, depth_path, pred_path, output_path, add_labels=True):
    """创建对比图：RGB | GT Depth | Predicted Depth"""
    try:
        # 读取三张图片
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path)
        pred_img = Image.open(pred_path).convert('RGB')
        
        # 将真实深度图转换为彩色（如果是灰度图）
        depth_img_colored = colorize_depth(depth_img)
        
        # 获取尺寸
        width, height = rgb_img.size
        
        # 确保所有图片尺寸一致
        depth_img_colored = depth_img_colored.resize((width, height), Image.Resampling.LANCZOS)
        pred_img = pred_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # 创建标签高度
        label_height = 30 if add_labels else 0
        
        # 创建合并图像（3列）
        comparison = Image.new('RGB', (width * 3, height + label_height), color='white')
        
        # 粘贴三张图片
        comparison.paste(rgb_img, (0, label_height))
        comparison.paste(depth_img_colored, (width, label_height))
        comparison.paste(pred_img, (width * 2, label_height))
        
        # 添加文字标签（可选）
        if add_labels:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison)
            
            # 使用默认字体
            try:
                # 尝试使用更大的字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 添加标签
            labels = ["RGB Image", "Ground Truth Depth", "Predicted Depth"]
            for i, label in enumerate(labels):
                # 计算文字位置（居中）
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                x = width * i + (width - text_width) // 2
                draw.text((x, 5), label, fill='black', font=font)
        
        # 保存对比图
        comparison.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"  [错误] 创建对比图失败: {str(e)}")
        return False


def process_scene(scene_path, scene_name, add_labels=True):
    """处理单个场景的所有图像"""
    print(f"\n{'='*70}")
    print(f"处理场景: {scene_name}")
    print(f"{'='*70}")
    
    # 检查必要的目录
    camera_dir = scene_path / "camera"
    depth_dir = scene_path / "depth"
    pred_dir = scene_path / "depth_prediction_colored"
    
    if not camera_dir.exists():
        print(f"  [警告] camera目录不存在，跳过")
        return 0, 0
    
    if not depth_dir.exists():
        print(f"  [警告] depth目录不存在，跳过")
        return 0, 0
    
    if not pred_dir.exists():
        print(f"  [警告] depth_prediction_colored目录不存在，跳过")
        return 0, 0
    
    # 获取所有RGB图像
    rgb_images = get_image_files(scene_path)
    if not rgb_images:
        print(f"  [警告] 没有找到RGB图像，跳过")
        return 0, 0
    
    # 创建输出目录
    output_dir = scene_path / "comparison"
    output_dir.mkdir(exist_ok=True)
    
    print(f"  找到 {len(rgb_images)} 张图像")
    print(f"  输出目录: {output_dir}")
    
    scene_start_time = time.time()
    success_count = 0
    
    # 处理每张图像
    for idx, rgb_path in enumerate(rgb_images):
        try:
            # 构建对应的深度图路径
            img_name = rgb_path.name
            depth_path = depth_dir / img_name
            pred_path = pred_dir / img_name
            output_path = output_dir / img_name
            
            # 检查文件是否存在
            if not depth_path.exists():
                print(f"  [警告] {img_name}: 缺少真实深度图，跳过")
                continue
            
            if not pred_path.exists():
                print(f"  [警告] {img_name}: 缺少预测深度图，跳过")
                continue
            
            # 创建对比图
            if create_comparison(rgb_path, depth_path, pred_path, output_path, add_labels):
                success_count += 1
            
            # 每10张图像打印一次进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(rgb_images):
                print(f"  进度: {success_count}/{len(rgb_images)} 张对比图已生成")
        
        except Exception as e:
            print(f"  [错误] 处理 {rgb_path.name} 失败: {str(e)}")
            continue
    
    scene_total_time = time.time() - scene_start_time
    
    # 场景处理总结
    print(f"\n  场景处理完成:")
    print(f"    成功: {success_count}/{len(rgb_images)} 张")
    print(f"    总耗时: {scene_total_time:.2f}秒")
    
    return success_count, scene_total_time


def main():
    parser = argparse.ArgumentParser(description='深度图对比可视化')
    parser.add_argument('--data_root', type=str, 
                        default='/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData',
                        help='数据根目录')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='指定要处理的场景（默认处理所有场景）')
    parser.add_argument('--no_labels', action='store_true',
                        help='不添加文字标签')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("深度图对比可视化工具")
    print("="*70)
    print(f"[数据路径] {args.data_root}")
    print(f"[添加标签] {'否' if args.no_labels else '是'}")
    
    # 记录总耗时
    total_start_time = time.time()
    
    # 获取要处理的场景
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[错误] 数据目录不存在: {data_root}")
        return
    
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = get_all_scenes(data_root)
    
    print(f"\n开始处理 {len(scenes)} 个场景...")
    print(f"场景列表: {', '.join(scenes)}")
    
    # 处理统计
    total_images = 0
    total_processing_time = 0
    scene_stats = []
    
    # 逐个处理场景
    for scene_name in scenes:
        scene_path = data_root / scene_name
        if not scene_path.exists():
            print(f"  [警告] 场景目录不存在: {scene_path}，跳过")
            continue
        
        success_count, scene_time = process_scene(
            scene_path, scene_name, add_labels=not args.no_labels
        )
        
        total_images += success_count
        total_processing_time += scene_time
        scene_stats.append({
            'name': scene_name,
            'count': success_count,
            'time': scene_time
        })
    
    # 总结报告
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("批量处理完成 - 总结报告")
    print("="*70)
    print(f"\n[处理统计]")
    print(f"  处理场景数: {len(scene_stats)}")
    print(f"  生成对比图: {total_images} 张")
    print(f"\n[耗时统计]")
    print(f"  图像处理耗时: {total_processing_time:.2f}秒")
    print(f"  总耗时: {total_time:.2f}秒")
    if total_images > 0:
        print(f"  平均处理速度: {total_images/total_processing_time:.2f} 张/秒")
    
    print(f"\n[各场景详情]")
    for stat in scene_stats:
        print(f"  {stat['name']:15s}: {stat['count']:4d} 张, 耗时 {stat['time']:6.2f}秒")
    
    print("="*70)
    print("✓ 所有场景处理完成！")
    print("="*70)


if __name__ == "__main__":
    main()
