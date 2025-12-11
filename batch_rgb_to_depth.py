#!/usr/bin/env python3
"""
批量RGB图像转深度图转换工具
功能：使用ZoeDepth模型将CollectedData中所有场景的RGB图像转换为深度图
输出：保存到各场景的depth_prediction目录
环境：需在iplanner环境下运行
"""

import sys
import os
import time
import torch
from PIL import Image
from pathlib import Path
import argparse

# 添加ZoeDepth到Python路径
sys.path.insert(0, '/home/tms01/Developments/iplanner_ws/ZoeDepth')

# 导入必要的ZoeDepth组件
from zoedepth.utils.misc import colorize, save_raw_16bit
from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth


def load_model_directly(device):
    """直接加载预训练模型，避免复杂的配置流程"""
    model_path = "/home/tms01/Developments/iplanner_ws/ZoeDepth/zoedepth/models/ZoeD_M12_N.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"    从 {model_path} 加载模型...")
    
    # 使用ZoeDepth.build方法构建模型
    model = ZoeDepth.build(
        midas_model_type="DPT_BEiT_L_384",
        pretrained_resource=f"local::{model_path}",
        use_pretrained_midas=False,
        train_midas=False,
        freeze_midas_bn=True
    )
    
    print("    权重加载成功")
    return model.to(device)


def get_all_scenes(data_root):
    """获取所有场景目录"""
    scenes = []
    for item in Path(data_root).iterdir():
        if item.is_dir() and (item / "camera").exists():
            scenes.append(item.name)
    return sorted(scenes)


def get_rgb_images(scene_path):
    """获取场景中所有RGB图像路径"""
    camera_dir = scene_path / "camera"
    if not camera_dir.exists():
        return []
    
    # 获取所有png文件并按数字排序
    images = list(camera_dir.glob("*.png"))
    images.sort(key=lambda x: int(x.stem))
    return images


def process_scene(model, device, scene_path, scene_name, save_colored=True):
    """处理单个场景的所有RGB图像"""
    print(f"\n{'='*70}")
    print(f"处理场景: {scene_name}")
    print(f"{'='*70}")
    
    # 获取所有RGB图像
    rgb_images = get_rgb_images(scene_path)
    if not rgb_images:
        print(f"  [警告] 场景 {scene_name} 中没有找到RGB图像，跳过")
        return 0, 0
    
    # 创建输出目录
    output_dir = scene_path / "depth_prediction"
    output_dir.mkdir(exist_ok=True)
    
    # 默认同时生成彩色深度图
    colored_dir = scene_path / "depth_prediction_colored"
    colored_dir.mkdir(exist_ok=True)
    
    print(f"  找到 {len(rgb_images)} 张RGB图像")
    print(f"  输出目录: {output_dir}")
    
    scene_start_time = time.time()
    success_count = 0
    total_inference_time = 0
    
    # 处理每张图像
    for idx, img_path in enumerate(rgb_images):
        try:
            # 读取图像
            image = Image.open(img_path).convert("RGB")
            
            # 深度推理
            inference_start = time.time()
            with torch.no_grad():
                depth = model.infer_pil(image)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # 保存原始深度图（16位PNG）
            output_filename = img_path.stem + ".png"
            output_path = output_dir / output_filename
            save_raw_16bit(depth, str(output_path))
            
            # 同时保存彩色深度图用于可视化
            depth_colored = colorize(depth)
            colored_path = colored_dir / output_filename
            Image.fromarray(depth_colored).save(str(colored_path))
            
            success_count += 1
            
            # 每10张图像打印一次进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(rgb_images):
                avg_time = total_inference_time / (idx + 1)
                print(f"  进度: {idx + 1}/{len(rgb_images)} "
                      f"| 平均推理耗时: {avg_time:.3f}秒/张")
        
        except Exception as e:
            print(f"  [错误] 处理 {img_path.name} 失败: {str(e)}")
            continue
    
    scene_total_time = time.time() - scene_start_time
    
    # 场景处理总结
    print(f"\n  场景处理完成:")
    print(f"    成功: {success_count}/{len(rgb_images)} 张")
    print(f"    总耗时: {scene_total_time:.2f}秒")
    if success_count > 0:
        print(f"    平均推理耗时: {total_inference_time/success_count:.3f}秒/张")
    
    return success_count, scene_total_time


def main():
    parser = argparse.ArgumentParser(description='批量RGB转深度图')
    parser.add_argument('--data_root', type=str, 
                        default='/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData',
                        help='数据根目录')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='指定要处理的场景（默认处理所有场景）')
    parser.add_argument('--no_colored', action='store_true',
                        help='不保存彩色深度图（默认会同时生成）')
    
    args = parser.parse_args()
    
    # 设置环境，自动选择GPU/CPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"
    
    print("\n" + "="*70)
    print("ZoeDepth 批量RGB转深度图工具")
    print("="*70)
    print(f"[设备选择] 使用设备: {device}")
    if device == "cuda":
        print(f"[设备信息] GPU型号: {device_name}")
    else:
        print("[警告] 运行在CPU上，这将很慢。请检查CUDA安装。")
    print(f"[数据路径] {args.data_root}")
    
    # 记录总耗时
    total_start_time = time.time()
    
    # 加载模型
    print("\n[步骤1] 加载ZoeDepth模型...")
    model_load_start = time.time()
    
    try:
        model = load_model_directly(device)
        model.eval()
        print("    模型加载成功")
    except Exception as e:
        print(f"    错误: 模型加载失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    model_load_time = time.time() - model_load_start
    print(f"    模型加载耗时: {model_load_time:.4f}秒")
    
    # 初始化MiDaS中可能缺失的属性
    try:
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'Block' in module.__class__.__name__:
                if not hasattr(module, 'drop_path'):
                    # 创建一个dummy drop_path
                    module.drop_path = torch.nn.Identity()
    except:
        pass
    
    # 获取要处理的场景
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[错误] 数据目录不存在: {data_root}")
        return
    
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = get_all_scenes(data_root)
    
    print(f"\n[步骤2] 开始处理 {len(scenes)} 个场景...")
    print(f"  场景列表: {', '.join(scenes)}")
    
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
            model, device, scene_path, scene_name, save_colored=not args.no_colored
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
    print(f"[模型信息] ZoeD_M12_N")
    print(f"[设备信息] {device_name}")
    print(f"\n[处理统计]")
    print(f"  处理场景数: {len(scene_stats)}")
    print(f"  转换图像数: {total_images} 张")
    print(f"\n[耗时统计]")
    print(f"  模型加载耗时: {model_load_time:.4f}秒")
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
