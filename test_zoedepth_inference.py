#!/usr/bin/env python3
"""
测试样例：使用ZoeDepth对图片进行深度估计
用途：演示如何使用ZoeDepth模型对单张图片进行推理
图片路径：/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData/forest/camera/0.png
在iplanner环境下运行
"""

import sys
import os
import time
import torch
from PIL import Image

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


def main():
    # 设置环境，自动选择GPU/CPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # 图片路径
    img_path = "/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData/campus/camera/0.png"
    
    if not os.path.exists(img_path):
        print(f"错误：图片文件不存在 {img_path}")
        return
    
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备选择] 使用设备: {device}")
    if device == "cpu":
        print("[警告] 运行在CPU上，这将很慢。请检查CUDA安装。")
    
    print("\n" + "="*60)
    print("ZoeDepth 深度估计推理测试")
    print("="*60)
    
    # 记录总耗时
    total_start_time = time.time()
    
    # 1. 加载模型
    print("\n[1] 加载ZoeDepth模型...")
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
    
    # 2. 读取图片
    print("\n[2] 读取图片...")
    image_load_start = time.time()
    
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size  # (width, height)
    print(f"    原始图片尺寸: {orig_size}")
    
    image_load_time = time.time() - image_load_start
    print(f"    图片读取耗时: {image_load_time:.4f}秒")
    
    # 3. 深度推理
    print("\n[3] 执行深度推理...")
    inference_start = time.time()
    
    try:
        # 初始化MiDaS中可能缺失的属性
        try:
            for name, module in model.named_modules():
                if hasattr(module, '__class__') and 'Block' in module.__class__.__name__:
                    if not hasattr(module, 'drop_path'):
                        # 创建一个dummy drop_path
                        module.drop_path = torch.nn.Identity()
        except:
            pass
        
        with torch.no_grad():
            depth = model.infer_pil(image)
        
        inference_time = time.time() - inference_start
        print(f"    推理耗时: {inference_time:.4f}秒")
        print(f"    深度图尺寸: {depth.shape}")
        print(f"    深度值范围: [{depth.min():.4f}, {depth.max():.4f}]")
    except Exception as e:
        print(f"    错误: 推理失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 保存结果
    print("\n[4] 保存深度估计结果...")
    save_start = time.time()
    
    # 创建输出目录
    output_dir = "/home/tms01/Developments/iplanner_ws/ZoeDepth/test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 4.1 保存原始深度图（16位）
        raw_depth_path = os.path.join(output_dir, "depth_raw.png")
        save_raw_16bit(depth, raw_depth_path)
        
        # 4.2 保存彩色化深度图
        depth_colored = colorize(depth)
        colored_depth_path = os.path.join(output_dir, "depth_colored.png")
        Image.fromarray(depth_colored).save(colored_depth_path)
        print(f"    彩色深度图已保存: {colored_depth_path}")
        
        # 4.3 保存对比图（原始图与深度图并排）
        depth_pil = Image.fromarray(depth_colored).resize(orig_size, Image.Resampling.LANCZOS)
        comparison = Image.new("RGB", (orig_size[0]*2, orig_size[1]))
        comparison.paste(image, (0, 0))
        comparison.paste(depth_pil.convert("RGB"), (orig_size[0], 0))
        comparison_path = os.path.join(output_dir, "comparison.png")
        comparison.save(comparison_path)
        print(f"    对比图已保存: {comparison_path}")
        
        save_time = time.time() - save_start
        print(f"    保存耗时: {save_time:.4f}秒")
    except Exception as e:
        print(f"    错误: 保存结果失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 总结
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("性能统计总结")
    print("="*60)
    print(f"模型加载耗时:   {model_load_time:.4f}秒 ({model_load_time/total_time*100:.1f}%)")
    print(f"图片读取耗时:   {image_load_time:.4f}秒 ({image_load_time/total_time*100:.1f}%)")
    print(f"深度推理耗时:   {inference_time:.4f}秒 ({inference_time/total_time*100:.1f}%)")
    print(f"结果保存耗时:   {save_time:.4f}秒 ({save_time/total_time*100:.1f}%)")
    print("-"*60)
    print(f"总耗时:        {total_time:.4f}秒")
    print(f"使用设备:      {device}")
    print("="*60)
    print(f"\n✓ 推理完成！输出文件位置: {output_dir}")


if __name__ == "__main__":
    main()
