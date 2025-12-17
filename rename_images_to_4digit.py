#!/usr/bin/env python3
"""
图片文件批量重命名工具
功能：将 CollectedData 下所有场景的图片文件名统一改为4位数字格式
示例：1.png -> 0001.png, 389.png -> 0389.png
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# 配置
DATA_ROOT = "/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data/CollectedData"

def get_image_number(filename):
    """从文件名中提取数字"""
    match = re.match(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return None

def rename_images_in_directory(dir_path, dry_run=True):
    """重命名指定目录下的所有图片"""
    if not os.path.exists(dir_path):
        return 0, 0
    
    # 获取所有 PNG 文件
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')])
    
    # 构建重命名映射
    rename_map = {}
    for filename in files:
        num = get_image_number(filename)
        if num is not None:
            new_name = f"{num:04d}.png"
            if filename != new_name:
                rename_map[filename] = new_name
    
    if not rename_map:
        return 0, len(files)
    
    # 执行重命名（使用临时后缀避免冲突）
    renamed_count = 0
    if not dry_run:
        # 第一步：重命名为临时文件名
        temp_map = {}
        for old_name, new_name in rename_map.items():
            old_path = os.path.join(dir_path, old_name)
            temp_name = f"temp_{new_name}"
            temp_path = os.path.join(dir_path, temp_name)
            os.rename(old_path, temp_path)
            temp_map[temp_name] = new_name
            renamed_count += 1
        
        # 第二步：去掉临时前缀
        for temp_name, final_name in temp_map.items():
            temp_path = os.path.join(dir_path, temp_name)
            final_path = os.path.join(dir_path, final_name)
            os.rename(temp_path, final_path)
    else:
        renamed_count = len(rename_map)
    
    return renamed_count, len(files)

def process_scene(scene_path, scene_name, dry_run=True):
    """处理单个场景的所有子目录"""
    subdirs = ['camera', 'depth', 'depth_prediction', 'comparison', 'scan']
    
    print(f"\n{'[预览]' if dry_run else '[执行]'} 场景: {scene_name}")
    print("-" * 60)
    
    total_renamed = 0
    total_files = 0
    
    for subdir in subdirs:
        dir_path = os.path.join(scene_path, subdir)
        renamed, total = rename_images_in_directory(dir_path, dry_run)
        
        if os.path.exists(dir_path):
            status = f"✓ {renamed}/{total} 个文件需要重命名" if renamed > 0 else f"✓ {total} 个文件已是4位格式"
            print(f"  {subdir:20s}: {status}")
            total_renamed += renamed
            total_files += total
    
    return total_renamed, total_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量重命名图片为4位数字格式')
    parser.add_argument('--execute', action='store_true', 
                        help='实际执行重命名（默认只预览）')
    parser.add_argument('--scenes', type=str, nargs='+', 
                        help='指定要处理的场景（默认处理所有）')
    
    args = parser.parse_args()
    
    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        return
    
    # 获取要处理的场景
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([p.name for p in data_root.iterdir() 
                        if p.is_dir() and (p/"camera").exists()])
    
    if not scenes:
        print("❌ 没有找到任何场景目录")
        return
    
    # 显示模式
    mode = "执行模式" if args.execute else "预览模式"
    print("=" * 70)
    print(f" 🔄 图片重命名工具 - {mode}")
    print("=" * 70)
    print(f" 📂 数据根目录: {data_root}")
    print(f" 🎬 待处理场景: {scenes}")
    print(f" 📝 格式转换: N.png → NNNN.png (4位数字)")
    print("=" * 70)
    
    if not args.execute:
        print("\n⚠️  当前为预览模式，不会实际修改文件")
        print("   使用 --execute 参数执行实际重命名\n")
    
    # 处理所有场景
    grand_total_renamed = 0
    grand_total_files = 0
    
    for scene_name in scenes:
        scene_path = data_root / scene_name
        renamed, total = process_scene(str(scene_path), scene_name, dry_run=not args.execute)
        grand_total_renamed += renamed
        grand_total_files += total
    
    # 总结
    print("\n" + "=" * 70)
    print(" 📊 处理汇总")
    print("=" * 70)
    print(f"  总文件数: {grand_total_files}")
    print(f"  需重命名: {grand_total_renamed}")
    print(f"  已是4位格式: {grand_total_files - grand_total_renamed}")
    
    if args.execute:
        print(f"\n✅ 重命名完成！共处理 {grand_total_renamed} 个文件")
    else:
        print(f"\n💡 预览完成！使用 --execute 参数执行实际重命名")
        print(f"   命令示例: python3 {__file__} --execute")

if __name__ == "__main__":
    main()
