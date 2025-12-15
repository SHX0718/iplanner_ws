#!/usr/bin/env python3
"""
TSDF 地图可视化工具
功能：
1. 从已有数据加载并可视化 TSDF 地图
2. 从点云重建并可视化 TSDF 地图
3. 支持代价地图和原始点云两种可视化模式

使用方法：
    # 可视化已有 TSDF 地图
    python visualize_tsdf.py --scene forest --mode cost
    
    # 从点云重建 TSDF 地图
    python visualize_tsdf.py --scene forest --rebuild
    
环境：需在 iplanner 环境下运行
"""

import sys
import os
import argparse
import numpy as np
import open3d as o3d

# 添加 iplanner 路径
sys.path.insert(0, '/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner')

from tsdf_map import TSDF_Map
from esdf_mapping import TSDF_Creator, DepthReconstruction

# 数据路径
DATA_ROOT = "/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data"
COLLECTED_DATA = os.path.join(DATA_ROOT, "CollectedData")
TRAINING_DATA = os.path.join(DATA_ROOT, "TrainingData")


def list_available_scenes():
    """列出可用的场景"""
    print("\n可用场景：")
    
    # 检查 CollectedData
    if os.path.exists(COLLECTED_DATA):
        scenes = [d for d in os.listdir(COLLECTED_DATA) 
                  if os.path.isdir(os.path.join(COLLECTED_DATA, d))]
        if scenes:
            print(f"  CollectedData: {', '.join(scenes)}")
    
    # 检查 TrainingData
    if os.path.exists(TRAINING_DATA):
        scenes = [d for d in os.listdir(TRAINING_DATA) 
                  if os.path.isdir(os.path.join(TRAINING_DATA, d))]
        if scenes:
            print(f"  TrainingData: {', '.join(scenes)}")
    print()


def visualize_existing_tsdf(scene_path, map_name="tsdf1", show_cost_map=True):
    """
    可视化已有的 TSDF 地图
    
    Args:
        scene_path: 场景数据路径
        map_name: 地图名称（默认 tsdf1）
        show_cost_map: True 显示代价地图，False 显示原始点云
    """
    print(f"\n{'='*60}")
    print(f"加载 TSDF 地图: {scene_path}")
    print(f"地图名称: {map_name}")
    print(f"{'='*60}")
    
    # 检查地图文件是否存在
    map_file = os.path.join(scene_path, "maps", "data", f"{map_name}_map.txt")
    if not os.path.exists(map_file):
        print(f"[错误] 地图文件不存在: {map_file}")
        print("请先使用 --rebuild 选项重建地图")
        return False
    
    # 加载 TSDF 地图
    tsdf_map = TSDF_Map()
    tsdf_map.ReadTSDFMap(scene_path, map_name)
    
    print(f"\n地图参数:")
    print(f"  - 体素大小: {tsdf_map.voxel_size:.4f} m")
    print(f"  - 地图尺寸: {tsdf_map.num_x} x {tsdf_map.num_y}")
    print(f"  - 起始坐标: ({tsdf_map.start_x:.2f}, {tsdf_map.start_y:.2f})")
    print(f"  - 清除距离: {tsdf_map.clear_dist:.2f} m")
    
    # 可视化
    if show_cost_map:
        print("\n[可视化] 显示代价地图（高度表示代价值）...")
        # 为代价地图添加颜色
        pcd = tsdf_map.pcd_tsdf
        points = np.asarray(pcd.points)
        # 根据高度（代价值）着色
        heights = points[:, 2]
        colors = colorize_by_height(heights)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], 
            window_name="TSDF Cost Map",
            width=1280, height=720)
    else:
        print("\n[可视化] 显示原始点云...")
        pcd = tsdf_map.pcd_viz
        # 为点云添加颜色
        points = np.asarray(pcd.points)
        colors = colorize_by_height(points[:, 2])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd],
            window_name="TSDF Point Cloud",
            width=1280, height=720)
    
    return True


def rebuild_tsdf_from_pointcloud(scene_path, map_name="tsdf1", voxel_size=0.05, 
                                  robot_size=0.3, robot_height=0.75, visualize=True):
    """
    从点云重建 TSDF 地图
    
    Args:
        scene_path: 场景数据路径
        map_name: 输出地图名称
        voxel_size: 体素大小
        robot_size: 机器人半径（用于膨胀障碍物）
        robot_height: 机器人高度
        visualize: 是否可视化
    """
    print(f"\n{'='*60}")
    print(f"从点云重建 TSDF 地图")
    print(f"场景路径: {scene_path}")
    print(f"{'='*60}")
    
    # 检查点云文件
    cloud_file = os.path.join(scene_path, "cloud.ply")
    if not os.path.exists(cloud_file):
        print(f"[错误] 点云文件不存在: {cloud_file}")
        return False
    
    print(f"\n参数设置:")
    print(f"  - 体素大小: {voxel_size} m")
    print(f"  - 机器人半径: {robot_size} m")
    print(f"  - 机器人高度: {robot_height} m")
    
    # 创建 TSDF
    print("\n[1/3] 初始化 TSDF Creator...")
    tsdf_creator = TSDF_Creator(scene_path, 
                                 voxel_size=voxel_size, 
                                 robot_size=robot_size, 
                                 robot_height=robot_height)
    
    print("[2/3] 读取点云并进行地形分析...")
    tsdf_creator.read_point_from_file("cloud.ply")
    
    print("[3/3] 创建 TSDF 地图...")
    data, coord, params = tsdf_creator.create_TSDF_map()
    
    # 保存地图
    print("\n保存 TSDF 地图...")
    tsdf_map = TSDF_Map()
    tsdf_map.DirectLoadMap(data, coord, params)
    tsdf_map.SaveTSDFMap(scene_path, map_name)
    
    # 可视化
    if visualize:
        print("\n[可视化] 显示结果...")
        
        # 可视化障碍物点云和自由空间点云
        print("  - 红色: 障碍物点云")
        print("  - 绿色: 自由空间点云")
        
        obs_pcd = tsdf_creator.obs_pcd
        obs_pcd.paint_uniform_color([1, 0, 0])  # 红色
        
        free_pcd = tsdf_creator.free_pcd
        free_pcd.paint_uniform_color([0, 1, 0])  # 绿色
        
        o3d.visualization.draw_geometries([obs_pcd, free_pcd],
            window_name="Terrain Analysis (Red=Obstacle, Green=Free)",
            width=1280, height=720)
        
        # 可视化 TSDF 代价地图
        print("\n[可视化] 显示 TSDF 代价地图...")
        tsdf_map.ShowTSDFMap(cost_map=True)
    
    return True


def reconstruct_and_create_tsdf(input_scene_path, output_path, map_name="tsdf1",
                                 voxel_size=0.05, robot_size=0.3, max_depth=10.0,
                                 is_flat_ground=True, visualize=True):
    """
    从深度图重建点云并创建 TSDF 地图（完整流程）
    
    Args:
        input_scene_path: 输入场景路径（包含 depth/ 文件夹）
        output_path: 输出路径
        map_name: 地图名称
        voxel_size: 体素大小
        robot_size: 机器人半径
        max_depth: 最大深度范围
        is_flat_ground: 是否假设平坦地面
        visualize: 是否可视化
    """
    print(f"\n{'='*60}")
    print(f"从深度图重建 TSDF 地图（完整流程）")
    print(f"输入路径: {input_scene_path}")
    print(f"输出路径: {output_path}")
    print(f"{'='*60}")
    
    # 检查深度图文件夹
    depth_folder = os.path.join(input_scene_path, "depth")
    if not os.path.exists(depth_folder):
        print(f"[错误] 深度图文件夹不存在: {depth_folder}")
        return False
    
    # Step 1: 深度图重建点云
    print("\n[Step 1/3] 从深度图重建点云...")
    depth_constructor = DepthReconstruction(
        input_scene_path, output_path, 
        start_id=0, iters=100, 
        voxel_size=voxel_size * 0.9,
        max_range=max_depth,
        is_max_iter=True
    )
    depth_constructor.depth_map_reconstruction(is_flat_ground=is_flat_ground)
    depth_constructor.save_reconstructed_data(image_type="depth")
    avg_height = depth_constructor.avg_height
    print(f"  平均高度: {avg_height:.2f} m")
    
    if visualize:
        print("  [可视化] 显示重建的点云...")
        depth_constructor.show_point_cloud()
    
    # Step 2: 创建 TSDF 地图
    print("\n[Step 2/3] 创建 TSDF 地图...")
    tsdf_creator = TSDF_Creator(output_path, 
                                 voxel_size=voxel_size, 
                                 robot_size=robot_size, 
                                 robot_height=avg_height)
    tsdf_creator.read_point_from_file("cloud.ply")
    data, coord, params = tsdf_creator.create_TSDF_map()
    
    if visualize:
        print("  [可视化] 显示障碍物和自由空间...")
        obs_pcd = tsdf_creator.obs_pcd
        obs_pcd.paint_uniform_color([1, 0, 0])
        free_pcd = tsdf_creator.free_pcd
        free_pcd.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([obs_pcd, free_pcd],
            window_name="Terrain Analysis",
            width=1280, height=720)
    
    # Step 3: 保存 TSDF 地图
    print("\n[Step 3/3] 保存 TSDF 地图...")
    tsdf_map = TSDF_Map()
    tsdf_map.DirectLoadMap(data, coord, params)
    tsdf_map.SaveTSDFMap(output_path, map_name)
    
    if visualize:
        print("  [可视化] 显示 TSDF 代价地图...")
        tsdf_map.ShowTSDFMap(cost_map=True)
    
    print("\n[完成] TSDF 地图创建成功！")
    return True


def colorize_by_height(heights, cmap='viridis'):
    """根据高度值生成颜色"""
    import matplotlib.pyplot as plt
    
    # 归一化高度值
    h_min, h_max = heights.min(), heights.max()
    if h_max - h_min > 1e-6:
        normalized = (heights - h_min) / (h_max - h_min)
    else:
        normalized = np.zeros_like(heights)
    
    # 使用 colormap
    cmap_func = plt.get_cmap(cmap)
    colors = cmap_func(normalized)[:, :3]  # 只取 RGB，不要 alpha
    
    return colors


def main():
    parser = argparse.ArgumentParser(description='TSDF 地图可视化工具')
    parser.add_argument('--scene', type=str, default=None,
                        help='场景名称 (如 forest, garage, campus 等)')
    parser.add_argument('--path', type=str, default=None,
                        help='自定义场景路径（优先于 --scene）')
    parser.add_argument('--map_name', type=str, default='tsdf1',
                        help='地图名称（默认: tsdf1）')
    parser.add_argument('--mode', type=str, choices=['cost', 'cloud'], default='cost',
                        help='可视化模式: cost=代价地图, cloud=点云（默认: cost）')
    parser.add_argument('--rebuild', action='store_true',
                        help='从点云重建 TSDF 地图')
    parser.add_argument('--full_rebuild', action='store_true',
                        help='从深度图完整重建（包含点云重建）')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='体素大小（默认: 0.05m）')
    parser.add_argument('--robot_size', type=float, default=0.3,
                        help='机器人半径（默认: 0.3m）')
    parser.add_argument('--list', action='store_true',
                        help='列出可用场景')
    
    args = parser.parse_args()
    
    # 列出可用场景
    if args.list:
        list_available_scenes()
        return
    
    # 确定场景路径
    if args.path:
        scene_path = args.path
    elif args.scene:
        # 优先检查 TrainingData，其次 CollectedData
        training_path = os.path.join(TRAINING_DATA, args.scene)
        collected_path = os.path.join(COLLECTED_DATA, args.scene)
        
        if os.path.exists(training_path):
            scene_path = training_path
        elif os.path.exists(collected_path):
            scene_path = collected_path
        else:
            print(f"[错误] 场景 '{args.scene}' 不存在")
            list_available_scenes()
            return
    else:
        print("[错误] 请指定场景名称 (--scene) 或路径 (--path)")
        parser.print_help()
        return
    
    print(f"\n使用场景路径: {scene_path}")
    
    # 执行操作
    if args.full_rebuild:
        # 从深度图完整重建
        output_path = scene_path.replace("CollectedData", "TrainingData")
        reconstruct_and_create_tsdf(
            scene_path, output_path, 
            map_name=args.map_name,
            voxel_size=args.voxel_size,
            robot_size=args.robot_size
        )
    elif args.rebuild:
        # 从点云重建
        rebuild_tsdf_from_pointcloud(
            scene_path, 
            map_name=args.map_name,
            voxel_size=args.voxel_size,
            robot_size=args.robot_size
        )
    else:
        # 可视化已有地图
        show_cost = (args.mode == 'cost')
        visualize_existing_tsdf(scene_path, args.map_name, show_cost_map=show_cost)


if __name__ == "__main__":
    main()
