#!/usr/bin/env python3
# ======================================================================
# RGB2Planner ROS 可视化节点
# 端到端模式: 直接从 RGB 图像预测路径 (带可视化)
# 
# 使用方法:
#   roslaunch iplanner_node iplanner_viz.launch mode:=rgb config:=vehicle_sim_rgb2planner
#
# Copyright (c) 2024
# ======================================================================

# 复用 iplanner_rgb_node 的所有功能
# 可视化版本主要在 RViz 中显示额外的调试信息

from iplanner_rgb_node import *

if __name__ == '__main__':
    node_name = "iplanner_rgb_viz"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    
    # 基本参数
    parser.add_argument('main_freq', type=int, default=5, help="主循环频率")
    parser.add_argument('model_save', type=str, default='/models/rgb2planner.pt', help="模型路径")
    parser.add_argument('crop_size', type=tuple, default=[384, 512], help='裁剪尺寸')
    
    # RGB 话题 (端到端模式)
    parser.add_argument('rgb_topic', type=str, default='/rgbd_camera/color/image', help='RGB 话题')
    parser.add_argument('depth_topic', type=str, default='/rgbd_camera/depth/image', help='深度话题 (备用)')
    
    # 其他话题
    parser.add_argument('goal_topic', type=str, default='/way_point', help='目标点话题')
    parser.add_argument('path_topic', type=str, default='/path', help='路径话题')
    parser.add_argument('image_topic', type=str, default='/iplanner_image', help='可视化图像话题')
    
    # 坐标系
    parser.add_argument('robot_id', type=str, default='base', help='机器人坐标系')
    parser.add_argument('world_id', type=str, default='odom', help='世界坐标系')
    
    # 图像处理
    parser.add_argument('image_flip', type=bool, default=False, help='是否翻转图像')
    parser.add_argument('conv_dist', type=float, default=0.5, help='目标收敛距离')
    
    # Fear 反应
    parser.add_argument('is_fear_act', type=bool, default=True, help='启用 fear 反应')
    parser.add_argument('buffer_size', type=int, default=3, help='Fear 缓冲区大小')
    parser.add_argument('angular_thred', type=float, default=0.5, help='角度阈值')
    parser.add_argument('track_dist', type=float, default=0.5, help='跟踪距离')
    parser.add_argument('joyGoal_scale', type=float, default=5.0, help='摇杆目标缩放')
    
    # 传感器偏移
    parser.add_argument('sensor_offset_x', type=float, default=0.0, help='传感器 X 偏移')
    parser.add_argument('sensor_offset_y', type=float, default=0.0, help='传感器 Y 偏移')
    
    # RGB2Planner 特有参数
    parser.add_argument('zoe_model_name', type=str, default='zoedepth_nk', help='ZoeDepth 模型名称')

    args = parser.parse_args()
    args.model_save = planner_path + args.model_save

    rospy.loginfo("[RGB2Planner Viz] 可视化模式启动")
    
    node = RGB2PlannerNode(args)
    node.spin()
