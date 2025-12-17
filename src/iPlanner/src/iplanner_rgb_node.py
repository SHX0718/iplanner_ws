#!/usr/bin/env python3
# ======================================================================
# RGB2Planner ROS 节点
# 端到端模式: 直接从 RGB 图像预测路径
# 
# 使用方法:
#   roslaunch iplanner_node iplanner.launch mode:=rgb config:=vehicle_sim_rgb2planner
#
# Copyright (c) 2024
# ======================================================================

import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
import time
from std_msgs.msg import Float32, Int16
import numpy as np
from sensor_msgs.msg import Image, Joy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy

rospack = rospkg.RosPack()
pack_path = rospack.get_path('iplanner_node')
planner_path = os.path.join(pack_path, 'iplanner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from iplanner.rgb2planner_net import RGB2PlannerNet
from iplanner.traj_opt import TrajOpt
from iplanner.rosutil import ROSArgparse
import torchvision.transforms as transforms


class RGB2PlannerAlgo:
    """RGB2Planner 算法封装"""
    
    def __init__(self, args):
        self.config(args)
        
        # 图像变换
        self.rgb_transform = transforms.Compose([
            transforms.Resize(tuple(self.crop_size)),
            transforms.ToTensor(),
        ])
        
        # 加载模型
        rospy.loginfo(f"[RGB2Planner] 加载模型: {self.model_save}")
        
        # 检查是否有训练好的模型
        if os.path.exists(self.model_save):
            checkpoint = torch.load(self.model_save, map_location='cpu')
            self.net = RGB2PlannerNet(
                k=5,
                freeze_zoe_encoder=True,
                pretrained_zoe=True,
                zoe_model_name=getattr(args, 'zoe_model_name', 'zoedepth_nk')
            )
            self.net.load_state_dict(checkpoint['model_state_dict'])
            rospy.loginfo("[RGB2Planner] 加载训练权重成功")
        else:
            # 使用预训练模型
            rospy.logwarn(f"[RGB2Planner] 未找到训练权重: {self.model_save}")
            rospy.logwarn("[RGB2Planner] 使用预训练 ZoeDepth 初始化")
            self.net = RGB2PlannerNet(
                k=5,
                freeze_zoe_encoder=True,
                pretrained_zoe=True,
                zoe_model_name=getattr(args, 'zoe_model_name', 'zoedepth_nk')
            )
        
        # 移动到设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = self.net.to(self.device)
        self.net.eval()
        
        rospy.loginfo(f"[RGB2Planner] 设备: {self.device}")
        
        # 轨迹生成器
        self.traj_generate = TrajOpt()
        
    def config(self, args):
        self.model_save = args.model_save
        self.crop_size = args.crop_size
        self.is_traj_shift = True
        self.sensor_offset_x = getattr(args, 'sensor_offset_x', 0.0)
        self.sensor_offset_y = getattr(args, 'sensor_offset_y', 0.0)
        
    def plan(self, rgb_image, goal_robot_frame):
        """
        从 RGB 图像规划路径
        
        Args:
            rgb_image: RGB 图像 (H, W, 3) numpy array
            goal_robot_frame: 目标点 (1, 3) tensor
        
        Returns:
            keypoints: 关键点
            traj: 轨迹
            fear: 恐惧置信度
            rgb_tensor: 处理后的 RGB tensor
        """
        # 预处理 RGB 图像
        if isinstance(rgb_image, np.ndarray):
            # 确保是 uint8
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)
            rgb_pil = PIL.Image.fromarray(rgb_image).convert('RGB')
        else:
            rgb_pil = rgb_image
        
        # 应用变换
        rgb_tensor = self.rgb_transform(rgb_pil).unsqueeze(0)  # (1, 3, H, W)
        
        if self.device == "cuda":
            rgb_tensor = rgb_tensor.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
        
        # 推理
        with torch.no_grad():
            output = self.net(rgb_tensor, goal_robot_frame)
        
        keypoints = output['keypoints']
        fear = output['fear']
        
        # 轨迹偏移
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat((
                torch.zeros(batch_size, 1, dims, device=keypoints.device),
                keypoints
            ), axis=1)
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        
        # 生成轨迹
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        
        return keypoints, traj, fear, rgb_tensor


class RGB2PlannerNode:
    """RGB2Planner ROS 节点"""
    
    def __init__(self, args):
        super(RGB2PlannerNode, self).__init__()
        self.config(args)

        # 初始化算法
        self.rgb2planner_algo = RGB2PlannerAlgo(args=args)
        self.tf_listener = tf.TransformListener()
        
        rospy.sleep(2.5)  # 等待 tf listener 就绪

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False

        # 规划状态
        self.planner_status = Int16()
        self.planner_status.data = 0
        self.is_goal_processed = False
        self.is_smartjoy = False

        # fear 反应
        self.fear_buffter = 0
        self.is_fear_reaction = False
        
        # 处理时间
        self.timer_data = Float32()
        
        # 订阅 RGB 话题
        rospy.Subscriber(self.rgb_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        timer_topic = '/ip_timer'
        status_topic = '/ip_planner_status'
        
        # 发布话题
        self.timer_pub = rospy.Publisher(timer_topic, Float32, queue_size=10)
        self.status_pub = rospy.Publisher(status_topic, Int16, queue_size=10)
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", Path, queue_size=10)
        self.image_pub = rospy.Publisher(args.image_topic, Image, queue_size=10)

        rospy.loginfo("[RGB2Planner] 节点就绪 (端到端RGB模式)")
        

    def config(self, args):
        self.main_freq = args.main_freq
        self.model_save = args.model_save
        
        # RGB 话题 (端到端模式)
        self.rgb_topic = getattr(args, 'rgb_topic', '/rgbd_camera/color/image')
        
        self.image_pub_topic = args.image_topic
        self.goal_topic = args.goal_topic
        self.path_topic = args.path_topic
        self.frame_id = args.robot_id
        self.world_id = args.world_id
        self.image_flip = args.image_flip
        self.conv_dist = args.conv_dist
        
        # fear 反应
        self.is_fear_act = args.is_fear_act
        self.buffer_size = args.buffer_size
        self.ang_thred = args.angular_thred
        self.track_dist = args.track_dist
        self.joyGoal_scale = args.joyGoal_scale
        
        rospy.loginfo(f"[RGB2Planner] 订阅 RGB 话题: {self.rgb_topic}")

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning and self.is_goal_init:
                cur_image = self.img.copy()
                start = time.time()
                
                # 网络规划
                self.preds, self.waypoints, fear_output, _ = self.rgb2planner_algo.plan(
                    cur_image, self.goal_rb
                )
                
                end = time.time()
                self.timer_data.data = (end - start) * 1000
                self.timer_pub.publish(self.timer_data)
                
                # 检查是否到达目标
                goal_dist = np.sqrt(self.goal_rb[0][0]**2 + self.goal_rb[0][1]**2)
                if goal_dist < self.conv_dist and self.is_goal_processed and not self.is_smartjoy:
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)
                    rospy.loginfo("Goal Arrived")
                
                # Fear 反应
                self.fear = torch.tensor([[0.0]], device=fear_output.device)
                if self.is_fear_act:
                    self.fear = fear_output
                    is_track_ahead = self.isForwardTraking(self.waypoints)
                    self.fearPathDetection(self.fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "当前路径预测无效")
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)
                
                self.pubPath(self.waypoints, self.is_goal_init)
                self.pubImage()
            r.sleep()
        rospy.spin()

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        fear_path = Path()
        if is_goal_init:
            for p in waypoints.squeeze(0):
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path.poses.append(pose)
        
        path.header.frame_id = fear_path.header.frame_id = self.frame_id
        path.header.stamp = fear_path.header.stamp = self.image_time
        
        if self.is_fear_reaction:
            fear_path.poses = path.poses.copy()
            path.poses = path.poses[:1]
        
        self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)

    def pubImage(self):
        """发布处理后的图像用于可视化"""
        try:
            img_vis = self.img.copy()
            if img_vis.dtype != np.uint8:
                if img_vis.max() <= 1.0:
                    img_vis = (img_vis * 255).astype(np.uint8)
                else:
                    img_vis = img_vis.astype(np.uint8)
            
            img_msg = ros_numpy.msgify(Image, img_vis, encoding='rgb8')
            img_msg.header.stamp = self.image_time
            img_msg.header.frame_id = self.frame_id
            
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr(f"发布图像错误: {e}")

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter += 1
        elif self.is_fear_reaction:
            self.fear_buffter -= 1
        
        if self.fear_buffter > self.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if phead is None or phead.dot(xhead) > 1.0 - self.ang_thred:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            rospy.loginfo("切换到 Smart Joystick 模式...")
            self.is_smartjoy = True
            self.fear_buffter = 0
            self.is_fear_reaction = False
        
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3]**2 + joy_msg.axes[4]**2) < 1e-3:
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.ready_for_planning = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.frame_id
                joy_goal.point.x = joy_msg.axes[4] * self.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.joyGoal_scale
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = rospy.Time.now()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False

    def goalCallback(self, msg):
        rospy.loginfo("收到新目标点")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        self.fear_buffter = 0
        self.is_fear_reaction = False
        self.planner_status.data = 0

    def imageCallback(self, msg):
        """RGB 图像回调"""
        self.image_time = msg.header.stamp
        
        # 解析 RGB 图像
        frame = ros_numpy.numpify(msg)
        
        # 处理图像翻转
        if self.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.img = np.array(frame.transpose(PIL.Image.ROTATE_180))
        else:
            self.img = frame

        if self.is_goal_init:
            goal_robot_frame = self.goal_pose
            if not self.goal_pose.header.frame_id == self.frame_id:
                try:
                    goal_robot_frame.header.stamp = self.tf_listener.getLatestCommonTime(
                        self.goal_pose.header.frame_id, self.frame_id
                    )
                    goal_robot_frame = self.tf_listener.transformPoint(self.frame_id, goal_robot_frame)
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("无法将目标点转换到机器人坐标系")
                    return
            
            goal_tensor = torch.tensor([
                goal_robot_frame.point.x,
                goal_robot_frame.point.y,
                goal_robot_frame.point.z
            ], dtype=torch.float32)[None, ...]
            
            self.goal_rb = goal_tensor
        else:
            return
        
        self.ready_for_planning = True
        self.is_goal_processed = True


if __name__ == '__main__':
    node_name = "iplanner_rgb_node"
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

    node = RGB2PlannerNode(args)
    node.spin()
