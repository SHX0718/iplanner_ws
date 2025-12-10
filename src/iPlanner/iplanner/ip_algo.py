# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import PIL
import math
import torch
import torchvision.transforms as transforms
import sys
import os
import numpy as np
import time

from iplanner import traj_opt

class IPlannerAlgo:
    def __init__(self, args):
        super(IPlannerAlgo, self).__init__()
        self.config(args)

        self.depth_transform = transforms.Compose([
            transforms.Resize(tuple(self.crop_size)),
            transforms.ToTensor()])

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = traj_opt.TrajOpt()
        
        # ZoeDepth 转换器（延迟初始化，仅在需要时初始化）
        self.zoe_converter = None
        self.use_zoe_depth = None  # None 表示尚未尝试初始化，False 表示初始化失败，True 表示初始化成功
        return None

    def config(self, args):
        self.model_save = args.model_save
        self.crop_size  = args.crop_size
        self.sensor_offset_x = args.sensor_offset_x
        self.sensor_offset_y = args.sensor_offset_y
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
        return None


    def _init_zoe_converter(self):
        """懒轭初始化 ZoeDepth 转换器"""
        if self.zoe_converter is None:
            try:
                print("[INFO] 开始初始化 ZoeDepth 转换器...")
                # 确保 ZoeDepth 路径在 sys.path 中
                _current_file = os.path.abspath(__file__)
                _iplanner_dir = os.path.dirname(_current_file)
                _src_dir = os.path.dirname(_iplanner_dir)
                _workspace_root = os.path.dirname(_src_dir)
                _zoedepth_path = os.path.join(_workspace_root, 'ZoeDepth')
                if _zoedepth_path not in sys.path:
                    sys.path.insert(0, _zoedepth_path)
                    print(f"[INFO] 添加 ZoeDepth 路径到 sys.path: {_zoedepth_path}")
                
                from iplanner.zoe_depth_wrapper import ZoeDepthConverter
                print("[INFO] 成功导入 ZoeDepthConverter")
                self.zoe_converter = ZoeDepthConverter()  # 使用默认配置：zoedepth_nk + CPU
                self.use_zoe_depth = True
                print("[INFO] ZoeDepth 转换器初始化成功")
            except Exception as e:
                print(f"[ERROR] 无法初始化 ZoeDepth: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                print("[WARNING] 将使用灰度转换模式")
                self.use_zoe_depth = False
    
    def plan(self, image, goal_robot_frame):
        # 懒轭初始化 ZoeDepth（第一次平面推理时）
        if self.use_zoe_depth is None or not hasattr(self, 'use_zoe_depth'):
            self._init_zoe_converter()
        
        # 确定是否为RGB图像
        is_rgb_image = isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3
        
        # 如果输入是RGB图像，尝试用ZoeDepth转换为深度图
        if is_rgb_image:
            if self.use_zoe_depth:
                # RGB输入且ZoeDepth可用：用ZoeDepth转换为深度图
                print(f"[INFO] 检测到RGB输入 {image.shape}，使用ZoeDepth转换")
                try:
                    print("[DEBUG] 准备转换RGB图像为PIL格式...")
                    rgb_pil = PIL.Image.fromarray(image.astype(np.uint8)).convert('RGB')
                    print(f"[DEBUG] PIL图像尺寸: {rgb_pil.size}")
                    
                    # 开始计时
                    print("[DEBUG] 开始调用 ZoeDepth 推理...")
                    start_time = time.time()
                    depth_array = self.zoe_converter.rgb_to_depth(rgb_pil)
                    zoe_time = time.time() - start_time
                    print("[DEBUG] ZoeDepth 推理完成")
                    
                    # 深度图转为numpy array（如果不是的话）
                    if isinstance(depth_array, torch.Tensor):
                        depth_array = depth_array.cpu().numpy()
                    img_to_process = depth_array
                    print(f"[INFO] ZoeDepth转换成功，深度图形状: {img_to_process.shape}")
                    print(f"[TIMING] ZoeDepth推理耗时: {zoe_time:.3f} 秒 ({1.0/zoe_time:.2f} FPS)")
                except Exception as e:
                    print(f"[WARNING] ZoeDepth转换失败: {e}，降级为灰度处理")
                    # 转换为灰度图
                    img_to_process = np.dot(image[...,:3], [0.299, 0.587, 0.114])
            else:
                # RGB输入但ZoeDepth不可用：转换为灰度图
                print(f"[WARNING] 检测到RGB输入但ZoeDepth不可用，转换为灰度")
                img_to_process = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:
            # 非RGB输入：直接使用（假设为灰度或深度图）
            img_to_process = image
        
        # 确保img_to_process是单通道
        if isinstance(img_to_process, np.ndarray):
            if len(img_to_process.shape) == 3:
                # 如果仍然是多通道，取第一个通道
                print(f"[WARNING] 输入仍为多通道 {img_to_process.shape}，仅使用第一个通道")
                img_to_process = img_to_process[:, :, 0]
        
        # 转换为PIL图像
        if img_to_process.dtype == np.float32 or img_to_process.dtype == np.float64:
            # 浮点数深度图：归一化到0-255
            if img_to_process.max() <= 1.0:
                img_for_pil = (img_to_process * 255).astype(np.uint8)
            else:
                # 假设是实际深度值，归一化
                valid_mask = img_to_process > 0
                if valid_mask.any():
                    img_normalized = np.zeros_like(img_to_process)
                    img_normalized[valid_mask] = (img_to_process[valid_mask] - img_to_process[valid_mask].min()) / (img_to_process[valid_mask].max() - img_to_process[valid_mask].min())
                    img_for_pil = (img_normalized * 255).astype(np.uint8)
                else:
                    img_for_pil = np.zeros_like(img_to_process, dtype=np.uint8)
        else:
            # 整型图像
            img_for_pil = img_to_process.astype(np.uint8)
        
        # 转换为PIL Image
        img = PIL.Image.fromarray(img_for_pil, mode='L')  # 'L' 表示灰度图
        
        # 应用变换（resize）
        img_tensor = self.depth_transform(img)  # 转换为单通道tensor
        
        # 将单通道深度图扩展到3通道（复制深度信息）
        img_tensor = img_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, H, W)
        
        print(f"[DEBUG] 最终输入形状: {img_tensor.shape}")
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
        with torch.no_grad():
            keypoints, fear = self.net(img_tensor, goal_robot_frame)
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat((torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), axis=1)
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints , step=0.1)
        
        return keypoints, traj, fear, img_tensor
