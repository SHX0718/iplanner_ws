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
import numpy as np

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
    
    def plan(self, image, goal_robot_frame):
        # 直接使用深度图输入（不使用 ZoeDepth 转换）
        img_to_process = image
        
        # 如果输入是RGB/3通道图像，取第一个通道或转为灰度
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            # 转换为灰度图
            img_to_process = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        
        # 确保img_to_process是单通道
        if isinstance(img_to_process, np.ndarray):
            if len(img_to_process.shape) == 3:
                # 如果仍然是多通道，取第一个通道
                print(f"[警告] 输入仍为多通道 {img_to_process.shape}，仅使用第一个通道")
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
