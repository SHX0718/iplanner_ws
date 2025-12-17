# ======================================================================
# RGB2Planner 数据加载器
# 用于端到端训练: 加载 RGB 图像、对应的深度图 (可选)、轨迹和目标点
#
# 数据目录结构:
#   data_root/
#     ├── TrainingData/
#     │   ├── scene_name/
#     │   │   ├── camera/       # RGB 图像 (0.png, 1.png, ...)
#     │   │   ├── depth/        # 深度图像 (0.png, 1.png, ...) - 可选
#     │   │   └── odom.txt      # 里程计数据
#
# Copyright (c) 2024
# ======================================================================

import os
import PIL
import torch
import numpy as np
import pypose as pp
from PIL import Image
from pathlib import Path
from random import sample
from operator import itemgetter
from typing import Tuple, List, Optional, Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

torch.set_default_dtype(torch.float32)


class RGB2PlannerData(Dataset):
    """
    RGB 到路径规划的数据集
    
    加载 RGB 图像和对应的轨迹数据用于端到端训练
    """
    
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 rgb_transform=None,
                 depth_transform=None,
                 sensor_offset_x: float = 0.0,
                 is_robot: bool = False,
                 goal_step: int = 5,
                 max_episode: int = 25,
                 max_depth: float = 10.0,
                 load_depth: bool = False,
                 train_ratio: float = 0.8):
        """
        Args:
            root: 场景数据根目录
            train: 是否为训练集
            rgb_transform: RGB 图像变换
            depth_transform: 深度图变换 (可选)
            sensor_offset_x: 传感器 X 轴偏移
            is_robot: 是否为机器人视角 (需要旋转图像)
            goal_step: 目标点采样步长
            max_episode: 最大回合长度
            max_depth: 最大深度值
            load_depth: 是否加载深度图 (用于深度监督)
            train_ratio: 训练集比例
        """
        super().__init__()
        
        self.root = root
        self.is_robot = is_robot
        self.max_depth = max_depth
        self.load_depth = load_depth
        self.sensor_offset_x = sensor_offset_x
        
        # 设置变换
        self.rgb_transform = rgb_transform or transforms.Compose([
            transforms.Resize((384, 512)),  # ZoeDepth 标准输入尺寸
            transforms.ToTensor(),
        ])
        
        self.depth_transform = depth_transform or transforms.Compose([
            transforms.Resize((360, 640)),  # iPlanner 标准尺寸
            transforms.ToTensor(),
        ])
        
        # 图像和里程计路径
        rgb_path = os.path.join(root, 'camera')
        depth_path = os.path.join(root, 'depth')
        odom_path = os.path.join(root, 'odom.txt')
        
        # 加载里程计数据
        odom_list = self._load_odom(odom_path)
        
        # 获取图像文件列表
        rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
        
        # 构建数据索引
        self.rgb_filenames = []
        self.depth_filenames = []
        self.odom_list = []
        self.goal_list = []
        
        N = len(odom_list)
        
        for ahead in range(1, max_episode + 1, goal_step):
            for i in range(N):
                odom = odom_list[i]
                goal_idx = min(i + ahead, N - 1)
                goal = odom_list[goal_idx]
                goal_relative = pp.Inv(odom) @ goal
                
                # 添加数据
                rgb_file = os.path.join(rgb_path, rgb_files[i])
                self.rgb_filenames.append(rgb_file)
                
                if self.load_depth and os.path.exists(depth_path):
                    depth_file = os.path.join(depth_path, rgb_files[i])
                    self.depth_filenames.append(depth_file)
                else:
                    self.depth_filenames.append(None)
                
                self.odom_list.append(odom.tensor())
                self.goal_list.append(goal_relative.tensor())
        
        # 训练/验证分割
        N = len(self.odom_list)
        indexfile = os.path.join(root, 'split_rgb.pt')
        
        if os.path.exists(indexfile):
            train_index, test_index = torch.load(indexfile)
            if len(train_index) + len(test_index) != N:
                print(f"[数据集] 数据量变化，重新生成分割")
                train_index, test_index = self._create_split(N, train_ratio, indexfile)
        else:
            train_index, test_index = self._create_split(N, train_ratio, indexfile)
        
        # 选择训练或测试数据
        if train:
            indices = train_index
        else:
            indices = test_index
        
        self.rgb_filenames = [self.rgb_filenames[i] for i in indices]
        self.depth_filenames = [self.depth_filenames[i] for i in indices]
        self.odom_list = [self.odom_list[i] for i in indices]
        self.goal_list = [self.goal_list[i] for i in indices]
        
        print(f"[数据集] 场景: {os.path.basename(root)}")
        print(f"  - 模式: {'训练' if train else '验证'}")
        print(f"  - 样本数: {len(self.rgb_filenames)}")
        print(f"  - 加载深度图: {load_depth}")
    
    def _load_odom(self, path: str) -> List:
        """加载里程计数据"""
        odom_list = []
        with open(path, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) >= 7:
                    # x, y, z, qx, qy, qz, qw
                    pose = pp.SE3(torch.tensor(values[:7]))
                    odom_list.append(pose)
        return odom_list
    
    def _create_split(self, n: int, ratio: float, path: str) -> Tuple[List, List]:
        """创建训练/验证分割"""
        indices = list(range(n))
        train_idx = sample(indices, int(ratio * n))
        test_idx = [i for i in indices if i not in train_idx]
        torch.save((train_idx, test_idx), path)
        return train_idx, test_idx
    
    def __len__(self) -> int:
        return len(self.rgb_filenames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Returns:
            dict: {
                'rgb': RGB 图像 (3, H, W),
                'depth': 深度图 (1, H, W) - 可选,
                'odom': 里程计 (7,),
                'goal': 目标点 (7,)
            }
        """
        # 加载 RGB 图像
        rgb_image = Image.open(self.rgb_filenames[idx]).convert('RGB')
        
        if self.is_robot:
            rgb_image = rgb_image.transpose(PIL.Image.ROTATE_180)
        
        rgb_tensor = self.rgb_transform(rgb_image)
        
        result = {
            'rgb': rgb_tensor,
            'odom': self.odom_list[idx],
            'goal': self.goal_list[idx]
        }
        
        # 可选: 加载深度图
        if self.load_depth and self.depth_filenames[idx] is not None:
            depth_image = Image.open(self.depth_filenames[idx])
            
            if self.is_robot:
                depth_image = depth_image.transpose(PIL.Image.ROTATE_180)
            
            depth_array = np.array(depth_image).astype(np.float32)
            depth_array[~np.isfinite(depth_array)] = 0.0
            depth_array = depth_array / 1000.0  # mm -> m
            depth_array[depth_array > self.max_depth] = 0.0
            
            depth_pil = Image.fromarray(depth_array)
            depth_tensor = self.depth_transform(depth_pil)
            
            result['depth'] = depth_tensor
        
        return result


class MultiSceneDataLoader:
    """
    多场景数据加载器
    
    支持从多个场景加载数据，用于多环境训练
    """
    
    def __init__(self,
                 data_root: str,
                 scene_list: List[str],
                 batch_size: int = 8,
                 train: bool = True,
                 load_depth: bool = False,
                 num_workers: int = 4,
                 **kwargs):
        """
        Args:
            data_root: 数据根目录
            scene_list: 场景名称列表
            batch_size: 批大小
            train: 是否为训练模式
            load_depth: 是否加载深度图
            num_workers: 数据加载线程数
            **kwargs: 传递给 RGB2PlannerData 的其他参数
        """
        self.train = train
        self.dataloaders = []
        self.scene_names = []
        
        for scene_name in scene_list:
            scene_path = os.path.join(data_root, 'TrainingData', scene_name)
            
            if not os.path.exists(scene_path):
                print(f"[警告] 场景不存在: {scene_path}")
                continue
            
            dataset = RGB2PlannerData(
                root=scene_path,
                train=train,
                load_depth=load_depth,
                **kwargs
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train,
                num_workers=num_workers,
                pin_memory=True
            )
            
            self.dataloaders.append(dataloader)
            self.scene_names.append(scene_name)
        
        print(f"\n[MultiSceneDataLoader] 加载 {len(self.dataloaders)} 个场景")
        for name, loader in zip(self.scene_names, self.dataloaders):
            print(f"  - {name}: {len(loader.dataset)} 样本")
    
    def __iter__(self):
        """迭代所有场景的数据"""
        for scene_name, loader in zip(self.scene_names, self.dataloaders):
            for batch in loader:
                batch['scene'] = scene_name
                yield batch
    
    def __len__(self) -> int:
        return sum(len(loader) for loader in self.dataloaders)
    
    def get_scene_loaders(self) -> List[Tuple[str, DataLoader]]:
        """获取场景名称和对应的数据加载器"""
        return list(zip(self.scene_names, self.dataloaders))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义批处理函数
    """
    result = {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'odom': torch.stack([item['odom'] for item in batch]),
        'goal': torch.stack([item['goal'] for item in batch]),
    }
    
    if 'depth' in batch[0]:
        result['depth'] = torch.stack([item['depth'] for item in batch])
    
    return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 RGB2PlannerData 数据加载器")
    print("=" * 60)
    
    # 测试路径
    data_root = "/home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner/data"
    scene_path = os.path.join(data_root, "CollectedData", "forest")
    
    if os.path.exists(scene_path):
        # 创建数据集
        dataset = RGB2PlannerData(
            root=scene_path,
            train=True,
            load_depth=False,
            goal_step=5,
            max_episode=25
        )
        
        print(f"\n数据集大小: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"\n样本内容:")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Odom shape: {sample['odom'].shape}")
        print(f"  Goal shape: {sample['goal'].shape}")
        
        if 'depth' in sample:
            print(f"  Depth shape: {sample['depth'].shape}")
        
        # 测试 DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        batch = next(iter(loader))
        print(f"\n批次内容:")
        print(f"  RGB batch: {batch['rgb'].shape}")
        print(f"  Odom batch: {batch['odom'].shape}")
        print(f"  Goal batch: {batch['goal'].shape}")
        
        print("\n测试完成!")
    else:
        print(f"测试数据目录不存在: {scene_path}")
        print("请确保已收集训练数据")
