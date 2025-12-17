# ======================================================================
# RGB2Planner 端到端训练脚本
# 
# 训练流程:
#   1. Stage 1: 冻结 ZoeDepth 编码器，只训练适配层和规划解码器
#   2. Stage 2 (可选): 解冻编码器，小学习率端到端微调
#
# 使用方法:
#   python train_rgb2planner.py --config config/rgb2planner_config.json
#
# Copyright (c) 2024
# ======================================================================

import os
import sys
import json
import time
import argparse
import tqdm
import random
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)

from rgb2planner_net import RGB2PlannerNet, create_rgb2planner_net
from rgb2planner_dataloader import RGB2PlannerData, MultiSceneDataLoader, collate_fn
from traj_cost import TrajCost
from traj_opt import TrajOpt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[警告] wandb 未安装，跳过实验追踪")


class RGB2PlannerTrainer:
    """RGB2Planner 端到端训练器"""
    
    def __init__(self, config_path: str = None, config: Dict = None):
        """
        Args:
            config_path: 配置文件路径
            config: 配置字典 (优先级高于配置文件)
        """
        # 加载配置
        self.config = self._load_config(config_path, config)
        
        # 设置设备
        self.device = self._setup_device()
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化优化器和调度器
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # 初始化数据加载器
        self.train_loaders, self.val_loaders = self._create_dataloaders()
        
        # 初始化轨迹代价计算器
        self.traj_costs = self._create_traj_costs()
        
        # 轨迹优化器
        self.traj_opt = TrajOpt()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_stage = 1  # 1: 冻结编码器, 2: 端到端微调
        
        # 初始化 wandb
        if self.config.get('use_wandb', False) and HAS_WANDB:
            self._init_wandb()
        
        print("\n" + "=" * 60)
        print("RGB2Planner 训练器初始化完成")
        print("=" * 60)
        self._print_config()
    
    def _load_config(self, config_path: str, config: Dict) -> Dict:
        """加载配置"""
        default_config = {
            # 模型配置
            'k': 5,
            'goal_channels': 64,
            'zoe_model_name': 'zoedepth_nk',
            'pretrained_zoe': True,
            'pretrained_planner': None,
            'freeze_zoe_encoder': True,
            
            # 数据配置
            'data_root': 'data',
            'scene_list': 'training_list.txt',
            'crop_size': [384, 512],
            'goal_step': 5,
            'max_episode': 25,
            'max_depth': 10.0,
            'load_depth': False,
            
            # 训练配置
            'batch_size': 4,
            'epochs': 50,
            'lr': 1e-4,
            'lr_encoder': 1e-5,
            'weight_decay': 1e-4,
            'patience': 5,
            'min_lr': 1e-6,
            'lr_factor': 0.5,
            
            # 损失权重
            'alpha': 0.5,   # 障碍物损失
            'beta': 1.0,    # 地形高度损失
            'gamma': 2.0,   # 运动损失
            'delta': 5.0,   # 目标损失
            'fear_ahead_dist': 2.0,
            
            # 阶段训练
            'stage1_epochs': 30,
            'stage2_epochs': 20,
            'enable_stage2': True,
            
            # 其他
            'gpu_id': 0,
            'num_workers': 4,
            'save_dir': 'models',
            'use_wandb': False,
            'log_interval': 10,
        }
        
        # 从文件加载
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
        
        # 从参数覆盖
        if config:
            default_config.update(config)
        
        return default_config
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['gpu_id']}")
            print(f"[设备] 使用 GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("[设备] 使用 CPU (训练会很慢)")
        return device
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        model = RGB2PlannerNet(
            k=self.config['k'],
            goal_channels=self.config['goal_channels'],
            freeze_zoe_encoder=self.config['freeze_zoe_encoder'],
            pretrained_zoe=self.config['pretrained_zoe'],
            pretrained_planner=self.config.get('pretrained_planner'),
            zoe_model_name=self.config['zoe_model_name'],
            output_intermediate_depth=self.config['load_depth']
        )
        return model.to(self.device)
    
    def _create_optimizer(self):
        """创建优化器和调度器"""
        param_groups = self.model.get_param_groups(self.config['lr'])
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['patience'],
            min_lr=self.config['min_lr'],
            verbose=True
        )
        
        return optimizer, scheduler
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        data_root = self.config['data_root']
        
        # 加载场景列表
        scene_list_path = os.path.join(data_root, self.config['scene_list'])
        if os.path.exists(scene_list_path):
            with open(scene_list_path, 'r') as f:
                scene_list = [line.strip() for line in f if line.strip()]
        else:
            # 自动发现场景
            training_data_dir = os.path.join(data_root, 'TrainingData')
            if os.path.exists(training_data_dir):
                scene_list = [d for d in os.listdir(training_data_dir) 
                             if os.path.isdir(os.path.join(training_data_dir, d))]
            else:
                scene_list = []
        
        if not scene_list:
            print("[警告] 未找到训练场景")
            return [], []
        
        print(f"[数据] 加载 {len(scene_list)} 个场景: {scene_list}")
        
        # 创建数据加载器
        train_loaders = []
        val_loaders = []
        
        for scene_name in scene_list:
            scene_path = os.path.join(data_root, 'TrainingData', scene_name)
            
            if not os.path.exists(scene_path):
                print(f"[警告] 场景不存在: {scene_path}")
                continue
            
            # 训练集
            train_dataset = RGB2PlannerData(
                root=scene_path,
                train=True,
                load_depth=self.config['load_depth'],
                goal_step=self.config['goal_step'],
                max_episode=self.config['max_episode'],
                max_depth=self.config['max_depth']
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                collate_fn=collate_fn,
                pin_memory=True
            )
            train_loaders.append((scene_name, train_loader))
            
            # 验证集
            val_dataset = RGB2PlannerData(
                root=scene_path,
                train=False,
                load_depth=self.config['load_depth'],
                goal_step=self.config['goal_step'],
                max_episode=self.config['max_episode'],
                max_depth=self.config['max_depth']
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                collate_fn=collate_fn,
                pin_memory=True
            )
            val_loaders.append((scene_name, val_loader))
        
        return train_loaders, val_loaders
    
    def _create_traj_costs(self) -> Dict[str, TrajCost]:
        """为每个场景创建轨迹代价计算器"""
        traj_costs = {}
        
        for scene_name, _ in self.train_loaders:
            traj_cost = TrajCost(self.config['gpu_id'])
            
            # 尝试加载地图
            map_path = os.path.join(self.config['data_root'], 'TrainingData', scene_name)
            try:
                traj_cost.SetMap(map_path, scene_name)
                print(f"[地图] {scene_name}: 加载成功")
            except Exception as e:
                print(f"[地图] {scene_name}: 加载失败 - {e}")
            
            traj_costs[scene_name] = traj_cost
        
        return traj_costs
    
    def _init_wandb(self):
        """初始化 wandb"""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project="rgb2planner",
            name=f"train_{date_str}",
            config=self.config
        )
    
    def _print_config(self):
        """打印配置"""
        print("\n配置:")
        print(f"  模型: ZoeDepth ({self.config['zoe_model_name']}) + PlannerDecoder")
        print(f"  关键点数: {self.config['k']}")
        print(f"  训练阶段: Stage1={self.config['stage1_epochs']}epochs, Stage2={self.config['stage2_epochs']}epochs")
        print(f"  学习率: {self.config['lr']} (编码器: {self.config['lr_encoder']})")
        print(f"  批大小: {self.config['batch_size']}")
        print(f"  设备: {self.device}")
    
    def compute_loss(self, keypoints: torch.Tensor, fear: torch.Tensor,
                     odom: torch.Tensor, goal: torch.Tensor,
                     traj_cost: TrajCost) -> tuple:
        """
        计算损失
        
        Args:
            keypoints: 预测关键点 (B, k, 3)
            fear: 预测 fear (B, 1)
            odom: 里程计 (B, 7)
            goal: 目标点 (B, 7)
            traj_cost: 轨迹代价计算器
        
        Returns:
            loss: 总损失
            fear_labels: fear 标签
        """
        # 生成轨迹
        waypoints = self.traj_opt.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        
        # 计算轨迹代价
        loss, fear_labels = traj_cost.CostofTraj(
            waypoints, odom, goal,
            ahead_dist=self.config['fear_ahead_dist'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            gamma=self.config['gamma'],
            delta=self.config['delta']
        )
        
        # Fear 损失
        fear_loss = F.binary_cross_entropy(fear, fear_labels)
        
        total_loss = loss + fear_loss
        
        return total_loss, fear_labels
    
    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_batches = 0
        
        # 打乱场景顺序
        scene_loaders = list(self.train_loaders)
        random.shuffle(scene_loaders)
        
        for scene_name, loader in scene_loaders:
            traj_cost = self.traj_costs.get(scene_name)
            if traj_cost is None:
                continue
            
            pbar = tqdm.tqdm(loader, desc=f"  {scene_name}")
            scene_loss = 0.0
            
            for batch_idx, batch in enumerate(pbar):
                # 移动到设备
                rgb = batch['rgb'].to(self.device)
                odom = batch['odom'].to(self.device)
                goal = batch['goal'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(rgb, goal)
                
                keypoints = output['keypoints']
                fear = output['fear']
                
                # 计算损失
                loss, _ = self.compute_loss(keypoints, fear, odom, goal, traj_cost)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                scene_loss += loss.item()
                total_batches += 1
                
                pbar.set_postfix({'loss': f"{scene_loss / (batch_idx + 1):.4f}"})
            
            total_loss += scene_loss
        
        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_batches = 0
        
        for scene_name, loader in self.val_loaders:
            traj_cost = self.traj_costs.get(scene_name)
            if traj_cost is None:
                continue
            
            for batch in loader:
                rgb = batch['rgb'].to(self.device)
                odom = batch['odom'].to(self.device)
                goal = batch['goal'].to(self.device)
                
                output = self.model(rgb, goal)
                
                loss, _ = self.compute_loss(
                    output['keypoints'], output['fear'],
                    odom, goal, traj_cost
                )
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"[保存] 最佳模型: {best_path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            print(f"[加载] 检查点: {path}, epoch={self.current_epoch}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        total_epochs = self.config['stage1_epochs']
        if self.config['enable_stage2']:
            total_epochs += self.config['stage2_epochs']
        
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            
            # 阶段切换
            if epoch == self.config['stage1_epochs'] and self.config['enable_stage2']:
                print("\n" + "=" * 40)
                print("切换到 Stage 2: 端到端微调")
                print("=" * 40)
                self.training_stage = 2
                self.model.unfreeze_encoder()
                # 重新创建优化器
                self.optimizer, self.scheduler = self._create_optimizer()
            
            print(f"\nEpoch {epoch + 1}/{total_epochs} (Stage {self.training_stage})")
            
            # 训练
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 日志
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {train_time:.1f}s")
            
            if HAS_WANDB and self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'stage': self.training_stage
                })
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            checkpoint_path = os.path.join(save_dir, f'rgb2planner_epoch{epoch + 1}.pt')
            self.save_checkpoint(checkpoint_path, is_best)
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")
        print("=" * 60)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RGB2Planner 端到端训练')
    
    parser.add_argument('--config', type=str, default='config/rgb2planner_config.json',
                        help='配置文件路径')
    parser.add_argument('--data-root', type=str, default=None,
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--wandb', action='store_true',
                        help='启用 wandb 日志')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 构建配置覆盖
    config_override = {}
    if args.data_root:
        config_override['data_root'] = args.data_root
    if args.epochs:
        config_override['stage1_epochs'] = args.epochs
        config_override['stage2_epochs'] = 0
        config_override['enable_stage2'] = False
    if args.batch_size:
        config_override['batch_size'] = args.batch_size
    if args.lr:
        config_override['lr'] = args.lr
    if args.gpu is not None:
        config_override['gpu_id'] = args.gpu
    if args.wandb:
        config_override['use_wandb'] = True
    
    # 创建训练器
    trainer = RGB2PlannerTrainer(
        config_path=args.config,
        config=config_override if config_override else None
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
