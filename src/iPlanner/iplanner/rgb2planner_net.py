# ======================================================================
# RGB2PlannerNet: 端到端 RGB 到路径规划网络
# 将 ZoeDepth 深度估计和 iPlanner 路径规划合并为统一网络
# 
# 架构:
#   RGB图像 -> ZoeDepth Encoder -> 特征适配层 -> iPlanner Decoder -> 路径关键点 + fear
#
# Copyright (c) 2024
# ======================================================================

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# 添加 ZoeDepth 路径
_current_file = os.path.abspath(__file__)
_iplanner_dir = os.path.dirname(os.path.dirname(_current_file))
_src_dir = os.path.dirname(_iplanner_dir)
_workspace_root = os.path.dirname(_src_dir)
_zoedepth_path = os.path.join(_workspace_root, 'ZoeDepth')

if _zoedepth_path not in sys.path:
    sys.path.insert(0, _zoedepth_path)


class FeatureAdapter(nn.Module):
    """特征适配层: 将 ZoeDepth 特征转换为 iPlanner Decoder 所需的格式"""
    
    def __init__(self, zoe_channels: int = 256, planner_channels: int = 512, 
                 input_size: Tuple[int, int] = (384, 512), 
                 output_size: Tuple[int, int] = (12, 20)):
        """
        Args:
            zoe_channels: ZoeDepth 输出特征通道数
            planner_channels: iPlanner Decoder 期望的通道数
            input_size: ZoeDepth 特征图大小 (H, W)
            output_size: iPlanner Decoder 期望的特征图大小 (H, W)
        """
        super().__init__()
        
        self.output_size = output_size
        
        # 通道适配: ZoeDepth 特征 -> iPlanner 期望的 512 通道
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(zoe_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, planner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(planner_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ZoeDepth 特征 (B, zoe_channels, H, W)
        Returns:
            适配后的特征 (B, planner_channels, out_H, out_W)
        """
        # 通道适配
        x = self.channel_adapter(x)
        
        # 空间下采样到目标尺寸
        if x.shape[2:] != self.output_size:
            x = F.adaptive_avg_pool2d(x, self.output_size)
        
        return x


class MetricHeadResidualAdapter(nn.Module):
    """
    Metric Head 残差适配层
    将 ZoeDepth NK 双头的 bin_embedding 特征通过残差连接传递到 Planner Decoder
    """
    
    def __init__(self, 
                 bin_embedding_dim: int = 128,
                 planner_channels: int = 512,
                 output_size: Tuple[int, int] = (12, 20),
                 use_both_heads: bool = False):
        """
        Args:
            bin_embedding_dim: ZoeDepth bin embedding 维度 (默认128)
            planner_channels: Planner Decoder 期望的通道数
            output_size: 输出特征图大小
            use_both_heads: 是否同时使用 N 和 K 两个头的特征
        """
        super().__init__()
        
        self.output_size = output_size
        self.use_both_heads = use_both_heads
        
        # 输入通道数: 单头128，双头256
        in_channels = bin_embedding_dim * 2 if use_both_heads else bin_embedding_dim
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, planner_channels, kernel_size=1),
            nn.BatchNorm2d(planner_channels),
        )
        
        # 残差缩放因子 (可学习)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, head_features: torch.Tensor, main_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_features: Metric Head 特征 (B, bin_embedding_dim, H, W)
            main_features: 主干特征 (B, planner_channels, H', W')
        Returns:
            融合后的特征 (B, planner_channels, H', W')
        """
        # 空间对齐
        if head_features.shape[2:] != self.output_size:
            head_features = F.interpolate(
                head_features, size=self.output_size, 
                mode='bilinear', align_corners=True
            )
        
        # 投影到 planner 通道数
        projected = self.projection(head_features)
        
        # 残差连接
        output = main_features + self.residual_scale * projected
        
        return output


class PlannerDecoder(nn.Module):
    """iPlanner 解码器: 从特征预测路径关键点和 fear 置信度"""
    
    def __init__(self, in_channels: int = 512, goal_channels: int = 64, k: int = 5):
        """
        Args:
            in_channels: 输入特征通道数
            goal_channels: 目标编码通道数
            k: 输出关键点数量
        """
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # 目标点编码
        self.fg = nn.Linear(3, goal_channels)
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels + goal_channels, 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        
        # 全连接层 - 路径预测
        self.fc1 = nn.Linear(256 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 3)
        
        # 全连接层 - fear 预测
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 特征图 (B, in_channels, H, W)
            goal: 目标点 (B, 3+) 至少包含 x, y, z 坐标
        Returns:
            keypoints: 路径关键点 (B, k, 3)
            fear: 恐惧置信度 (B, 1)
        """
        # 目标编码并扩展到空间维度
        goal_enc = self.fg(goal[:, 0:3])
        goal_enc = goal_enc[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        
        # 拼接特征和目标编码
        x = torch.cat((x, goal_enc), dim=1)
        
        # 卷积处理
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        
        # 共享特征
        f = self.relu(self.fc1(x))
        
        # 路径预测
        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        keypoints = x.reshape(-1, self.k, 3)
        
        # Fear 预测
        c = self.relu(self.frc1(f))
        fear = self.sigmoid(self.frc2(c))
        
        return keypoints, fear


class RGB2PlannerNet(nn.Module):
    """
    端到端 RGB 到路径规划网络
    
    将 ZoeDepth 的深度估计能力与 iPlanner 的路径规划能力合并,
    实现从单张 RGB 图像直接预测可行走路径。
    
    架构:
        RGB -> ZoeDepth Encoder -> Feature Adapter -> Planner Decoder -> Path + Fear
                                        ↑
                               Metric Head Features (残差连接)
    """
    
    def __init__(self, 
                 k: int = 5,
                 goal_channels: int = 64,
                 freeze_zoe_encoder: bool = True,
                 pretrained_zoe: bool = True,
                 pretrained_planner: str = None,
                 zoe_model_name: str = "zoedepth_nk",
                 output_intermediate_depth: bool = False,
                 use_metric_head_residual: bool = True,
                 metric_head_mode: str = "auto",
                 use_both_heads: bool = False):
        """
        Args:
            k: 路径关键点数量
            goal_channels: 目标编码通道数
            freeze_zoe_encoder: 是否冻结 ZoeDepth 编码器
            pretrained_zoe: 是否使用预训练 ZoeDepth
            pretrained_planner: 预训练 iPlanner 权重路径 (可选)
            zoe_model_name: ZoeDepth 模型名称 ("zoedepth" 或 "zoedepth_nk")
            output_intermediate_depth: 是否输出中间深度图 (用于监督或可视化)
            use_metric_head_residual: 是否使用 Metric Head 特征的残差连接
            metric_head_mode: Metric Head 选择模式 ("auto", "nyu", "kitti")
            use_both_heads: 是否同时使用两个头的特征
        """
        super().__init__()
        
        self.k = k
        self.freeze_zoe_encoder = freeze_zoe_encoder
        self.output_intermediate_depth = output_intermediate_depth
        self.use_metric_head_residual = use_metric_head_residual
        self.metric_head_mode = metric_head_mode
        self.use_both_heads = use_both_heads
        self.zoe_model_name = zoe_model_name
        
        # 1. 加载 ZoeDepth 模型
        self.zoe_model = self._load_zoe_model(zoe_model_name, pretrained_zoe)
        
        # 2. 提取 ZoeDepth 的核心组件
        # ZoeDepth 结构: core (MiDaS DPT) -> depth_head
        self.depth_encoder = self.zoe_model.core  # MiDaS DPT backbone
        
        # 保留深度头用于输出深度图 (可选)
        if output_intermediate_depth:
            self.depth_head = self._get_depth_head()
        
        # 3. 特征适配层
        # MiDaS DPT 输出通道数通常为 256 (中间特征)
        self.feature_adapter = FeatureAdapter(
            zoe_channels=256,  # MiDaS DPT 特征通道数
            planner_channels=512,
            output_size=(12, 20)  # 匹配原始 iPlanner 解码器期望
        )
        
        # 4. Metric Head 残差适配层 (仅对 zoedepth_nk 模型启用)
        if use_metric_head_residual and zoe_model_name == "zoedepth_nk":
            self.metric_head_adapter = MetricHeadResidualAdapter(
                bin_embedding_dim=128,  # ZoeDepth NK 的 bin embedding 维度
                planner_channels=512,
                output_size=(12, 20),
                use_both_heads=use_both_heads
            )
            print(f"[RGB2PlannerNet] 启用 Metric Head 残差连接")
            print(f"  - 模式: {metric_head_mode}")
            print(f"  - 使用双头: {use_both_heads}")
        else:
            self.metric_head_adapter = None
        
        # 5. 规划解码器
        self.planner_decoder = PlannerDecoder(
            in_channels=512,
            goal_channels=goal_channels,
            k=k
        )
        
        # 加载预训练 iPlanner 权重 (可选)
        if pretrained_planner:
            self._load_planner_weights(pretrained_planner)
        
        # 冻结 ZoeDepth 编码器
        if freeze_zoe_encoder:
            self._freeze_encoder()
            
        print(f"[RGB2PlannerNet] 初始化完成")
        print(f"  - ZoeDepth 模型: {zoe_model_name}")
        print(f"  - 冻结编码器: {freeze_zoe_encoder}")
        print(f"  - 关键点数量: {k}")
        print(f"  - 输出中间深度图: {output_intermediate_depth}")
        print(f"  - Metric Head 残差: {use_metric_head_residual and zoe_model_name == 'zoedepth_nk'}")
    
    def _load_zoe_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """加载 ZoeDepth 模型"""
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config
        
        # 确定预训练权重路径
        pretrained_models = {
            "zoedepth": "ZoeDepthNKv1_14-Dec_21-34-fa7c108ac8c1_best.pt",
            "zoedepth_nk": "ZoeDepthNKv1_14-Dec_21-34-fa7c108ac8c1_best.pt",
        }
        
        if pretrained:
            model_filename = pretrained_models.get(model_name, "ZoeDepthNKv1_14-Dec_21-34-fa7c108ac8c1_best.pt")
            local_model_path = os.path.join(_workspace_root, 'ZoeDepth', 'zoedepth', 'models', model_filename)
            
            if os.path.exists(local_model_path):
                pretrained_resource = f"local::{local_model_path}"
                print(f"[RGB2PlannerNet] 加载本地预训练权重: {local_model_path}")
            else:
                pretrained_resource = None
                print(f"[RGB2PlannerNet] 警告: 本地权重不存在, 使用随机初始化")
        else:
            pretrained_resource = None
            
        conf = get_config(model_name, "train", pretrained_resource=pretrained_resource)
        conf['train_midas'] = not self.freeze_zoe_encoder
        
        model = build_model(conf)
        return model
    
    def _get_depth_head(self):
        """获取深度预测头"""
        # ZoeDepth 的深度头结构取决于具体模型
        # 这里我们复用 zoe_model 的推理功能
        return None  # 使用 zoe_model.infer() 代替
    
    def _load_planner_weights(self, path: str):
        """加载预训练 iPlanner 解码器权重"""
        if os.path.exists(path):
            try:
                planner_net, _ = torch.load(path, map_location='cpu')
                # 提取解码器权重
                decoder_state = {}
                for name, param in planner_net.named_parameters():
                    if 'decoder' in name:
                        new_name = name.replace('decoder.', '')
                        decoder_state[new_name] = param
                
                self.planner_decoder.load_state_dict(decoder_state, strict=False)
                print(f"[RGB2PlannerNet] 加载 iPlanner 解码器权重: {path}")
            except Exception as e:
                print(f"[RGB2PlannerNet] 警告: 无法加载 iPlanner 权重 - {e}")
    
    def _freeze_encoder(self):
        """冻结 ZoeDepth 编码器参数"""
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        print("[RGB2PlannerNet] ZoeDepth 编码器已冻结")
    
    def unfreeze_encoder(self, learning_rate_scale: float = 0.1):
        """解冻编码器用于端到端微调"""
        for param in self.depth_encoder.parameters():
            param.requires_grad = True
        self.freeze_zoe_encoder = False
        print(f"[RGB2PlannerNet] ZoeDepth 编码器已解冻 (建议学习率缩放: {learning_rate_scale}x)")
    
    def extract_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        从 RGB 图像提取深度特征
        
        Args:
            rgb: RGB 图像 (B, 3, H, W), 值范围 [0, 1] 或 [0, 255]
        Returns:
            特征图 (B, 256, H/4, W/4) 或类似尺寸
        """
        # 归一化到 [0, 1] 如果需要
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # 通过 MiDaS DPT backbone 提取特征
        # MiDaS DPT 的 forward 返回多尺度特征
        with torch.set_grad_enabled(not self.freeze_zoe_encoder):
            features = self.depth_encoder(rgb)
        
        # 取最后一层特征 (通常是最高分辨率的)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        elif isinstance(features, dict):
            # 某些版本可能返回字典
            features = features.get('out', list(features.values())[-1])
        
        return features
    
    def extract_metric_head_features(self, rgb: torch.Tensor) -> Optional[torch.Tensor]:
        """
        提取 ZoeDepth NK 的 Metric Head 特征 (bin embedding)
        
        Args:
            rgb: RGB 图像 (B, 3, H, W)
        Returns:
            bin embedding 特征 (B, 128, H', W') 或 (B, 256, H', W') 如果 use_both_heads
            如果不是 zoedepth_nk 模型则返回 None
        """
        if self.zoe_model_name != "zoedepth_nk" or self.metric_head_adapter is None:
            return None
        
        # 归一化
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        with torch.set_grad_enabled(not self.freeze_zoe_encoder):
            # 通过 ZoeDepth NK 的前向传播提取中间特征
            b, c, h, w = rgb.shape
            
            # 1. 获取 backbone 特征
            rel_depth, out = self.zoe_model.core(rgb, denorm=False, return_rel_depth=True)
            outconv_activation = out[0]
            btlnck = out[1]
            x_blocks = out[2:]
            
            # 2. 通过 conv2 处理瘦颈特征
            x_d0 = self.zoe_model.conv2(btlnck)
            x = x_d0
            
            # 3. 域分类器预测
            embedding = self.zoe_model.patch_transformer(x)[0]
            domain_logits = self.zoe_model.mlp_classifier(embedding)
            domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
            
            # 4. 根据模式选择头
            if self.metric_head_mode == "auto":
                bin_conf_name = ["nyu", "kitti"][torch.argmax(domain_vote, dim=-1).squeeze().item()]
            elif self.metric_head_mode == "nyu":
                bin_conf_name = "nyu"
            elif self.metric_head_mode == "kitti":
                bin_conf_name = "kitti"
            else:
                bin_conf_name = "nyu"  # 默认
            
            # 5. 获取对应头的 bin embedding
            if self.use_both_heads:
                # 同时使用两个头的特征
                embeddings = []
                for head_name in ["nyu", "kitti"]:
                    # 获取 seed bin 特征
                    seed_bin_regressor = self.zoe_model.seed_bin_regressors[head_name]
                    _, seed_b_centers = seed_bin_regressor(x)
                    b_embedding = self.zoe_model.seed_projector(x)
                    
                    # 通过 attractors
                    attractors = self.zoe_model.attractors[head_name]
                    conf = [c for c in self.zoe_model.bin_conf if c.name == head_name][0]
                    min_depth = conf['min_depth']
                    max_depth = conf['max_depth']
                    
                    if self.zoe_model.bin_centers_type in ['normed', 'hybrid2']:
                        b_prev = (seed_b_centers - min_depth) / (max_depth - min_depth)
                    else:
                        b_prev = seed_b_centers
                    prev_b_embedding = b_embedding
                    
                    for projector, attractor, x_block in zip(self.zoe_model.projectors, attractors, x_blocks):
                        b_embedding = projector(x_block)
                        b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
                        b_prev = b
                        prev_b_embedding = b_embedding
                    
                    embeddings.append(b_embedding)
                
                # 拼接两个头的特征
                head_features = torch.cat(embeddings, dim=1)
            else:
                # 只使用选定的头
                seed_bin_regressor = self.zoe_model.seed_bin_regressors[bin_conf_name]
                _, seed_b_centers = seed_bin_regressor(x)
                b_embedding = self.zoe_model.seed_projector(x)
                
                attractors = self.zoe_model.attractors[bin_conf_name]
                conf = [c for c in self.zoe_model.bin_conf if c.name == bin_conf_name][0]
                min_depth = conf['min_depth']
                max_depth = conf['max_depth']
                
                if self.zoe_model.bin_centers_type in ['normed', 'hybrid2']:
                    b_prev = (seed_b_centers - min_depth) / (max_depth - min_depth)
                else:
                    b_prev = seed_b_centers
                prev_b_embedding = b_embedding
                
                for projector, attractor, x_block in zip(self.zoe_model.projectors, attractors, x_blocks):
                    b_embedding = projector(x_block)
                    b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
                    b_prev = b
                    prev_b_embedding = b_embedding
                
                head_features = b_embedding
        
        return head_features
    
    def forward(self, rgb: torch.Tensor, goal: torch.Tensor, 
                output_depth: bool = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb: RGB 图像 (B, 3, H, W)
            goal: 目标点 (B, 3+) 
            output_depth: 是否输出深度图 (覆盖初始化设置)
        
        Returns:
            dict: {
                'keypoints': 路径关键点 (B, k, 3),
                'fear': 恐惧置信度 (B, 1),
                'depth': 深度图 (B, 1, H, W) - 可选,
                'head_used': 使用的 Metric Head 名称 - 可选
            }
        """
        output_depth = output_depth if output_depth is not None else self.output_intermediate_depth
        
        # 1. 提取深度特征
        features = self.extract_features(rgb)
        
        # 2. 特征适配
        adapted_features = self.feature_adapter(features)
        
        # 3. Metric Head 残差连接 (如果启用)
        if self.metric_head_adapter is not None:
            head_features = self.extract_metric_head_features(rgb)
            if head_features is not None:
                adapted_features = self.metric_head_adapter(head_features, adapted_features)
        
        # 4. 路径规划
        keypoints, fear = self.planner_decoder(adapted_features, goal)
        
        result = {
            'keypoints': keypoints,
            'fear': fear
        }
        
        # 5. 可选: 输出深度图
        if output_depth:
            with torch.no_grad():
                # 使用 ZoeDepth 完整推理获取深度图
                depth = self.zoe_model(rgb)
                if isinstance(depth, dict):
                    depth = depth.get('metric_depth', depth.get('out'))
                result['depth'] = depth
        
        return result
    
    def get_trainable_params(self) -> list:
        """获取可训练参数 (用于优化器)"""
        params = []
        
        # 特征适配层
        params.extend(list(self.feature_adapter.parameters()))
        
        # Metric Head 残差适配层
        if self.metric_head_adapter is not None:
            params.extend(list(self.metric_head_adapter.parameters()))
        
        # 规划解码器
        params.extend(list(self.planner_decoder.parameters()))
        
        # 如果解冻编码器
        if not self.freeze_zoe_encoder:
            params.extend(list(self.depth_encoder.parameters()))
        
        return params
    
    def get_param_groups(self, base_lr: float) -> list:
        """
        获取参数组 (用于不同学习率)
        
        Args:
            base_lr: 基础学习率
        Returns:
            参数组列表
        """
        param_groups = [
            {'params': self.feature_adapter.parameters(), 'lr': base_lr},
            {'params': self.planner_decoder.parameters(), 'lr': base_lr},
        ]
        
        # Metric Head 残差适配层
        if self.metric_head_adapter is not None:
            param_groups.append({
                'params': self.metric_head_adapter.parameters(),
                'lr': base_lr
            })
        
        if not self.freeze_zoe_encoder:
            param_groups.append({
                'params': self.depth_encoder.parameters(), 
                'lr': base_lr * 0.1  # 编码器使用较小学习率
            })
        
        return param_groups


def create_rgb2planner_net(config: Dict[str, Any] = None) -> RGB2PlannerNet:
    """
    工厂函数: 根据配置创建 RGB2PlannerNet
    
    Args:
        config: 配置字典
    Returns:
        RGB2PlannerNet 实例
    """
    default_config = {
        'k': 5,
        'goal_channels': 64,
        'freeze_zoe_encoder': True,
        'pretrained_zoe': True,
        'pretrained_planner': None,
        'zoe_model_name': 'zoedepth_nk',
        'output_intermediate_depth': False,
        'use_metric_head_residual': True,
        'metric_head_mode': 'auto',
        'use_both_heads': False,
    }
    
    if config:
        default_config.update(config)
    
    return RGB2PlannerNet(**default_config)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 RGB2PlannerNet")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 创建模型
    model = RGB2PlannerNet(
        k=5,
        freeze_zoe_encoder=True,
        pretrained_zoe=True,
        output_intermediate_depth=True
    )
    model = model.to(device)
    model.eval()
    
    # 测试输入
    batch_size = 2
    rgb = torch.rand(batch_size, 3, 384, 512).to(device)
    goal = torch.rand(batch_size, 7).to(device)  # x, y, z + quaternion
    
    print(f"\n输入:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Goal: {goal.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(rgb, goal, output_depth=True)
    
    print(f"\n输出:")
    print(f"  Keypoints: {output['keypoints'].shape}")
    print(f"  Fear: {output['fear'].shape}")
    if 'depth' in output:
        print(f"  Depth: {output['depth'].shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    
    print("\n测试完成!")
