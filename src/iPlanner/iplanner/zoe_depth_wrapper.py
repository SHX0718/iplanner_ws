import sys
import os
import torch
from PIL import Image

# 添加 ZoeDepth 路径
_workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_workspace_root, 'ZoeDepth'))

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

class ZoeDepthConverter:
    def __init__(self, model_name="zoedepth"):
        """
        初始化 ZoeDepth 模型
        model_name: "zoedepth", "zoedepth_nk" 等
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        conf = get_config(model_name, "infer")
        self.model = build_model(conf)
        self.model = self.model.to(self.device)
        
    def rgb_to_depth(self, rgb_image):
        """
        将 RGB 图像转换为深度图
        rgb_image: PIL Image 或 numpy array
        返回: numpy array 格式的深度图
        """
        if isinstance(rgb_image, str):
            rgb_image = Image.open(rgb_image).convert("RGB")
        
        depth = self.model.infer_pil(rgb_image)
        return depth
    
    def rgb_to_depth_tensor(self, rgb_image):
        """返回 torch tensor 格式的深度图"""
        if isinstance(rgb_image, str):
            rgb_image = Image.open(rgb_image).convert("RGB")
        
        depth = self.model.infer_pil(rgb_image, output_type="tensor")
        return depth