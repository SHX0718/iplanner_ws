import sys
import os
import torch
import numpy as np
from PIL import Image

# 设置离线模式,避免网络请求
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
os.environ['TORCH_HUB_DIR'] = os.path.expanduser('~/.cache/torch/hub')
# 设置较短的超时时间
import socket
original_timeout = socket.getdefaulttimeout()
socket.setdefaulttimeout(5)  # 5秒超时

# 添加 ZoeDepth 路径 - 在工作区根目录查找
_current_file = os.path.abspath(__file__)
_iplanner_dir = os.path.dirname(os.path.dirname(_current_file))  # iplanner 包目录
_src_dir = os.path.dirname(_iplanner_dir)  # src 目录
_workspace_root = os.path.dirname(_src_dir)  # 工作区根目录

_zoedepth_path = os.path.join(_workspace_root, 'ZoeDepth')
print(f"[INFO] Looking for ZoeDepth at: {_zoedepth_path}")

if not os.path.exists(_zoedepth_path):
    print(f"[ERROR] ZoeDepth directory not found at {_zoedepth_path}")
    raise FileNotFoundError(f"ZoeDepth directory not found at {_zoedepth_path}")

sys.path.insert(0, _zoedepth_path)

try:
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    print(f"[INFO] Successfully imported ZoeDepth modules")
except ModuleNotFoundError as e:
    print(f"[ERROR] Failed to import ZoeDepth: {e}")
    print(f"[INFO] ZoeDepth sys.path entry: {_zoedepth_path}")
    print(f"[INFO] sys.path: {sys.path}")
    raise

class ZoeDepthConverter:
    def __init__(self, model_name="zoedepth_nk", force_cpu=True, use_local=True, optimize_inference=True):
        """
        初始化 ZoeDepth 模型
        model_name: "zoedepth", "zoedepth_nk" 等（默认使用 zoedepth_nk，对应 ZoeD_M12_NK.pt）
        force_cpu: 强制使用 CPU（默认 True，避免 CUDA 兼容性问题）
        use_local: 使用本地缓存模型,避免网络请求（默认 True）
        optimize_inference: 优化推理性能（使用torch.inference_mode, 默认 True）
        """
        if force_cpu:
            self.device = "cpu"
            print("[INFO] 强制使用 CPU 运行 ZoeDepth")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.use_local = use_local
        self.optimize_inference = optimize_inference
        
        # 根据模型类型设置本地权重文件路径
        pretrained_models = {
            "zoedepth": "ZoeD_M12_N.pt",
            "zoedepth_nk": "ZoeD_M12_NK.pt",
        }
        model_filename = pretrained_models.get(model_name, None)
        
        # 构建本地模型权重文件的完整路径
        # 从当前文件向上找到工作区根目录: iplanner/zoe_depth_wrapper.py -> iplanner -> iPlanner -> src -> iplanner_ws
        _current_file = os.path.abspath(__file__)
        _iplanner_pkg_dir = os.path.dirname(os.path.dirname(_current_file))  # iplanner 包目录
        _src_dir = os.path.dirname(_iplanner_pkg_dir)  # src 目录
        _workspace_root = os.path.dirname(_src_dir)  # 工作区根目录
        local_model_path = os.path.join(_workspace_root, 'ZoeDepth', 'zoedepth', 'models', model_filename)
        
        # 检查本地文件是否存在
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model file not found: {local_model_path}")
        
        # 使用本地路径作为pretrained_resource
        pretrained_resource = f"local::{local_model_path}"
        
        # 加载配置并指定预训练权重
        conf = get_config(model_name, "infer", pretrained_resource=pretrained_resource)
        
        # 如果使用本地模式,设置force_reload=False避免网络请求
        if self.use_local:
            # 确保不重新下载MiDaS backbone
            conf['force_reload'] = False
            print(f"[INFO] 使用本地缓存的MiDaS模型,跳过网络下载")
        
        try:
            import time
            start_time = time.time()
            self.model = build_model(conf)
            load_time = time.time() - start_time
            print(f"[INFO] 模型构建耗时: {load_time:.2f}秒")
        except (OSError, socket.timeout, Exception) as e:
            # 如果失败,恢复网络超时设置并重试
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                print(f"[ERROR] 网络超时,无法下载MiDaS模型: {e}")
                print("[SOLUTION] 请确保:")
                print("  1. ~/.cache/torch/hub/intel-isl_MiDaS_master/ 目录存在")
                print("  2. 或者配置网络代理后重试")
                raise RuntimeError("无法加载MiDaS模型,请检查网络或本地缓存") from e
            else:
                print(f"[WARN] 本地加载失败: {e}")
                print("[INFO] 尝试使用网络模式加载...")
                socket.setdefaulttimeout(30)  # 延长超时
                self.model = build_model(conf)
        finally:
            # 恢复原始超时设置
            socket.setdefaulttimeout(original_timeout)
        
        self.model = self.model.to(self.device)
        # 设置为评估模式（禁用dropout和batch norm）
        self.model.eval()
        
        # 优化推理性能
        if self.optimize_inference:
            # 禁用梯度计算以加速推理
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[INFO] 已启用推理优化")
        
        print(f"[INFO] ZoeDepth模型加载完成")
        
    def rgb_to_depth(self, rgb_image):
        """
        将 RGB 图像转换为深度图
        rgb_image: PIL Image、numpy array 或图像文件路径
        返回: numpy array 格式的深度图
        """
        if isinstance(rgb_image, str):
            # 从文件路径加载
            rgb_image = Image.open(rgb_image).convert("RGB")
        elif isinstance(rgb_image, np.ndarray):
            # 从numpy数组转换为PIL Image
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                # 如果是浮点数，转换到0-255范围
                rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_image).convert("RGB")
        # 否则假设已经是PIL Image
        
        # 推理，使用inference_mode优化性能
        if self.optimize_inference:
            with torch.inference_mode():
                depth = self.model.infer_pil(rgb_image)
        else:
            depth = self.model.infer_pil(rgb_image)
        return depth
    
    def rgb_to_depth_tensor(self, rgb_image):
        """
        将 RGB 图像转换为深度图（返回tensor）
        rgb_image: PIL Image、numpy array 或图像文件路径
        返回: torch tensor 格式的深度图
        """
        if isinstance(rgb_image, str):
            # 从文件路径加载
            rgb_image = Image.open(rgb_image).convert("RGB")
        elif isinstance(rgb_image, np.ndarray):
            # 从numpy数组转换为PIL Image
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                # 如果是浮点数，转换到0-255范围
                rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_image).convert("RGB")
        # 否则假设已经是PIL Image
        
        # 推理，使用inference_mode优化性能
        if self.optimize_inference:
            with torch.inference_mode():
                depth = self.model.infer_pil(rgb_image, output_type="tensor")
        else:
            depth = self.model.infer_pil(rgb_image, output_type="tensor")
        return depth