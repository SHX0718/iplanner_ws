#!/bin/bash
# 快速可视化 TSDF 地图脚本
# 使用方法:
#   ./viz_tsdf.sh test_env 3d         # 3D 可视化代价地图
#   ./viz_tsdf.sh garage 2d           # 2D 热力图
#   ./viz_tsdf.sh forest topview      # 2D 俯视图
#   ./viz_tsdf.sh campus export       # 导出为文件

WORKSPACE="/home/tms01/Developments/iplanner_ws"
SCENE="${1:-test_env}"
MODE="${2:-3d}"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate iplanner

cd "$WORKSPACE"

case "$MODE" in
    "3d")
        echo "3D 可视化代价地图: $SCENE"
        python visualize_tsdf_map.py --scene "$SCENE" --view 3d --type cost
        ;;
    "2d")
        echo "2D 热力图: $SCENE"
        python visualize_tsdf_map.py --scene "$SCENE" --view 2d --type cost
        ;;
    "cloud")
        echo "3D 点云可视化: $SCENE"
        python visualize_tsdf_map.py --scene "$SCENE" --view 3d --type cloud
        ;;
    "topview")
        echo "2D 俯视图（多视角）: $SCENE"
        python visualize_tsdf_map.py --scene "$SCENE" --topview
        ;;
    "export")
        echo "导出为文件: $SCENE"
        python visualize_tsdf_map.py --scene "$SCENE" --export
        ;;
    *)
        echo "使用方法:"
        echo "  ./viz_tsdf.sh <场景名> <模式>"
        echo ""
        echo "可用模式:"
        echo "  3d       - 3D 可视化代价地图"
        echo "  2d       - 2D 热力图"
        echo "  cloud    - 3D 点云可视化"
        echo "  topview  - 2D 俯视图（多视角对比）"
        echo "  export   - 导出为 PLY、PNG、NumPy 文件"
        echo ""
        echo "示例:"
        echo "  ./viz_tsdf.sh test_env 3d"
        echo "  ./viz_tsdf.sh garage 2d"
        echo "  ./viz_tsdf.sh forest export"
        ;;
esac
