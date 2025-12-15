#!/bin/bash
# 快速重建 TSDF 地图脚本
# 使用方法:
#   ./rebuild_tsdf.sh test_env          # 快速重建 test_env 场景（8个线程）
#   ./rebuild_tsdf.sh forest 16         # 使用 16 个线程重建 forest 场景
#   ./rebuild_tsdf.sh all               # 重建所有场景

WORKSPACE="/home/tms01/Developments/iplanner_ws"
SCENE="${1:-test_env}"
NUM_THREADS="${2:-8}"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate iplanner

cd "$WORKSPACE"

echo "========================================"
echo "快速 TSDF 地图重建工具"
echo "========================================"
echo "场景: $SCENE"
echo "线程数: $NUM_THREADS"
echo ""

if [ "$SCENE" = "all" ]; then
    # 重建所有场景
    for scene in test_env garage forest indoor campus tunnel; do
        echo "正在处理场景: $scene"
        python visualize_tsdf_fast.py --scene "$scene" --full_rebuild --num_threads "$NUM_THREADS"
        echo ""
    done
else
    # 重建单个场景
    python visualize_tsdf_fast.py --scene "$SCENE" --full_rebuild --num_threads "$NUM_THREADS"
fi

echo "========================================"
echo "完成！"
echo "========================================"
