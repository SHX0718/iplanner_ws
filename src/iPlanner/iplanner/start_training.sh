#!/bin/bash
# RGB2PlannerNet 训练启动脚本

echo "========================================"
echo "RGB2PlannerNet 端到端训练"
echo "========================================"

# 1. 激活 Conda 环境
echo "[1/4] 激活 Conda 环境..."
cd /home/tms01/Developments/iplanner_ws
source quick_start.sh

# 2. 进入训练目录
echo "[2/4] 进入训练目录..."
cd /home/tms01/Developments/iplanner_ws/src/iPlanner/iplanner

# 3. 检查 GPU
echo "[3/4] 检查 GPU 可用性..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# 4. 启动训练
echo "[4/4] 启动训练..."
echo "----------------------------------------"

python train_rgb2planner.py \
  --config config/rgb2planner_config.json \
  --gpu 0 \
  --wandb

# 若想恢复训练，使用：
# python train_rgb2planner.py --resume models/rgb2planner_epoch15.pt --wandb

echo "========================================"
echo "训练完成或中断"
echo "========================================"
