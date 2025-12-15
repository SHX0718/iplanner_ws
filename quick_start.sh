#!/bin/bash
# ================================================================
# iPlanner 快速启动脚本
# 功能：选择场景并启动所有必要的ROS节点
# 每个命令在独立终端中运行
# ================================================================

WORKSPACE="/home/tms01/Developments/iplanner_ws"
CONDA_ENV="iplanner"

# 初始化conda并激活环境，然后source ROS工作空间
SETUP_CMD="source ~/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${WORKSPACE} && source devel/setup.bash"

# 可用场景列表
SCENES=("garage" "forest" "indoor" "campus" "tunnel")

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}    iPlanner 快速启动脚本${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# ---------------------- 选择场景 ----------------------
echo -e "${GREEN}请选择仿真场景：${NC}"
for i in "${!SCENES[@]}"; do
    echo "  $((i+1)). ${SCENES[$i]}"
done
echo ""

while true; do
    read -p "请输入选项 [1-5]: " scene_choice
    if [[ "$scene_choice" =~ ^[1-5]$ ]]; then
        SELECTED_SCENE="${SCENES[$((scene_choice-1))]}"
        break
    else
        echo -e "${RED}无效选项，请重新输入${NC}"
    fi
done
echo -e "${YELLOW}已选择场景: ${SELECTED_SCENE}${NC}"
echo ""

# ---------------------- 是否采集数据 ----------------------
read -p "是否启动数据采集器？[y/N]: " collect_data
if [[ "$collect_data" =~ ^[Yy]$ ]]; then
    ENABLE_COLLECTOR=true
    echo -e "${YELLOW}将启动数据采集器${NC}"
else
    ENABLE_COLLECTOR=false
    echo -e "${YELLOW}不启动数据采集器${NC}"
fi
echo ""

# ---------------------- 确认启动 ----------------------
echo -e "${BLUE}--------------------------------------${NC}"
echo -e "${GREEN}即将启动以下节点：${NC}"
echo "  1. roscore"
echo "  2. vehicle_simulator (场景: ${SELECTED_SCENE})"
echo "  3. iplanner_viz"
if [ "$ENABLE_COLLECTOR" = true ]; then
    echo "  4. data_collector"
fi
echo -e "${BLUE}--------------------------------------${NC}"
echo ""

read -p "确认启动？[Y/n]: " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo -e "${RED}已取消${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}正在启动...${NC}"

# ---------------------- 启动节点 ----------------------

# 1. 启动 roscore
echo -e "${YELLOW}[1/4] 启动 roscore...${NC}"
gnome-terminal --title="roscore" -- bash -c "${SETUP_CMD} && roscore; exec bash" 2>/dev/null &
sleep 3

# 2. 启动 vehicle_simulator
echo -e "${YELLOW}[2/4] 启动 vehicle_simulator (${SELECTED_SCENE})...${NC}"
gnome-terminal --title="vehicle_simulator - ${SELECTED_SCENE}" -- bash -c "${SETUP_CMD} && roslaunch vehicle_simulator vehicle_simulator.launch world_name:=${SELECTED_SCENE}; exec bash" 2>/dev/null &
sleep 5

# 3. 启动 iplanner_viz
echo -e "${YELLOW}[3/4] 启动 iplanner_viz...${NC}"
gnome-terminal --title="iplanner_viz" -- bash -c "${SETUP_CMD} && roslaunch iplanner_node iplanner_viz.launch; exec bash" 2>/dev/null &
sleep 2

# 4. 可选：启动 data_collector
if [ "$ENABLE_COLLECTOR" = true ]; then
    echo -e "${YELLOW}[4/4] 启动 data_collector...${NC}"
    gnome-terminal --title="data_collector" -- bash -c "${SETUP_CMD} && roslaunch iplanner_node data_collector.launch; exec bash" 2>/dev/null &
    sleep 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  所有节点已启动！${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "提示: 关闭各终端窗口可停止对应节点"
echo -e "      或使用 Ctrl+C 在各终端中停止"
