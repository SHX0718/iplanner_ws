## **RGB2PlannerNet**端到端网络架构图
```mermaid
graph TB
    subgraph 输入层
        RGB[RGB图像<br/>B x 3 x 384 x 512]
        GOAL[目标点<br/>B x 7<br/>x,y,z,qx,qy,qz,qw]
    end

    subgraph ZoeDepth编码器[ZoeDepth Encoder - MiDaS DPT-BEiT]
        RGB --> PATCH[Patch Embedding<br/>图像分块]
        PATCH --> VIT[Vision Transformer<br/>BEiT-L-384]
        VIT --> FEAT1[多尺度特征提取]
        FEAT1 --> FEAT2[DPT Decoder<br/>特征融合]
        FEAT2 --> ZOE_FEAT[深度特征<br/>B x 256 x H/4 x W/4]
    end

    subgraph 特征适配层[Feature Adapter]
        ZOE_FEAT --> CONV1[Conv2d 256→256<br/>3x3, BN, ReLU]
        CONV1 --> CONV2[Conv2d 256→512<br/>3x3, BN, ReLU]
        CONV2 --> POOL[Adaptive AvgPool<br/>→ 12 x 20]
        POOL --> ADAPTED[适配特征<br/>B x 512 x 12 x 20]
    end

    subgraph 目标编码[Goal Encoding]
        GOAL --> FC_G[Linear 3→64]
        FC_G --> EXPAND[空间扩展<br/>B x 64 x 12 x 20]
    end

    subgraph 规划解码器[Planner Decoder]
        ADAPTED --> CAT{Concat}
        EXPAND --> CAT
        CAT --> FUSED[融合特征<br/>B x 576 x 12 x 20]
        FUSED --> DCONV1[Conv2d 576→512<br/>5x5, ReLU]
        DCONV1 --> DCONV2[Conv2d 512→256<br/>3x3, ReLU]
        DCONV2 --> FLAT[Flatten]
        FLAT --> FC1[Linear→1024, ReLU]
        
        FC1 --> FC2[Linear 1024→512]
        FC2 --> FC3[Linear 512→15]
        FC3 --> KEYPOINTS[路径关键点<br/>B x 5 x 3]
        
        FC1 --> FRC1[Linear 1024→128]
        FRC1 --> FRC2[Linear 128→1]
        FRC2 --> SIGMOID[Sigmoid]
        SIGMOID --> FEAR[Fear置信度<br/>B x 1]
    end

    subgraph 可选输出[可选: 深度图输出]
        ZOE_FEAT -.-> DEPTH_HEAD[ZoeDepth Head]
        DEPTH_HEAD -.-> DEPTH[深度图<br/>B x 1 x H x W]
    end

    subgraph 输出层
        KEYPOINTS --> TRAJ[轨迹生成<br/>TrajOpt]
        TRAJ --> PATH[可行驶路径]
        FEAR --> AVOID[避障决策]
    end

    style RGB fill:#e1f5fe
    style GOAL fill:#e1f5fe
    style KEYPOINTS fill:#c8e6c9
    style FEAR fill:#c8e6c9
    style PATH fill:#a5d6a7
    style AVOID fill:#a5d6a7
```

## 两阶段训练流程
```mermaid
graph LR
    subgraph Stage1[Stage 1: 冻结编码器训练 - 30 epochs]
        S1_DATA[RGB + 轨迹数据] --> S1_ZOE[ZoeDepth Encoder<br/>❄️ 冻结]
        S1_ZOE --> S1_ADAPT[Feature Adapter<br/>🔥 训练]
        S1_ADAPT --> S1_DEC[Planner Decoder<br/>🔥 训练]
        S1_DEC --> S1_LOSS[损失计算]
        S1_LOSS --> S1_BP[反向传播<br/>仅更新适配层+解码器]
    end

    subgraph Stage2[Stage 2: 端到端微调 - 20 epochs]
        S2_DATA[RGB + 轨迹数据] --> S2_ZOE[ZoeDepth Encoder<br/>🔥 训练 lr=0.1x]
        S2_ZOE --> S2_ADAPT[Feature Adapter<br/>🔥 训练]
        S2_ADAPT --> S2_DEC[Planner Decoder<br/>🔥 训练]
        S2_DEC --> S2_LOSS[损失计算]
        S2_LOSS --> S2_BP[反向传播<br/>更新全部参数]
    end

    Stage1 --> Stage2
```

## 损失函数组成图
```mermaid
graph TB
    subgraph 输入
        PRED_WP[预测路径点]
        PRED_FEAR[预测Fear]
        GT_ODOM[里程计]
        GT_GOAL[目标点]
        MAP[TSDF地图]
    end

    subgraph 损失计算
        PRED_WP --> OLOSS[障碍物损失<br/>α=0.5]
        MAP --> OLOSS
        
        PRED_WP --> HLOSS[地形高度损失<br/>β=1.0]
        MAP --> HLOSS
        
        PRED_WP --> MLOSS[运动平滑损失<br/>γ=2.0]
        GT_GOAL --> MLOSS
        
        PRED_WP --> GLOSS[目标到达损失<br/>δ=5.0]
        GT_GOAL --> GLOSS
        
        PRED_FEAR --> FLOSS[Fear分类损失<br/>BCE]
        OLOSS --> FEAR_LABEL[Fear标签生成]
        FEAR_LABEL --> FLOSS
    end

    subgraph 总损失
        OLOSS --> TOTAL[L_total]
        HLOSS --> TOTAL
        MLOSS --> TOTAL
        GLOSS --> TOTAL
        FLOSS --> TOTAL
    end

    TOTAL --> BP[反向传播]
```

## 数据流与维度变化图
```mermaid
graph LR
    subgraph 数据维度变化
        I1[RGB<br/>B×3×384×512] --> I2[Patch<br/>B×768×24×32]
        I2 --> I3[ViT特征<br/>B×768×24×32]
        I3 --> I4[DPT特征<br/>B×256×96×128]
        I4 --> I5[适配特征<br/>B×512×12×20]
        I5 --> I6[融合特征<br/>B×576×12×20]
        I6 --> I7[卷积后<br/>B×256×8×16]
        I7 --> I8[展平<br/>B×32768]
        I8 --> I9[全连接<br/>B×1024]
        I9 --> I10[关键点<br/>B×5×3]
    end
```

## 推理流程图
```mermaid
graph TB
    START[开始推理] --> LOAD[加载RGB2PlannerNet]
    LOAD --> CHECK{GPU可用?}
    CHECK -->|是| GPU[模型移至GPU]
    CHECK -->|否| CPU[使用CPU]
    GPU --> INPUT[输入RGB图像 + 目标点]
    CPU --> INPUT
    INPUT --> FORWARD[前向传播]
    FORWARD --> EXTRACT[ZoeDepth特征提取<br/>~0.4s]
    EXTRACT --> ADAPT[特征适配<br/>~0.01s]
    ADAPT --> DECODE[路径解码<br/>~0.01s]
    DECODE --> OUTPUT[输出路径关键点 + Fear]
    OUTPUT --> TRAJ[轨迹插值优化]
    TRAJ --> RESULT[可执行路径]
    RESULT --> END[结束]
```

## 双头特征
```mermaid
graph TB
    subgraph 改进方案
        RGB[RGB] --> BACKBONE[MiDaS Backbone]
        BACKBONE --> SHARED[共享特征 256ch]
        
        SHARED --> ADAPTER[Feature Adapter]
        
        subgraph 可选: 利用双头
            SHARED --> HEAD_N[Head N 特征]
            SHARED --> HEAD_K[Head K 特征]
            HEAD_N --> CONCAT{特征拼接}
            HEAD_K --> CONCAT
            CONCAT --> ADAPTER2[增强适配层]
        end
        
        ADAPTER --> PLANNER[Planner Decoder]
        ADAPTER2 -.-> PLANNER
        
        GOAL[目标点] --> PLANNER
        PLANNER --> PATH[路径 + Fear]
    end
```

## 残差链接架构
```mermaid
graph TB
    subgraph RGB2PlannerNet["RGB2PlannerNet (带残差连接)"]
        RGB[RGB输入] --> BACKBONE[MiDaS DPT<br/>Backbone]
        
        BACKBONE --> FEAT[共享特征<br/>256ch]
        BACKBONE --> OUT[多尺度特征<br/>x_blocks]
        BACKBONE --> BTL[瓶颈特征<br/>btlnck]
        
        BTL --> ROUTER[Domain Router]
        ROUTER --> SELECT{选择头}
        
        BTL --> HEAD_N[Metric Head N<br/>NYU]
        BTL --> HEAD_K[Metric Head K<br/>KITTI]
        OUT --> HEAD_N
        OUT --> HEAD_K
        
        HEAD_N --> BEMB_N[bin_embedding N<br/>128ch]
        HEAD_K --> BEMB_K[bin_embedding K<br/>128ch]
        
        BEMB_N --> RESIDUAL_ADAPTER[残差适配层<br/>MetricHeadResidualAdapter]
        BEMB_K -.-> RESIDUAL_ADAPTER
        
        FEAT --> ADAPTER[Feature Adapter<br/>256→512ch]
        ADAPTER --> ADD((+))
        RESIDUAL_ADAPTER --> |残差连接<br/>scale=0.1| ADD
        
        ADD --> DECODER[Planner Decoder]
        GOAL[目标点] --> DECODER
        DECODER --> PATH[路径关键点]
        DECODER --> FEAR[Fear置信度]
    end

    style HEAD_N fill:#e3f2fd
    style HEAD_K fill:#fff3e0
    style RESIDUAL_ADAPTER fill:#e8f5e9
    style ADD fill:#ffeb3b
    ```