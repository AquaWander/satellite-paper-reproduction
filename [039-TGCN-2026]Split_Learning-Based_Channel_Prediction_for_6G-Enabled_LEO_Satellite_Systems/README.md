# TGCN-2026 论文复现代码

## 论文信息

**标题**: Split Learning-Based Channel Prediction for 6G-Enabled LEO Satellite Systems

**作者**: K. Weththasinghe, Q. T. Ngo, Y. He, B. Jayawickrama (University of Technology Sydney)

**发表**: IEEE Transactions on Green Communications and Networking, Vol. 10, 2026, pp. 2609-2625

**DOI**: 10.1109/TGCN.2026.3680486

## 项目结构

```
[039-TGCN-2026]Split_Learning-Based_Channel_Prediction_for_6G-Enabled_LEO_Satellite_Systems/
├── config.py              # 仿真参数配置 (Table II & III)
├── channel_model.py       # TDL-D 信道模型、路径损耗、多普勒、NMSE计算
├── models.py              # TD-CNN-LSTM / CNN-LSTM / LSTM-only 网络架构
├── simulation.py          # 训练和评估逻辑
├── plotting.py            # IEEE期刊风格绘图
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[039-TGCN-2026]Split_Learning-Based_Channel_Prediction_for_6G-Enabled_LEO_Satellite_Systems"
python run_reproduction.py
```

## 复现结果

### Fig. 8: 训练收敛曲线
三种训练范式对比：Fully offline / Fully online / Proposed hybrid offline-online
- Hybrid在~190 epochs收敛（论文~600 epochs），比全在线快约60%
- 最终NMSE: Hybrid (-10.7dB) < Offline (-8.7dB) < Online (-3.2dB)

### Fig. 9: NMSE性能对比
不同TDD模式和模型在E_b/N_0 = -10~20 dB范围的NMSE性能
- SNR=10dB DSUUU: Proposed (-17.6dB) > CNN-LSTM [7] (-15.2dB) > LSTM-only [29] (-12.2dB)
- DSUUU优于DSUUD（更多上行时隙→更多训练数据）

### Fig. 13: 数据速率对比
数据速率 = log2(1 + Gamma_eff)，Gamma_eff = P*(1-NMSE)/(P*NMSE + sigma^2)
- DSUUU Proposed最高: ~6.4 bps/Hz
- Proposed始终优于CNN-LSTM [7]

## 核心算法

### TD-CNN-LSTM 架构
- **SM_LEO (卫星端)**: TimeDistributed Conv2D(16, 3x3) → LayerNorm → Conv2D(32, 3x3) → LayerNorm → Flatten (cut layer)
- **SM_UE (用户端)**: LSTM(512)×2 → Dropout(0.25) → Dense(1024, LeakyReLU) → Dense(1200)

### 分裂学习训练
模型分为LEO端和UE端，前向传播在cut layer处将smashed data从LEO发送到UE，反向传播时梯度从UE传回LEO更新参数。

### 混合离线-在线训练
1. 离线阶段：用理想CSI数据预训练
2. 在线阶段：用估计信道数据微调
3. 收敛速度比纯在线训练快约60%

## 仿真参数

| 参数 | 值 |
|---|---|
| 卫星高度 | 600 km |
| 仰角 | 90° |
| 天线阵元 | 256 (UPA) |
| 用户簇数 | 5 |
| 簇半径 | 10 km |
| 子载波数 | 300 |
| DMRS符号数 | 2 |
| 输入时间步 | 5 |
| LSTM单元 | 512 |
| SNR范围 | -10~20 dB |

## 依赖库

```bash
pip install numpy matplotlib torch scipy
```
