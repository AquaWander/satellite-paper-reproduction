# TWC-2026 论文复现代码

## 论文信息

**标题**: Collaborative Beam Hopping of Load Balancing and Interference Avoidance for Multi-GEO Satellite Systems Using QMIX

**作者**: Ning Chen, Ailing Xiao, Sheng Wu, Feng Ding, Chunxiao Jiang, Wei Zhang

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[046-TWC-2026]Collaborative_Beam_Hopping_of_Load_Balancing_and_Interference_Avoidance_for_Multi-GEO_Satellite_Systems_Using_QMIX/
├── config.py              # 仿真参数配置
├── environment.py         # 多GEO卫星跳波束仿真环境
├── qmix_agent.py          # QMIX 算法实现 (CTDE框架)
├── baselines.py           # 对比算法 (RBH, PBH, DFBH, GABH, SADRL-BH, MADRL-BH, LBGT-MSBH, PH-MSBH, I-MSBH, CMARL-MSBH)
├── simulation.py          # 主仿真逻辑
├── plotting.py            # 绘图模块
├── run_reproduction.py    # 快速复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[046-TWC-2026]Collaborative_Beam_Hopping_of_Load_Balancing_and_Interference_Avoidance_for_Multi-GEO_Satellite_Systems_Using_QMIX"
python run_reproduction.py
```

## 复现结果

### Fig.9: 吞吐量对比

复现了 7 种单星 BH 算法在流量需求 500-3000 Mbps 下的吞吐量对比。启发式算法 (RBH, PBH, DFBH, GABH) 能够服务全部流量需求。

### Fig.10: 延迟公平性对比

展示了各算法的延迟偏移。DFBH (延迟公平跳波束) 在延迟公平性方面表现最好。

### Fig.11: 多星 MSBH 对比

(a) 平均流量满意度率: 启发式算法达到 Sat=1.0
(b) 归一化负载偏移: 各算法负载均衡对比

## 核心算法

### QMIX (CTDE框架)

- **智能体网络**: 前馈网络 (输入归一化流量 → 隐藏层 ReLU → 输出 Q 值)
- **Mixing Network + Hypernetwork**: 根据全局状态生成归一化权重, 保证 Q_tot 对 Q_i 的单调性
- **经验回放**: 存储 (s, a, r, s') 经验
- **软更新**: θ' = τ·θ + (1-τ)·θ', τ=0.005

### 优化目标 (Eq.28)

```
Ft = ω_T·T^{lt} - ω_D·D^{lt} - ω_L·L^{lt}
```

其中 ω_T=ω_D=ω_L=1

### 干扰惩罚 (Eq.29-32)

星内同小区重复照射、星间同小区重复照射、邻小区同频干扰三类惩罚。

## 依赖库

```bash
pip install numpy matplotlib scipy
```

## 复现说明

### 仿真规模

- 3 颗 GEO 卫星 (Viasat-3)
- 每星 100 小区 (原文 441, 简化)
- 每星 10 波束 (原文 21, 简化)
- 5 个子频带 × 200 MHz = 1 GHz 总带宽

### 训练规模

- QMIX: 300 epochs (原文 3500)
- SADRL/MADRL/CMARL: 300 epochs

### 说明

由于计算资源限制，DRL 算法的训练轮数 (300) 远小于论文 (3500)，因此 DRL 算法 (SADRL, MADRL, CMSBH) 的性能未完全收敛。启发式算法 (RBH, PBH, DFBH, GABH) 的结果与论文趋势一致。