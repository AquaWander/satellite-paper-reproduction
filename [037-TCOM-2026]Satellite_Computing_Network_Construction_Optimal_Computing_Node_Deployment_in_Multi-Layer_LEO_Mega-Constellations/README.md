# TCOM-2026 论文复现代码

## 论文信息

**标题**: Satellite Computing Network Construction: Optimal Computing Node Deployment in Multi-Layer LEO Mega-Constellations

**作者**: Xiao Jia, Di Zhou, Min Sheng, Yan Shi, Sijing Ji, Jiandong Li (西安电子科技大学)

**发表**: IEEE Transactions on Communications, Vol. 74, 2026

## 项目结构

```
[037-TCOM-2026]Satellite_Computing_Network_Construction.../
├── config.py              # 仿真参数配置（星座、通信、延迟、能耗）
├── deployment.py          # 核心算法：覆盖节点数计算、Lee距离、线性码构造
├── simulation.py          # 主仿真逻辑：N_C趋势、延迟对比、节点数对比
├── plotting.py            # IEEE期刊风格绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
│   ├── fig01_satellite_computing_network_architecture.png
│   ├── fig02_coverage_structures.png
│   ├── fig03_computing_node_deployment_3d.png
│   ├── fig4.png            # 计算节点数量(a)和信令延迟(b) vs 网络规模
│   ├── fig5.png            # LEO vs MEO 信令分发延迟对比
│   └── fig6.png            # 不同方案计算节点数
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[037-TCOM-2026]Satellite_Computing_Network_Construction_Optimal_Computing_Node_Deployment_in_Multi-Layer_LEO_Mega-Constellations"
python run_reproduction.py
```

## 复现结果

### Fig.4: 计算节点数量和信令分发延迟随网络规模和可达跳数的变化

- **(a)** 计算节点数量N_C随网络规模N=M(10-100)和可达跳数J(1-6)的变化
- **(b)** 信令分发平均延迟随网络规模的变化（物理模型：ISL距离=2π(R+h)/N，延迟随N减小）

核心公式验证（Table III全部21个数据点精确匹配）：
- V_I(J) = 2J² + 2J + 1（菱形覆盖）
- V_II(J) = (4/3)J³ + 2J² + (8/3)J + 1（点波束三维覆盖）
- N_C = ⌊N·M·L / V(J)⌋

### Fig.5: LEO与MEO计算节点的信令分发延迟对比

L=7层, N=M=50网络下，对比LEO和MEO两种计算节点部署方案：
- LEO计算节点: 延迟低（2.9~16.3ms），通过短距离ISL分发信令
- MEO计算节点: 延迟高（31.4~37.1ms），需LEO→MEO→LEO长距离传输

### Fig.6: 不同部署方案所需计算节点数量

三个星座场景下（Starlink 72×22×3, OneWeb 18×36×3, 大型网络 50×50×7），对比3种方案：
- LEO Spot Beam（本文方法）: 点波束三维覆盖
- LEO Polygon Beam: 二维菱形覆盖，节点数最多
- MEO Computing: 覆盖范围最大，节点数最少

## 核心算法

**Algorithm 1 — 点波束计算节点部署**：
1. 计算V(J)（点波束三维覆盖节点数）
2. 构造基向量 a₁=[J, J+1, 0], a₂=[0, J, J+1]
3. 生成线性码 cₗ = mod(i·a₁ + j·a₂, V(J))
4. 复制平移到整个(N, M, L)网络并裁剪

该算法将纠错码理论（Lee距离下的完美纠错码）应用于卫星网络计算节点部署，时间复杂度O(V(J)² + N·M·L)。

## 依赖库

```bash
pip install numpy matplotlib PyMuPDF Pillow
```
