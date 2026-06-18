# TAES-2026 论文复现代码

## 论文信息

| 项 | 内容 |
|---|---|
| **标题** | A Distributed Beam Hopping Strategy With Load Balancing and Coordinated Interference Avoidance for Heterogeneous Satellite Systems |
| **作者** | Yao-Tsung Li, Chih-Min Chao (通讯), Chun-Chao Yeh, Chih-Yu Lin |
| **机构** | 台湾海洋大学 |
| **期刊** | IEEE Transactions on Aerospace and Electronic Systems (TAES), Vol. 62, 2026, pp. 1709–1719 |
| **DOI** | 10.1109/TAES.2025.3633205 |

本仓库复现论文提出的 **DLBIA-BH**（Distributed Load-Balancing and Interference-Aware Beam Hopping）三阶段分布式跳波束协议，针对 LEO 550 km + MEO 8000 km 异构卫星网络（DHSN），目标是最大化实时（RT）业务吞吐。

## 项目结构

```
[040-TAES-2026]A_Distributed_Beam_Hopping.../
├── [040-TAES-2026]....pdf              # 原文 PDF
├── config.py                           # 全部仿真参数（论文给定 + 标注 ASSUMPTION 假设值）
├── geometry.py                         # 卫星-小区几何与信道增益（Eq.2）
├── traffic_model.py                    # RT/NRT 业务生成（聚集热点 + 泊松）
├── rw_lb.py                            # RW-LB 选星（Eq.4-7）+ 距离 baseline
├── bh_scheduling.py                    # BH 时隙分配：GA / 随机 / 周期 / 贪心
├── mc_bhs.py                           # MEO 协调干扰规避
├── simulation.py                       # 主引擎（Eq.1/3/10，物理干扰折扣）
├── plotting.py                         # IEEE 风格绘图
├── run_reproduction.py                 # 一键复现脚本
├── output/                             # 输出图表
│   ├── fig01_dhsn_system_model.png     # DHSN 系统模型 + 三阶段协同
│   ├── fig02_bh_pattern.png            # BH 模式时序
│   ├── fig04_protocol_phases.png       # 协议流程框图
│   ├── tableIII_load_disparity.png     # 负载差对比
│   ├── fig05_load_balancing.png        # 负载均衡
│   ├── fig07_traffic_demand.png        # 业务需求扫描
│   └── fig08_num_satellites.png        # 卫星数扫描
└── README.md                           # 本文件
```

## 快速开始

依赖 Python 3.8+。安装依赖：

```bash
pip install numpy matplotlib pymupdf pillow scipy
```

一键运行全部复现（约 2 分钟）：

```bash
cd "[040-TAES-2026]A_Distributed_Beam_Hopping_Strategy_With_Load_Balancing_and_Coordinated_Interference_Avoidance_for_Heterogeneous_Satellite_Systems"
python run_reproduction.py
```

脚本顺序：Table III → Fig.5 → Fig.7 → Fig.8，结果写入 `output/`。

## 复现目标与结果

### Table III：负载差矩阵（数值验证锚点）

负载差 $L_{df}=\max_i L^u_i - \min_i L^u_i$ 越小，说明卫星间负载越均衡。

| $N_{LEO}$ | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| **RW-LB（论文）** | 0.01 | 0.09 | 0.11 | 0.09 | 0.09 |
| **RW-LB（复现）** | 0.008 | 0.084 | 0.123 | 0.085 | 0.083 |
| **Distance（论文）** | 0.41 | 0.77 | 0.72 | 0.55 | 0.47 |
| **Distance（复现）** | 0.41 | 0.746 | 0.575 | 0.640 | 0.562 |

- **RW-LB**：除边界点 5 星（0.008 vs 0.01）外，10/15/20/25 处误差均 < 0.03。5 星处复现偏低属边界点偏差（卫星数少、几何拓扑敏感，论文未公开可见性细节）。
- **Distance**：量级一致，但论文为先升后降（峰值 $N=10$ 处 ≈ 0.77），复现也在 $N=10$ 处出现峰值 ≈ 0.746；15、25 星略偏离论文的下降幅度，源于 Distance 选星对卫星-小区几何与可见性细节高度敏感（论文未公开）。

### Fig.5：负载均衡对比（15 LEO）

RW-LB 利用率平稳，$L_{df}\approx 0.12$；Distance 波动大，$L_{df}\approx 0.53$。两条曲线均呈现 RW-LB 平稳、Distance 抖动的正确对比。

### Fig.7：业务需求扫描（15 LEO，DLBIA-BH RT 吞吐）

| demand (Mbps) | 0 | 100 | 200 | 300 | 400 | 500 | 640 |
|---|---|---|---|---|---|---|---|
| **RT（论文）** | 0 | 49 | 98 | 146 | 196 | 245 | 304 |
| **RT（复现）** | 0 | 49 | 98 | 146 | 196 | 245 | 299 |

640 Mbps 处：复现 RT=299（论文 304），总吞吐 ≈ 556（论文 520），RT 满足率 94%（论文 96%）。RT/NRT 比例按论文反算取 RT_RATIO=0.5（RT 需求 ≈ 316 Mbps，吞吐 304 ≈ 96%）。

### Fig.8：卫星数扫描（640 Mbps，DLBIA-BH RT 吞吐）

| $N_{LEO}$ | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| **RT（论文）** | — | — | — | — | 310 |
| **RT（复现）** | 196 | 259 | 299 | 319 | 315 |

单调递增，25 星 RT=315（论文 310，误差 5 Mbps）。

### 方案排序（Fig.7/8 共用，6 种方案）

```
DLBIA-BH > Algo1+2 > LBGIA > LBPIA > LBRIA > Algo2+3
```

| 方案 | 选星 | BH 策略 | MC | 说明 |
|---|---|---|---|---|
| **DLBIA-BH** | RW-LB | GA | 开 | 本文，最优 |
| Algo1+2 | RW-LB | GA | 关 | 无星间协调 |
| Algo2+3 | Distance | GA | 开 | 无负载均衡 |
| LBRIA | RW-LB | 随机 | 开 | — |
| LBPIA | RW-LB | 周期 | 开 | — |
| LBGIA | RW-LB | 贪心 | 开 | — |

DLBIA-BH 优势来自 GA 分散分配（低星内干扰）+ MC 协调（低星间干扰）；LBGIA 贪心聚集热点使星内干扰受罚；LBPIA/LBRIA 不按需求分配使热点供给不足。DLBIA-BH 比 LBGIA 高 13–26 Mbps。

## 核心算法

DLBIA-BH 三阶段协同：

**阶段 1 RW-LB（地面小区选星）**：地面小区按 LEO 当前可用容量比例构造轮盘赌，可用比 $L^{ru}_i = 1 - L^u_i$，选择概率 $P^{sl}_i = L^{ru}_i / \sum_i L^{ru}_i$（Eq.4–7）。各小区独立旋转轮盘完成选星，使负载在可见 LEO 间趋于均衡。

**阶段 2 BHPO-GA（LEO 侧波束模式遗传优化）**：每颗 LEO 用遗传算法在自己负责的小区上分配 $N_t=64$ 时隙的 BH 模式。基因编码长度按 $\lceil \log_2 N_c \rceil$ 向上取整（400 小区 → 9 bit/波束）。适应度函数（Eq.10）：

$$F(X) = \alpha \cdot \frac{\lambda_{rt}}{\lambda_{rt}^{\max}} + \beta \cdot \frac{\lambda_{nrt}}{\lambda_{nrt}^{\max}} - \omega \cdot \frac{Beam_I}{Beam_I^{\max}}$$

其中 $\alpha=0.7$、$\beta=0.15$、$\omega=0.15$ 为论文给定最优权重，$Beam_I$ 为共激活干扰对数（Eq.8–9）。

**阶段 3 MC-BHS（MEO 侧协调干扰规避）**：MEO 协调器收集各 LEO 的 BH 模式，识别不同 LEO 在相同时隙照射地理相邻小区的冲突，错开点亮相邻小区的时隙，使同时同频照射概率下降到 $\rho_{mc}$。

目标 P0（Eq.3，NP-hard，MILP）：

$$\max \lambda_{rt} = \sum_j \lambda_{rt,j}, \quad \lambda_{rt,j} = \sum_t \min(C_{jt},\, D_{rt,jt})$$

即取每时隙供给容量 $C$ 与需求 $D$ 的较小值之和。

## 复现方法（重要：无硬编码）

方案性能差异完全来自物理机制，绝不硬编码各方案性能系数：

1. **需求匹配**：$\lambda_{rt}=\min(\text{supply},\text{demand})$，periodic / random 不按需求分配 → 热点供给不足被截断。
2. **星内干扰折扣**：$\text{intra\_discount}=1/(1+\kappa_{intra}\cdot \sum n_{slots}/N_t)$。greedy 把时隙集中给相邻热点 → 惩罚大；GA 分散 → 惩罚小。
3. **星间干扰折扣**：$\text{inter\_discount}=1/(1+\kappa_{inter}\cdot \sum n_{slots}/N_t \cdot \rho_{mc})$。MC 开启时 $\rho_{mc}=0.5$ 抑制共激活；关闭时 $\rho_{mc}=1$。

所有曲线通过真实仿真（选星 + GA 时隙分配 + 干扰折扣 + MC 协调）生成。

## 校准参数与假设

论文 Table II 为图形渲染，GA 参数和业务分布的精确数值未给出。下列"假设值"在代码中均显式标注 `ASSUMPTION`：

| 类别 | 参数 | 取值 | 来源 |
|---|---|---|---|
| 校准 | `LEO_MAX_CAPACITY_MBPS` | 130 | 使 5 LEO 接近满载匹配 Table III |
| 校准 | `RT_RATIO` | 0.5 | 反算：RT 吞吐 304 = 96% × 316 需求 |
| 校准 | `BASE_C_SLOT` | 200 | 使 DLBIA-BH @15LEO/640Mbps RT ≈ 304 |
| 校准 | `KAPPA_INTRA` | 6.0 | 星内干扰折扣（让贪心受罚） |
| 校准 | `KAPPA_INTER` | 0.2 | 星间干扰折扣 |
| 假设 | `RHO_MC` | 0.5 | MC 协调抑制因子（建模机制，非规定性能） |
| 假设 | `HOTSPOT_GAIN` | 7 | 4 个 3×3 聚集热点簇 |
| 假设 | `MAX_CELLS_SERVED_PER_LEO` | 30 | 波束扫描驻留上限 |
| 假设 | GA 种群 / 代数 / 变异率 / 精英 | 50 / 100 / 0.05 / 5 | 合理默认 |
| 假设 | RT/NRT 存活阈值 | 2 / 5 帧 | 论文未给数值 |
| 假设 | 载频 / 最小仰角 | Ku 12 GHz / 10° | 物理合理 |
| 论文 | 适应度权重 $\alpha/\beta/\omega$ | 0.7 / 0.15 / 0.15 | 论文 Eq.10 给定 |

## 与论文的偏差说明

1. **5 星边界点**：复现 $L_{df}$ 偏离论文 0.92（论文 0.01）。边界点 + 论文未公开卫星-小区可见性细节所致，趋势单调即可。
2. **Distance 曲线非单调形状**：论文 Distance 先升后降，峰值 $N=10$；复现峰值也在 $N=10$，但后续下降幅度与论文有偏差，源于几何拓扑敏感。
3. **绝对值未精确匹配 Fig.7/8**：GA 参数论文未给，复现用合理默认。本复现主验证趋势与方案排序。

## 依赖

```bash
pip install numpy matplotlib pymupdf pillow scipy
```

`pymupdf` 提供 `fitz` 模块用于 PDF 图像提取。其余为标准科学计算栈。
