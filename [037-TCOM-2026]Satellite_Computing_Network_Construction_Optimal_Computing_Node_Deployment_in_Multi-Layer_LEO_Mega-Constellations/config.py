"""
config.py — 论文仿真参数配置
Satellite Computing Network Construction: Optimal Computing Node Deployment
in Multi-Layer LEO Mega-Constellations (IEEE TCOM 2026)
"""

import numpy as np

# ============================================================
# 星座配置 (Table II)
# ============================================================
CONSTELLATIONS = {
    # Starlink 三层星座
    "Starlink": {
        "altitude_km": 550,       # 轨道高度 (km)
        "inclination_deg": 53.0,  # 轨道倾角 (度)
        "satellites_per_orbit": 22,  # N: 每轨道卫星数
        "num_orbits": 72,         # M: 轨道数
        "layers": 3,              # L: 层数
        "total_satellites": 22 * 72,  # 1584
    },
    # OneWeb 三层星座
    "OneWeb": {
        "altitude_km": 1200,      # 轨道高度 (km)
        "inclination_deg": 87.9,  # 轨道倾角 (度)
        "satellites_per_orbit": 36,  # N: 每轨道卫星数
        "num_orbits": 18,         # M: 轨道数
        "layers": 3,              # L: 层数
        "total_satellites": 36 * 18,  # 648
    },
}

# ============================================================
# 默认仿真参数
# ============================================================
DEFAULT_L = 7           # 层数
DEFAULT_N = 50          # 每轨道卫星数 (仿真用)
DEFAULT_M = 50          # 轨道数 (仿真用)
DEFAULT_J = 1           # 默认可达跳数

# 网络规模扫描范围 (Fig.4)
NETWORK_SIZES = np.arange(10, 101, 10)  # N=M 从 10 到 100

# 可达跳数范围 (Fig.4, Fig.5, Fig.6)
HOP_RANGE = np.arange(1, 7)  # J = 1, 2, 3, 4, 5, 6

# ============================================================
# 通信参数
# ============================================================
FREQ_GHZ = 30.0         # 通信频率 (GHz), Ka频段
FREQ_HZ = FREQ_GHZ * 1e9
ANTENNA_GAIN_DB = 18.0   # 天线增益 G_t, G_r (dBi)
SENSITIVITY_DBM = -105.0  # 天线灵敏度 (dBm)

# ============================================================
# 延迟模型参数（物理延迟模型）
# ============================================================
# 轨道参数
DEFAULT_LEO_ALT_KM = 550.0     # LEO默认轨道高度 (km)
LAYER_HEIGHT_DIFF_KM = 100.0   # LEO层间高度差 (km)

# 处理延迟
PROCESSING_DELAY_MS = 0.5       # 每跳处理延迟 (ms)

# 跨层ISL参数
CROSS_LAYER_DELAY_MS = 1.0      # 跨层链路传播延迟基数 (ms, 后续物理计算覆盖)

# MEO 计算节点参数
MEO_ALTITUDE_KM = 5000.0        # MEO轨道高度 (km)
MEO_BEAM_ANGLE_DEG = 10.0       # MEO波束覆盖角 (度)

# 保留兼容旧代码的别名
ISL_HOP_DELAY_MS = 5.0          # (兼容，实际延迟由物理模型计算)
MEO_HOP_DELAY_MS = 15.0         # (兼容)
GS_HOP_DELAY_MS = 20.0          # (兼容)

# ============================================================
# 能耗模型参数 (Eq.26, Eq.27)
# ============================================================
FREE_SPACE_LOSS_CONST = 32.45  # 自由空间损耗常数
TRANSMIT_POWER_W = 10.0        # 发射功率 (W)
RETRANSMISSION_FACTOR = 1.2    # 平均重传次数
CROSS_LAYER_ENERGY_FACTOR = 1.2  # 跨层链路能耗增加20%

# ============================================================
# 物理常数
# ============================================================
EARTH_RADIUS_KM = 6371.0   # 地球半径 (km)
LIGHT_SPEED = 3e8           # 光速 (m/s)

# ============================================================
# PSO 优化参数 (对比方法)
# ============================================================
PSO_NUM_PARTICLES = 50
PSO_MAX_ITER = 200
PSO_W = 0.7    # 惯性权重
PSO_C1 = 1.5   # 认知系数
PSO_C2 = 1.5   # 社会系数

# ============================================================
# 绘图配置
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# IEEE 期刊风格全局配置
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 6
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.minor.width'] = 0.5
rcParams['ytick.minor.width'] = 0.5
rcParams['grid.linewidth'] = 0.3
rcParams['grid.alpha'] = 0.3

# 图尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色方案（高对比度，灰度可区分）
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]
