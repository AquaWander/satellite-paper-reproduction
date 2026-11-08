"""
config.py — 论文仿真参数配置
[046-TWC-2026] Collaborative Beam Hopping using QMIX
"""

# ========== 星座参数 ==========
N_SATELLITES = 3          # 卫星数量 N_s
ORBIT_HEIGHT_KM = 36000.0  # GEO 轨道高度 (km)
N_BEAMS = 10               # 每星波束数 N_k (简化, 原文21)
N_CELLS = 100              # 每星可视小区数 N_v (简化, 原文441)
TOTAL_CELLS = N_SATELLITES * N_CELLS  # 总小区数

# ========== 频段参数 ==========
FREQ_GHZ = 20.0            # Ka-band 下行频率 (GHz)
FREQ_HZ = FREQ_GHZ * 1e9   # Hz
TOTAL_BW_HZ = 1e9          # 总带宽 1 GHz
N_SUBBANDS = 5             # 子频带数 N^{sub}
SUB_BW_HZ = TOTAL_BW_HZ / N_SUBBANDS  # 200 MHz

# ========== 功率参数 ==========
P_TOT_DBW = 33.0           # 卫星总功率 (dBW)
P_TOT_W = 10**(P_TOT_DBW/10)  # W
P_MAX_DBW = 30.0           # 单波束最大功率 (dBW)
P_MAX_W = 10**(P_MAX_DBW/10)
P_BEAM_DBW = 19.78         # 单波束发射功率 (dBW)
P_BEAM_W = 10**(P_BEAM_DBW/10)

# ========== 天线参数 (ITU-R) ==========
G_TRAN_MAX_DBI = 40.31     # 发射天线最大增益 (dBi)
G_REC_MAX_DBI = 31.53      # 接收天线最大增益 (dBi)
HPBW_DEG = 5.83            # 半功率波束宽度 (deg)
ANTENNA_A = 3.16           # ITU-R S.672-4 参数 a
ANTENNA_B = 6.32           # ITU-R S.672-4 参数 b
ANTENNA_LS_DB = -30.0      # 旁瓣限制 (dB)
REC_DIAMETER_M = 0.18      # 接收天线口径 (m)

# ========== 路径损耗 ==========
PATH_LOSS_DB = 209.59      # 路径损耗 (dB)

# ========== 干扰惩罚参数 ==========
ETA_PENALTY = 0.1          # 惩罚系数 η

# ========== 优化目标权重 ==========
W_T = 1.0                  # 吞吐量权重 ω_T
W_D = 1.0                  # 延迟偏移权重 ω_D
W_L = 1.0                  # 负载偏移权重 ω_L

# ========== 流量模型 ==========
TRAFFIC_LAMBDA_MIN = 50    # 泊松分布 λ 最小值 (Mbps)
TRAFFIC_LAMBDA_MAX = 500   # 泊松分布 λ 最大值 (Mbps)

# ========== QMIX 网络参数 ==========
QMIX_HIDDEN_DIM = 128      # 隐藏层维度 (简化, 原文256)
QMIX_LR = 0.0001           # 学习率
QMIX_GAMMA = 0.99          # 折扣因子
QMIX_EPSILON_START = 1.0   # ε-greedy 初始值
QMIX_EPSILON_END = 0.05    # ε-greedy 终值
QMIX_EPSILON_DECAY = 500   # 衰减步数
QMIX_BUFFER_SIZE = 2000    # 经验回放池大小
QMIX_BATCH_SIZE = 32       # 批量大小
QMIX_TAU = 0.005           # 目标网络软更新系数
QMIX_N_EPOCHS = 300        # 训练轮数 (简化, 原文3500)
QMIX_N_STEPS_PER_EPOCH = 20  # 每轮步数
QMIX_GRU_HIDDEN_DIM = 64   # GRU 隐藏层维度

# ========== 仿真参数 ==========
N_TRAFFIC_POINTS = 10      # 流量需求点数
TRAFFIC_RANGE = (500, 3000)  # 流量需求范围 (Mbps)
N_RUNS = 1                 # 每个配置运行次数
N_TIMESLOTS = 100          # 每轮评估时隙数
N_TIMESLOTS = 100          # 每轮评估时隙数

# ========== 对比算法参数 ==========
GA_POP_SIZE = 50           # 遗传算法种群大小
GA_N_GENERATIONS = 100     # 遗传算法迭代代数
GA_MUTATION_RATE = 0.1     # 变异率

# ========== 绘图参数 ==========
FIGSIZE_SINGLE = (3.5, 2.8)     # 单栏图尺寸 (inches)
FIGSIZE_DOUBLE = (7.16, 3.5)    # 双栏图尺寸 (inches)
FONT_SIZE = 10                  # 字体大小 (pt)
MARKER_SIZE = 5                 # 标记大小
LINE_WIDTH = 1.5                # 数据线宽
AXIS_LINE_WIDTH = 0.8           # 坐标轴线宽
DPI = 300                       # 输出 DPI

# ========== 曲线样式 ==========
CURVE_STYLES = {
    'RBH':         {'color': '#7f7f7f', 'marker': 'o', 'linestyle': '-', 'label': 'RBH'},
    'PBH':         {'color': '#a0522d', 'marker': 's', 'linestyle': '--', 'label': 'PBH'},
    'DFBH':        {'color': '#2e8b57', 'marker': '^', 'linestyle': '-.', 'label': 'DFBH'},
    'GABH':        {'color': '#daa520', 'marker': 'D', 'linestyle': ':', 'label': 'GABH'},
    'SADRL-BH':    {'color': '#4169e1', 'marker': 'v', 'linestyle': '--', 'label': 'SADRL-BH'},
    'MADRL-BH':    {'color': '#ff6347', 'marker': '<', 'linestyle': '-.', 'label': 'MADRL-BH'},
    'CMSBH':       {'color': '#000000', 'marker': '*', 'linestyle': '-', 'label': 'CMSBH'},
    'LBGT-MSBH':   {'color': '#8b4513', 'marker': 'p', 'linestyle': '--', 'label': 'LBGT-MSBH'},
    'PH-MSBH':     {'color': '#4682b4', 'marker': 'h', 'linestyle': '-.', 'label': 'PH-MSBH'},
    'I-MSBH':      {'color': '#9acd32', 'marker': 'x', 'linestyle': ':', 'label': 'I-MSBH'},
    'CMARL-MSBH':  {'color': '#ba55d3', 'marker': '+', 'linestyle': '--', 'label': 'CMARL-MSBH'},
}