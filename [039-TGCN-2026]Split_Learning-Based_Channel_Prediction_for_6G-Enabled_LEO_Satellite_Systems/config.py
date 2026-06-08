# -*- coding: utf-8 -*-
"""
仿真参数配置文件
论文: Split Learning-Based Channel Prediction for 6G-Enabled LEO Satellite Systems
期刊: IEEE TGCN, Vol. 10, 2026
"""

import numpy as np

# ============================================================
# 卫星系统参数 (Table II)
# ============================================================
SATELLITE_ALTITUDE = 600e3        # 卫星高度 A_S = 600 km
EARTH_RADIUS = 6371e3             # 地球半径
CARRIER_FREQ = 20e9               # 载波频率 f_c = 20 GHz (Ka频段)
SPEED_OF_LIGHT = 3e8              # 光速 c
ELEVATION_ANGLE = 90              # 仰角 epsilon = 90度 (默认)

# 天线参数
N_ANTENNA = 256                   # 天线阵元数 N_a
N_RF = 12                         # RF链数 N_RF

# 用户参数
N_CLUSTERS = 5                    # 用户簇数 C
CLUSTER_RADIUS = 10e3             # 簇半径 r_c = 10 km
USERS_PER_CLUSTER = 100           # 每簇用户数 M_c

# OFDM参数
N_SYMBOLS = 14                    # OFDM符号数/时隙 N_S
N_SUBCARRIERS = 300               # 子载波数 N_SC
N_DMRS = 2                        # DMRS符号数 N_p
N_TIME_STEPS = 5                  # 输入时间步数 N_U (默认)
N_CH = 2                          # 信道分量 N_CH (实/虚部)

# TDD模式
# DSUUU: 1下行 + 1特殊 + 3上行 = 更多上行数据用于预测
# DSUUD: 1下行 + 1特殊 + 2上行 + 1下行 = 较少上行数据
TDD_MODES = {
    'DSUUU': {'downlink': 1, 'special': 1, 'uplink': 3},
    'DSUUD': {'downlink': 2, 'special': 1, 'uplink': 2},
}

# ============================================================
# 神经网络参数 (Table III)
# ============================================================
CONV1_FILTERS = 16                 # Conv1滤波器数
CONV2_FILTERS = 32                 # Conv2滤波器数
CONV_KERNEL_SIZE = 3               # 卷积核大小 3x3
LSTM_UNITS = 512                   # LSTM单元数
DROPOUT_RATE = 0.25                # Dropout率
DENSE1_UNITS = 1024                # Dense1单元数

# 输入输出维度
INPUT_SHAPE = (N_TIME_STEPS, N_DMRS, N_SUBCARRIERS, N_CH)  # (5, 2, 300, 2)
OUTPUT_DIM = N_DMRS * N_SUBCARRIERS * N_CH                   # = 1200

# ============================================================
# 训练参数
# ============================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS_CONVERGENCE = 1500       # 收敛实验最大epoch数

# SNR范围
SNR_RANGE_DB = np.arange(-10, 21, 2)  # -10 ~ 20 dB, 步进2
SNR_EVAL_DB = np.arange(-10, 21, 2)   # 评估用SNR点

# ============================================================
# 发射功率与噪声
# ============================================================
P_TRANSMIT_DBM = 43               # 发射功率 43 dBm (卫星)
NOISE_FIGURE_DB = 7               # 接收机噪声系数 7 dB
THERMAL_NOISE_DBM = -174          # 热噪声功率谱密度 -174 dBm/Hz
BANDWIDTH = 100e6                  # 系统带宽 100 MHz

# ============================================================
# 信道模型参数 (3GPP TR 38.811 TDL-D简化)
# ============================================================
# Rician K因子
RICIAN_K_DB = 10                   # K因子 10 dB
# 多普勒参数
SATELLITE_VELOCITY = 7.56e3       # LEO卫星速度 ~7.56 km/s

# TDL-D 延迟扩展 (简化)
TDL_D_DELAYS = np.array([0, 1e-9, 2e-9, 5e-9, 8e-9, 14e-9])  # 简化的6抽头延迟
TDL_D_POWERS_DB = np.array([0, -0.2, -0.8, -2.0, -3.5, -5.2]) # 简化功率profile

# ============================================================
# 绘图参数 (IEEE期刊标准)
# ============================================================
FIG_FONT_FAMILY = 'Times New Roman'
FIG_FONT_SIZE = 10
FIG_SINGLE_COL = (3.5, 2.8)       # 单栏图尺寸 (英寸)
FIG_DOUBLE_COL = (7.16, 3.5)      # 双栏图尺寸 (英寸)
FIG_LINEWIDTH = 1.5               # 数据线宽
FIG_AXIS_LINEWIDTH = 0.8          # 坐标轴线宽
FIG_DPI = 300                      # 输出DPI
OUTPUT_DIR = 'output'

# 颜色方案 (IEEE期刊常用)
COLORS = {
    'proposed_dsuuu': '#0072BD',   # 蓝色 - 本文方法 DSUUU
    'proposed_dsuud': '#D95319',   # 橙色 - 本文方法 DSUUD
    'cnn_lstm_dsuuu': '#EDB120',   # 黄色 - CNN-LSTM [7] DSUUU
    'cnn_lstm_dsuud': '#7E2F8E',   # 紫色 - CNN-LSTM [7] DSUUD
    'lstm_dsuuu': '#77AC30',       # 绿色 - LSTM [29] DSUUU
    'lstm_dsuud': '#A2142F',       # 红色 - LSTM [29] DSUUD
    'offline': '#0072BD',          # 蓝色 - 全离线
    'online': '#D95319',           # 橙色 - 全在线
    'hybrid': '#77AC30',           # 绿色 - 混合
}

# 标记样式
MARKERS = {
    'proposed': 'o',
    'cnn_lstm': 's',
    'lstm': '^',
    'offline': 'D',
    'online': 'v',
    'hybrid': 'o',
}

# 线型
LINESTYLES = {
    'dsuuu': '-',
    'dsuud': '--',
    'offline': '-',
    'online': '--',
    'hybrid': '-.',
}

# 随机种子
RANDOM_SEED = 42
