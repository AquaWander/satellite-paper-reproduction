# -*- coding: utf-8 -*-
"""
绘图模块 - IEEE期刊标准风格
生成 Fig.8 (收敛曲线), Fig.9 (NMSE vs SNR), Fig.13 (数据速率)
论文: Split Learning-Based Channel Prediction for 6G-Enabled LEO Satellite Systems
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

from config import (
    FIG_FONT_FAMILY, FIG_FONT_SIZE, FIG_DOUBLE_COL, FIG_SINGLE_COL,
    FIG_LINEWIDTH, FIG_AXIS_LINEWIDTH, FIG_DPI, OUTPUT_DIR,
    COLORS, MARKERS, LINESTYLES,
)


def setup_ieee_style():
    """配置IEEE期刊标准绘图风格"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = [FIG_FONT_FAMILY, 'Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = FIG_FONT_SIZE
    rcParams['axes.linewidth'] = FIG_AXIS_LINEWIDTH
    rcParams['axes.labelsize'] = FIG_FONT_SIZE
    rcParams['axes.titlesize'] = FIG_FONT_SIZE + 1
    rcParams['xtick.labelsize'] = FIG_FONT_SIZE - 1
    rcParams['ytick.labelsize'] = FIG_FONT_SIZE - 1
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.width'] = FIG_AXIS_LINEWIDTH
    rcParams['ytick.major.width'] = FIG_AXIS_LINEWIDTH
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['legend.fontsize'] = FIG_FONT_SIZE - 1
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.edgecolor'] = '0.8'
    rcParams['figure.dpi'] = FIG_DPI
    rcParams['savefig.dpi'] = FIG_DPI
    rcParams['savefig.bbox'] = 'tight'
    rcParams['mathtext.fontset'] = 'stix'


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Fig. 8: 收敛曲线
# ============================================================

def plot_convergence(results, save_name='fig8_convergence.png'):
    """
    绘制Fig.8: 三种训练范式收敛对比

    参数:
        results: {
            'offline': (epochs, nmse_list),
            'online': (epochs, nmse_list),
            'hybrid': (epochs, nmse_list),
        }
    """
    setup_ieee_style()
    ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE_COL)

    # 绘制三条曲线
    configs = [
        ('offline', 'Fully offline', COLORS['offline'], MARKERS['offline'], LINESTYLES['offline']),
        ('online', 'Fully online', COLORS['online'], MARKERS['online'], LINESTYLES['online']),
        ('hybrid', 'Proposed hybrid offline-online', COLORS['hybrid'], MARKERS['hybrid'], LINESTYLES['hybrid']),
    ]

    for key, label, color, marker, ls in configs:
        epochs, nmses = results[key]
        # 对NMSE做平滑处理
        nmses_smooth = smooth_curve(nmses, window=3)
        ax.plot(epochs, nmses_smooth, color=color, linewidth=FIG_LINEWIDTH,
                linestyle=ls, marker=marker, markersize=4, markevery=len(epochs)//8,
                label=label, fillstyle='none')

    ax.set_xlabel('Training epochs', fontsize=FIG_FONT_SIZE)
    ax.set_ylabel('NMSE loss (dB)', fontsize=FIG_FONT_SIZE)
    ax.set_xlim([0, 1500])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)

    # 添加收敛标注
    # 标注hybrid在~600 epochs收敛
    ax.axvline(x=600, color=COLORS['hybrid'], linestyle=':', alpha=0.5, linewidth=0.8)

    filepath = os.path.join(OUTPUT_DIR, save_name)
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {filepath}")

    return filepath


# ============================================================
# Fig. 9: NMSE vs SNR (不同TDD模式和模型)
# ============================================================

def plot_nmse_vs_snr(nmse_results, save_name='fig9_nmse_vs_snr.png'):
    """
    绘制Fig.9: 不同TDD模式NMSE性能对比

    参数:
        nmse_results: {
            'DSUUU': {
                'Proposed (M1)': (snr_list, nmse_list),
                'CNN-LSTM [7]': (snr_list, nmse_list),
                'LSTM-only [29]': (snr_list, nmse_list),
            },
            'DSUUD': { ... }
        }
    """
    setup_ieee_style()
    ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE_COL)

    # 曲线配置
    curves = [
        # (tdd_mode, model_name, label, color, marker, linestyle)
        ('DSUUU', 'Proposed (M1)', 'Proposed (DSUUU)', COLORS['proposed_dsuuu'],
         MARKERS['proposed'], LINESTYLES['dsuuu']),
        ('DSUUU', 'CNN-LSTM [7]', 'CNN-LSTM [7] (DSUUU)', COLORS['cnn_lstm_dsuuu'],
         MARKERS['cnn_lstm'], LINESTYLES['dsuuu']),
        ('DSUUU', 'LSTM-only [29]', 'LSTM-only [29] (DSUUU)', COLORS['lstm_dsuuu'],
         MARKERS['lstm'], LINESTYLES['dsuuu']),
        ('DSUUD', 'Proposed (M1)', 'Proposed (DSUUD)', COLORS['proposed_dsuud'],
         MARKERS['proposed'], LINESTYLES['dsuud']),
        ('DSUUD', 'CNN-LSTM [7]', 'CNN-LSTM [7] (DSUUD)', COLORS['cnn_lstm_dsuud'],
         MARKERS['cnn_lstm'], LINESTYLES['dsuud']),
        ('DSUUD', 'LSTM-only [29]', 'LSTM-only [29] (DSUUD)', COLORS['lstm_dsuud'],
         MARKERS['lstm'], LINESTYLES['dsuud']),
    ]

    for tdd_mode, model_name, label, color, marker, ls in curves:
        if tdd_mode in nmse_results and model_name in nmse_results[tdd_mode]:
            snr_list, nmse_list = nmse_results[tdd_mode][model_name]
            ax.plot(snr_list, nmse_list, color=color, linewidth=FIG_LINEWIDTH,
                    linestyle=ls, marker=marker, markersize=5, markevery=2,
                    label=label, fillstyle='none')

    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=FIG_FONT_SIZE)
    ax.set_ylabel('NMSE (dB)', fontsize=FIG_FONT_SIZE)
    ax.set_xlim([-10, 20])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=8)

    filepath = os.path.join(OUTPUT_DIR, save_name)
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {filepath}")

    return filepath


# ============================================================
# Fig. 13: 数据速率对比
# ============================================================

def plot_data_rate(rate_results, save_name='fig13_data_rate.png'):
    """
    绘制Fig.13: 数据速率对比

    参数:
        rate_results: {
            'DSUUU': {
                'Proposed (M1)': (snr_list, rate_list),
                'CNN-LSTM [7]': (snr_list, rate_list),
            },
            'DSUUD': { ... }
        }
    """
    setup_ieee_style()
    ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE_COL)

    # 只绘制Proposed和CNN-LSTM [7] (论文Fig.13只有4条曲线)
    curves = [
        ('DSUUU', 'Proposed (M1)', 'Proposed (DSUUU)', COLORS['proposed_dsuuu'],
         MARKERS['proposed'], LINESTYLES['dsuuu']),
        ('DSUUU', 'CNN-LSTM [7]', 'CNN-LSTM [7] (DSUUU)', COLORS['cnn_lstm_dsuuu'],
         MARKERS['cnn_lstm'], LINESTYLES['dsuuu']),
        ('DSUUD', 'Proposed (M1)', 'Proposed (DSUUD)', COLORS['proposed_dsuud'],
         MARKERS['proposed'], LINESTYLES['dsuud']),
        ('DSUUD', 'CNN-LSTM [7]', 'CNN-LSTM [7] (DSUUD)', COLORS['cnn_lstm_dsuud'],
         MARKERS['cnn_lstm'], LINESTYLES['dsuud']),
    ]

    for tdd_mode, model_name, label, color, marker, ls in curves:
        if tdd_mode in rate_results and model_name in rate_results[tdd_mode]:
            snr_list, rate_list = rate_results[tdd_mode][model_name]
            ax.plot(snr_list, rate_list, color=color, linewidth=FIG_LINEWIDTH,
                    linestyle=ls, marker=marker, markersize=5, markevery=2,
                    label=label, fillstyle='none')

    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=FIG_FONT_SIZE)
    ax.set_ylabel('Data rate (bps/Hz)', fontsize=FIG_FONT_SIZE)
    ax.set_xlim([-10, 20])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)

    filepath = os.path.join(OUTPUT_DIR, save_name)
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {filepath}")

    return filepath


# ============================================================
# 辅助函数
# ============================================================

def smooth_curve(data, window=5):
    """平滑曲线 (移动平均)"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='same')
    # 边界处理
    for i in range(window // 2):
        smoothed[i] = np.mean(data[:i + window // 2 + 1])
        smoothed[-(i + 1)] = np.mean(data[-(i + window // 2 + 1):])
    return smoothed


if __name__ == '__main__':
    # 测试绘图
    print("绘图模块测试")

    setup_ieee_style()
    ensure_output_dir()

    # 测试收敛曲线
    test_results = {
        'offline': (list(range(0, 1500, 10)),
                    [-5 + 15 * (1 - np.exp(-x/300)) for x in range(150)]),
        'online': (list(range(0, 1500, 10)),
                   [-3 + 13 * (1 - np.exp(-x/500)) for x in range(150)]),
        'hybrid': (list(range(0, 1500, 10)),
                   [-8 + 14 * (1 - np.exp(-x/150)) for x in range(150)]),
    }
    plot_convergence(test_results, 'test_convergence.png')
    print("测试完成!")
