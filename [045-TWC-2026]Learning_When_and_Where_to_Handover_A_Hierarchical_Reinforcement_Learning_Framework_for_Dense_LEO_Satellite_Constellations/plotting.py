# -*- coding: utf-8 -*-
"""
plotting.py
================================================================================
IEEE 期刊风格绘图模块。
    - Fig.6：双栏尺寸 2×3 子图布局（成功率×3 + 频次×3）
    - Fig.7：单栏尺寸，3 纬度点 × 5 算法
    - Fig.9：单栏尺寸，可见卫星数 3→20 × 5 算法

5 算法固定顺序 + color+marker+linestyle 三重区分。
================================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

from config import OUTPUT_DIR, ALGORITHMS

# ============================================================================
# IEEE 风格全局设置
# ============================================================================
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

# 单栏 / 双栏尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 5 算法固定顺序：全实线 + 颜色 + 标记三重区分（线型统一实线，与原文一致）
#   MO-HRL 蓝●圆 / MO-RL 橙■方 / SL-RL 黄▲三角 / SQ-RL 紫◆菱形 / ST-RL 绿▼倒三角
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '-', '-', '-', '-']  # 全实线：靠颜色+标记区分

# 算法 → 视觉索引（按 ALGORITHMS 顺序）
def _style_for(algo_name):
    idx = next(i for i, a in enumerate(ALGORITHMS) if a['name'] == algo_name)
    return COLORS[idx], MARKERS[idx], LINESTYLES[idx]


def _fill_nan(y):
    """线性插值填补内部 nan（首尾 nan 保留），让曲线连续。"""
    y = np.array(y, dtype=float)
    n = len(y)
    idx = np.arange(n)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return y
    y_filled = np.interp(idx, idx[valid], y[valid])
    # 首尾 nan 保留
    if not valid[0]:
        y_filled[:np.argmax(valid)] = np.nan
    if not valid[-1]:
        last_valid = n - 1 - np.argmax(valid[::-1])
        y_filled[last_valid + 1:] = np.nan
    return y_filled


def _plot_curves(ax, curves, x):
    """在 ax 上绘制 5 条曲线（全实线 + 颜色 + 标记三重区分，nan 插值连续）。"""
    for algo_name, y in curves.items():
        c, m, ls = _style_for(algo_name)
        y_plot = _fill_nan(y)
        ax.plot(x, y_plot, color=c, marker=m, linestyle=ls,
                label=algo_name, markersize=rcParams['lines.markersize'],
                linewidth=rcParams['lines.linewidth'],
                markevery=max(1, len(x) // 12))  # 标记稀疏化避免拥挤


# ============================================================================
# Fig.5：空间层 DQN 训练曲线（4 权重配置）
# ============================================================================
def plot_fig5(curves_data, savepath=None, smooth_window=15):
    """
    绘制 Fig.5（4 条 reward-vs-episode 曲线，4 权重配置）。

    curves_data : dict，train_rl.run_fig5 返回的 dict，含:
        - tags: list[str]
        - curves: {tag: [reward_per_episode]}
        - weights: list[(wL,wH,wT)]
    smooth_window : 滑动均值窗口（展示收敛趋势，原始曲线过抖）

    配色：均衡权重(0.4,0.3,0.3) 用 MO-HRL 的蓝色突出（论文中收敛最高）。
    """
    if savepath is None:
        savepath = os.path.join(OUTPUT_DIR, 'fig05_spatial_dqn_training.png')

    tags = curves_data['tags']
    curves = curves_data['curves']
    weights = curves_data['weights']

    # 4 条曲线配色：均衡权重用 MO-HRL 蓝，其余用调色板的橙/黄/紫
    palette = ['#D95319', '#EDB120', '#7E2F8E', '#0072BD']  # 橙黄紫蓝
    markers = ['s', '^', 'D', 'o']

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i, tag in enumerate(tags):
        w = weights[i]
        y = np.array(curves[tag], dtype=float)
        n = len(y)
        x = np.arange(1, n + 1)
        # 滑动均值平滑
        if smooth_window > 1 and n > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            y_smooth = np.convolve(y, kernel, mode='valid')
            x_smooth = np.arange(smooth_window, n + 1)
        else:
            y_smooth = y; x_smooth = x
        # 标签：均衡权重特别标注
        is_eq = (abs(w[0]-0.4) < 1e-3 and abs(w[1]-0.3) < 1e-3
                 and abs(w[2]-0.3) < 1e-3)
        label = (f'w=(0.4,0.3,0.3)' if is_eq
                 else f'w=({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})')
        lw = 2.0 if is_eq else 1.4
        ax.plot(x_smooth, y_smooth, color=palette[i], linestyle='-',
                linewidth=lw, label=label)
        # 标记稀疏化（在曲线上叠 5 个点）
        n_pts = len(x_smooth)
        if n_pts > 5:
            mk_idx = np.linspace(0, n_pts - 1, 5).astype(int)
            ax.plot(x_smooth[mk_idx], y_smooth[mk_idx], color=palette[i],
                    marker=markers[i], linestyle='None',
                    markersize=rcParams['lines.markersize'])

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Episode reward')
    ax.set_title('Spatial-layer DQN convergence (Fig.5)')
    ax.grid(True, which='both', linestyle='--', alpha=rcParams['grid.alpha'])
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='0.5', fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ============================================================================
# Fig.4：时间层 PPO+GRU 训练曲线（3 reward_case）
# ============================================================================
def plot_fig4(curves_data, savepath=None, smooth_window=15):
    """
    绘制 Fig.4（3 条 reward-vs-episode 曲线，3 reward_case 配置）。

    curves_data : dict，train_rl.run_fig4 返回的 dict，含:
        - cases: list[str] ['case2','case23','case234']
        - curves: {case: [reward_per_episode]}
    smooth_window : 滑动均值窗口（展示收敛趋势）

    配色：case234（MO-HRL 完整版）用蓝色突出（论文中收敛最高），
          case23 用橙色，case2 用灰色（最低/最慢）。
    """
    if savepath is None:
        savepath = os.path.join(OUTPUT_DIR, 'fig04_temporal_ppo_gru_training.png')

    cases = curves_data['cases']
    curves = curves_data['curves']

    # 3 条曲线配色：case234 用 MO-HRL 蓝突出，case23 橙，case2 灰
    # 按 case234 → case23 → case2 顺序配色（论文收敛高→低）
    palette_map = {'case234': '#0072BD',   # 蓝（MO-HRL，最高）
                   'case23':  '#D95319',   # 橙
                   'case2':   '#7F7F7F'}   # 灰（最低/最慢）
    markers_map = {'case234': 'o',
                   'case23':  's',
                   'case2':   '^'}
    # 标签
    label_map = {'case234': r'$\mathrm{Case\,1+2+3+4}$ (full, MO-HRL)',
                 'case23':  r'$\mathrm{Case\,1+2+3}$',
                 'case2':   r'$\mathrm{Case\,1+2}$'}
    # 绘制顺序：case2 → case23 → case234（让 case234 蓝色在最上层）
    draw_order = [c for c in ['case2', 'case23', 'case234'] if c in curves]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for case in draw_order:
        y = np.array(curves[case], dtype=float)
        n = len(y)
        x = np.arange(1, n + 1)
        if smooth_window > 1 and n > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            y_smooth = np.convolve(y, kernel, mode='valid')
            x_smooth = np.arange(smooth_window, n + 1)
        else:
            y_smooth = y; x_smooth = x
        is_full = (case == 'case234')
        lw = 2.0 if is_full else 1.4
        ax.plot(x_smooth, y_smooth, color=palette_map.get(case, '#777'),
                linestyle='-', linewidth=lw, label=label_map.get(case, case))
        n_pts = len(x_smooth)
        if n_pts > 5:
            mk_idx = np.linspace(0, n_pts - 1, 5).astype(int)
            ax.plot(x_smooth[mk_idx], y_smooth[mk_idx],
                    color=palette_map.get(case, '#777'),
                    marker=markers_map.get(case, 'o'), linestyle='None',
                    markersize=rcParams['lines.markersize'])

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Episode reward')
    ax.set_title('Temporal-layer PPO+GRU convergence (Fig.4)')
    ax.grid(True, which='both', linestyle='--', alpha=rcParams['grid.alpha'])
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='0.5', fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ============================================================================
# Fig.6：6 子图（2×3）
# ============================================================================
def plot_fig6(data, savepath=None):
    """
    绘制 Fig.6（2×3 子图布局）。
    data: run_fig6() 返回的 dict。
    """
    if savepath is None:
        savepath = os.path.join(OUTPUT_DIR, 'fig06_handover_factors.png')

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.5))
    keys = ['a', 'b', 'c', 'd', 'e', 'f']
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for key, (r, c) in zip(keys, positions):
        ax = axes[r, c]
        d = data[key]
        _plot_curves(ax, d['curves'], d['x'])
        ax.set_xlabel(d['xlabel'])
        ax.set_ylabel(d['ylabel'])
        ax.set_title(d['title'])
        ax.grid(True, which='both', linestyle='--', alpha=rcParams['grid.alpha'])
        # Y 轴范围：成功率 [0,1]，频次 [0,1]（频次=切换/时隙占比）
        if 'success' in d['ylabel'].lower():
            ax.set_ylim(0, 1.05)
        elif 'frequency' in d['ylabel'].lower():
            ax.set_ylim(0, 1.0)
        # 图例只在 (a) 子图显示一次
        if key == 'a':
            ax.legend(loc='lower right', framealpha=0.9, edgecolor='0.5')

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ============================================================================
# 通用分组柱状图绘制工具
# ============================================================================
def _plot_grouped_bars(ax, curves, x_labels, ylabel, group_label=None,
                       bar_value_fmt=None, ylim=None, legend_loc='upper right'):
    """
    在 ax 上绘制分组柱状图。
      curves      : dict {algo_name: np.array[len(groups)]}
      x_labels    : list[str], 每组的 X 轴标签
      ylabel      : str
      bar_value_fmt: None 或格式串（如 '%.2f'），若给定则在柱顶标数值
      ylim        : (lo, hi) 或 None
      legend_loc  : 图例位置
    """
    algo_names = list(curves.keys())            # 按 ALGORITHMS 固定顺序
    n_algo = len(algo_names)
    n_group = len(x_labels)
    x = np.arange(n_group)
    group_width = 0.82
    bar_width = group_width / n_algo

    for i, name in enumerate(algo_names):
        c, _, _ = _style_for(name)
        offset = (i - (n_algo - 1) / 2.0) * bar_width
        bars = ax.bar(x + offset, curves[name], bar_width,
                      color=c, edgecolor='black', linewidth=0.4,
                      label=name)
        if bar_value_fmt is not None:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2.0, h,
                        bar_value_fmt % h,
                        ha='center', va='bottom',
                        fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(group_label if group_label else '')
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, axis='y', linestyle='--', alpha=rcParams['grid.alpha'])
    ax.legend(loc=legend_loc, framealpha=0.9, edgecolor='0.5',
              ncol=1, fontsize=8)
    return ax


# ============================================================================
# Fig.7：分组柱状图（3 纬度组 × 5 算法）
# ============================================================================
def plot_fig7(data, savepath=None):
    """
    绘制 Fig.7 分组柱状图：3 个纬度组（Low/Mid/Mid-High），每组 5 根柱（5 算法）。
    Y 轴 = 切换成功率 [0,1]。
    """
    if savepath is None:
        savepath = os.path.join(OUTPUT_DIR, 'fig07_success_rate_latitude.png')

    # 用 "Low [0°N,2°N]" 长标签会拥挤，柱状图用短标签
    short_labels = ['Low', 'Mid', 'Mid-High']

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    _plot_grouped_bars(
        ax, data['curves'],
        x_labels=short_labels,
        ylabel=data['ylabel'],
        group_label='Latitude region',
        bar_value_fmt='%.2f',
        ylim=(0.0, 1.08),
        legend_loc='upper left',
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ============================================================================
# Fig.9：分组柱状图（5 可见卫星数组 × 5 算法）
# ============================================================================
def plot_fig9(data, savepath=None):
    """
    绘制 Fig.9 分组柱状图：5 个可见卫星数组（4/8/12/16/20），每组 5 根柱。
    Y 轴 = 传输速率 (Mbps)。
    """
    if savepath is None:
        savepath = os.path.join(OUTPUT_DIR, 'fig09_transmission_rate.png')

    x_labels = [str(int(s)) for s in data['x']]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _plot_grouped_bars(
        ax, data['curves'],
        x_labels=x_labels,
        ylabel=data['ylabel'],
        group_label='Number of visible satellites',
        bar_value_fmt='%.0f',
        ylim=(0.0, None),
        legend_loc='upper right',
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


if __name__ == '__main__':
    import simulation as sim
    print('[plot] 绘制 Fig.6 ...')
    p6 = plot_fig6(sim.run_fig6())
    print('[plot] 绘制 Fig.7 ...')
    p7 = plot_fig7(sim.run_fig7())
    print('[plot] 绘制 Fig.9 ...')
    p9 = plot_fig9(sim.run_fig9())
    print('[plot] 完成:')
    print('  ', p6)
    print('  ', p7)
    print('  ', p9)
