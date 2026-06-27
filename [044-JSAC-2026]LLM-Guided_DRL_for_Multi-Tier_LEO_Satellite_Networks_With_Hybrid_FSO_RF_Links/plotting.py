"""
plotting.py
=========
IEEE 期刊风格绘图.

生成:
  - fig03a_convergence_reward.png   6 算法 episodic reward vs episode  (Fig.3a)
  - fig03b_convergence_handover.png 6 算法 切换次数 vs episode         (Fig.3b)
  - fig03c_llm_convergence.png      5 LLM episodic reward vs episode   (Fig.3c)
  - fig04_avg_performance.png       6 算法 双 y 轴柱状图 (f1, f2)      (Fig.4)

风格: Times New Roman 10pt, 单栏 (3.5, 2.8) / 双栏 (7.16, 3.5),
      线宽 1.5pt, 轴 0.8pt, 刻度朝内, 三重区分(颜色+标记+线型), 300 dpi.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import config as C


# ============================================================================
# 全局 IEEE 风格
# ============================================================================
def _set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _smooth(y, window=9):
    """滑动平均平滑 (保留原始趋势). window 为奇数."""
    if len(y) < window:
        return y
    half = window // 2
    out = np.convolve(y, np.ones(window) / window, mode="same")
    out[:half] = y[:half]
    out[-half:] = y[-half:]
    return out


def _style_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.5)


# ============================================================================
# Fig.3(a)/(b): 6 算法收敛曲线
# ============================================================================
def plot_convergence(results, ylabel, filename, title=None,
                     figsize=(3.5, 2.8), smooth_win=11):
    """
    results: dict[algo_name] = np.ndarray (n_seeds, n_episodes) 跨种子平均
    """
    _set_ieee_style()
    fig, ax = plt.subplots(figsize=figsize)
    algos = C.ALGO_LIST
    for i, algo in enumerate(algos):
        if algo not in results:
            continue
        data = results[algo]                      # (n_seeds, n_episodes)
        mean = data.mean(axis=0)
        sm = _smooth(mean, smooth_win)
        color = C.COLOR_CYCLE[i % len(C.COLOR_CYCLE)]
        marker = C.MARKER_CYCLE[i % len(C.MARKER_CYCLE)]
        ls = C.LS_CYCLE[i % len(C.LS_CYCLE)]
        x = np.arange(len(mean))
        ax.plot(x, sm, color=color, linestyle=ls, linewidth=1.5,
                label=algo)
        # 稀疏标记 (每 N 点一个, 辅助区分)
        n_mark = max(1, len(mean) // 8)
        ax.plot(x[::n_mark], sm[::n_mark], color=color, marker=marker,
                linestyle="None", markersize=3.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    _style_axes(ax)
    ax.legend(framealpha=0.9, loc="best", ncol=2, columnspacing=0.8,
              handlelength=2.0)
    fig.savefig(os.path.join(C.OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  saved {filename}")


# ============================================================================
# Fig.4: 双 y 轴柱状图 (f1 速率 / f2 切换)
# ============================================================================
def plot_avg_performance(f1_means, f2_means, filename, figsize=(4.6, 2.9)):
    """
    f1_means, f2_means: dict[algo] = float  (跨种子平均的末期均值)
    """
    _set_ieee_style()
    fig, ax1 = plt.subplots(figsize=figsize)
    algos = C.ALGO_LIST
    x = np.arange(len(algos))
    f1 = np.array([f1_means.get(a, 0.0) for a in algos])
    f2 = np.array([f2_means.get(a, 0.0) for a in algos])

    # 左 y: f1 (速率, 归一化速率和) — 蓝色
    bars1 = ax1.bar(x - 0.2, f1, width=0.4, color="#0072BD",
                    edgecolor="black", linewidth=0.6, label="f1 (Rate)")
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel(r"Average $f_1$ (normalized rate sum)")
    ax1.set_xticks(x)
    # 6 个算法名(含 "LTQC-DAM" 8 字符)旋转 30° 防重叠
    ax1.set_xticklabels(algos, rotation=30, ha="right")
    ax1.tick_params(axis="y", direction="in")
    for b, v in zip(bars1, f1):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7)

    # 右 y: f2 (切换次数) — 橙色
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, f2, width=0.4, color="#D95319",
                    edgecolor="black", linewidth=0.6, label="f2 (Handover)")
    ax2.set_ylabel(r"Average $f_2$ (handover count)")
    ax2.tick_params(axis="y", direction="in")
    for b, v in zip(bars2, f2):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}",
                 ha="center", va="bottom", fontsize=7)

    # 合并图例 (右下角). 置于 ax2(twinx 顶层) 上 + 高 zorder, 保证压在所有柱之上
    lines = [bars1, bars2]
    labels = [l.get_label() for l in lines]
    leg = ax2.legend(lines, labels, loc="lower right", framealpha=0.95,
                     facecolor="white", edgecolor="black", fancybox=False)
    leg.set_zorder(100)
    ax1.set_ylim(top=max(f1) * 1.15)
    ax2.set_ylim(top=max(f2) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  saved {filename}")


# ============================================================================
# Fig.3(c): 5 LLM 收敛曲线
# ============================================================================
def plot_llm_convergence(results, filename, figsize=(3.5, 2.8), smooth_win=11):
    """results: dict[llm_name] = (n_seeds, n_episodes)"""
    _set_ieee_style()
    fig, ax = plt.subplots(figsize=figsize)
    for i, llm in enumerate(C.LLM_LIST):
        if llm not in results:
            continue
        data = results[llm]
        mean = data.mean(axis=0)
        sm = _smooth(mean, smooth_win)
        color = C.COLOR_CYCLE[i % len(C.COLOR_CYCLE)]
        marker = C.MARKER_CYCLE[i % len(C.MARKER_CYCLE)]
        ls = C.LS_CYCLE[i % len(C.LS_CYCLE)]
        x = np.arange(len(mean))
        ax.plot(x, sm, color=color, linestyle=ls, linewidth=1.5, label=llm)
        n_mark = max(1, len(mean) // 8)
        ax.plot(x[::n_mark], sm[::n_mark], color=color, marker=marker,
                linestyle="None", markersize=3.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic reward")
    _style_axes(ax)
    ax.legend(framealpha=0.9, loc="best", ncol=2)
    fig.savefig(os.path.join(C.OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  saved {filename}")


# ============================================================================
# 辅助: 收敛曲线 + 阴影 (std) 版本 (可选, 更丰富)
# ============================================================================
def plot_convergence_with_std(results, ylabel, filename,
                              figsize=(3.5, 2.8), smooth_win=11):
    _set_ieee_style()
    fig, ax = plt.subplots(figsize=figsize)
    for i, algo in enumerate(C.ALGO_LIST):
        if algo not in results:
            continue
        data = results[algo]
        mean = _smooth(data.mean(axis=0), smooth_win)
        std = data.std(axis=0)
        color = C.COLOR_CYCLE[i % len(C.COLOR_CYCLE)]
        x = np.arange(len(mean))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
        ax.plot(x, mean, color=color, linewidth=1.5, label=algo)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    ax.legend(framealpha=0.9, loc="best", ncol=2)
    fig.savefig(os.path.join(C.OUTPUT_DIR, filename))
    plt.close(fig)
