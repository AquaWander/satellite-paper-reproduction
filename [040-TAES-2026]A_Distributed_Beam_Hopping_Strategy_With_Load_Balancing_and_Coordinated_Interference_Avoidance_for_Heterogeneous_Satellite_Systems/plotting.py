# -*- coding: utf-8 -*-
"""绘图模块 (IEEE 风格)。"""
import os
import numpy as np
import config as C
from matplotlib import pyplot as plt


def _style_cycle(n):
    return zip(C.COLORS[:n], C.MARKERS[:n], C.LINESTYLES[:n])


def plot_tableIII(results, savepath):
    """Table III 负载差分组柱状图。"""
    n_groups = len(C.N_LEO_SWEEP)
    x = np.arange(n_groups)
    width = 0.35
    fig, ax = plt.subplots(figsize=C.FIG_DOUBLE)
    rw = results["RW-LB"]
    dist = results["Distance-based"]
    b1 = ax.bar(x - width / 2, rw, width, label="RW-LB",
                color=C.COLORS[0], edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + width / 2, dist, width, label="Distance-based",
                color=C.COLORS[1], edgecolor='black', linewidth=0.5)
    # 论文值折线叠加
    ax.plot(x, C.TABLEIII_PAPER["RW-LB"], 'o--', color=C.COLORS[0],
            alpha=0.5, markersize=4, label="RW-LB (paper)")
    ax.plot(x, C.TABLEIII_PAPER["Distance-based"], 's--', color=C.COLORS[1],
            alpha=0.5, markersize=4, label="Distance (paper)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in C.N_LEO_SWEEP])
    ax.set_xlabel("Number of LEO Satellites")
    ax.set_ylabel(r"Load Disparity $L_{df}$")
    ax.set_title("Table III: Load Disparity vs Number of LEOs (640 Mbps)")
    ax.legend(framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=':')
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_fig05(util_rwlb, util_dist, savepath):
    """Fig.5: 15 LEO 利用率, RW-LB vs Distance."""
    n = len(util_rwlb)
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=C.FIG_DOUBLE)
    ax.plot(x, util_rwlb, '-o', color=C.COLORS[0], label="RW-LB")
    ax.plot(x, util_dist, '-s', color=C.COLORS[1], label="Distance-based")
    ax.set_xlabel("LEO Satellite Index")
    ax.set_ylabel(r"Capacity Utilization $L_i^u$")
    ax.set_title("Fig. 5: Load Balancing (15 LEOs, 640 Mbps)")
    ax.set_xticks(x)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def _plot_sweep_subfig(ax, x, data_by_scheme, ylabel, title):
    for k, (scheme, vals) in enumerate(data_by_scheme.items()):
        col, mk, ls = list(_style_cycle(len(data_by_scheme)))[k]
        ax.plot(x, vals, linestyle=ls, marker=mk, color=col, label=scheme)
    ax.set_xlabel("Traffic Demand (Mbps)" if "Traffic" in title or "demand" in ylabel.lower()
                  else "Number of LEO Satellites")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(framealpha=0.9, loc="best")


def plot_fig07(sweep_data, savepath):
    """Fig.7: 业务需求扫描 (4 子图, 6 方案)。"""
    x = C.DEMAND_SWEEP
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.5))
    metric_spec = [
        ("rt_throughput", r"RT Throughput (Mbps)", "(a) RT Throughput"),
        ("total_throughput", r"Total Throughput (Mbps)", "(b) Total Throughput"),
        ("rt_satisf", r"RT Satisfaction Rate (%)", "(c) RT Satisfaction"),
        ("total_satisf", r"Total Satisfaction Rate (%)", "(d) Total Satisfaction"),
    ]
    for ax, (m, ylab, title) in zip(axs.ravel(), metric_spec):
        d = {s: [v * 100 if "satisf" in m else v for v in sweep_data[s][m]] for s in sweep_data}
        _plot_sweep_subfig(ax, x, d, ylab, title)
        ax.set_xlabel("Total Traffic Demand (Mbps)")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_fig08(sweep_data, savepath):
    """Fig.8: 卫星数扫描 (4 子图, 6 方案)。"""
    x = C.N_LEO_SWEEP
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.5))
    metric_spec = [
        ("rt_throughput", r"RT Throughput (Mbps)", "(a) RT Throughput"),
        ("total_throughput", r"Total Throughput (Mbps)", "(b) Total Throughput"),
        ("rt_satisf", r"RT Satisfaction Rate (%)", "(c) RT Satisfaction"),
        ("total_satisf", r"Total Satisfaction Rate (%)", "(d) Total Satisfaction"),
    ]
    for ax, (m, ylab, title) in zip(axs.ravel(), metric_spec):
        d = {s: [v * 100 if "satisf" in m else v for v in sweep_data[s][m]] for s in sweep_data}
        _plot_sweep_subfig(ax, x, d, ylab, title)
        ax.set_xlabel("Number of LEO Satellites")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
