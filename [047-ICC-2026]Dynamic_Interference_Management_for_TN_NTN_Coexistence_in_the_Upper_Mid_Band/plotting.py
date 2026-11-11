"""按原文图号输出；全部曲线来自仿真或实际训练日志。"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulation import METHODS


def style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 11, "legend.fontsize": 7, "figure.dpi": 300,
        "savefig.dpi": 300, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in", "grid.linewidth": 0.3,
        "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def save_cdf(data, path, kind):
    style()
    has_zero = any(np.any(row[method]["inr_db"] <= -299.99) for row in data.values() for method in METHODS)
    if has_zero:
        fig, (zero_ax, ax) = plt.subplots(1, 2, sharey=True, figsize=(7.16, 3.5),
                                         gridspec_kw={"width_ratios": [0.5, 6.5], "wspace": 0.06})
        zero_ax.set_xlim(-300.5, -299.5)
        zero_ax.set_xticks([-300], ["I=0"])
        zero_ax.set_ylabel("CDF")
        zero_ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
        # 断轴明确区分零干扰概率质量，避免-300dB数值floor压缩有限样本曲线。
        for target, xpos in ((zero_ax, 1.0), (ax, 0.0)):
            for ypos in (0.0, 1.0):
                target.plot([xpos-0.012, xpos+0.012], [ypos-0.012, ypos+0.012],
                            transform=target.transAxes, color="black", clip_on=False, linewidth=0.7)
    else:
        fig, ax = plt.subplots(figsize=(7.16, 3.5))
        ax.set_ylabel("CDF")
    colors = ["#0072BD", "#D95319", "#609225"]
    styles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    labels = ["Low density", "Medium density", "High density"] if kind == "density" else ["T1:T2=100:0", "T1:T2=50:50", "T1:T2=0:100"]
    for method_index, method in enumerate(METHODS):
        for scenario_index, (scenario, methods) in enumerate(data.items()):
            x = np.sort(methods[method]["inr_db"])
            y = np.arange(1, len(x) + 1) / len(x)
            if has_zero:
                count = np.count_nonzero(x <= -299.99)
                mass = count / len(x)
                zero_ax.plot([-300.4, -300, -300, -299.6], [0, 0, mass, mass],
                             color=colors[method_index], linestyle=styles[scenario_index], linewidth=1.2)
                keep = x > -299.99
                x, y = x[keep], y[keep]
                if len(x) == 0:
                    ax.plot([], [], color=colors[method_index], linestyle=styles[scenario_index],
                            marker=markers[scenario_index], label=f"{method}: {labels[scenario_index]}")
                    continue
            indices = np.unique(np.linspace(0, len(x) - 1, min(len(x), 700)).astype(int))
            ax.plot(x[indices], y[indices], color=colors[method_index],
                linestyle=styles[scenario_index], marker=markers[scenario_index],
                markersize=3, markevery=70, label=f"{method}: {labels[scenario_index]}")
    ax.set_xlabel("INR (dB)")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, framealpha=0.9, columnspacing=0.9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.31)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_active(data, path):
    style()
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    x = np.arange(len(data))
    for index, method in enumerate(METHODS[1:]):
        means = [np.mean(d[method]["active_per_seed"]) * 100 for d in data.values()]
        errors = [(np.std(d[method]["active_per_seed"], ddof=1) * 100
                   if len(d[method]["active_per_seed"]) > 1 else 0.0) for d in data.values()]
        ax.bar(x + (index - 0.5) * 0.34, means, 0.34, yerr=errors, capsize=3,
               label=method, color=["#0072BD", "#609225"][index])
    ax.set_xticks(x, ["Low density", "Medium density", "High density"])
    ax.set_ylabel("Active BS ratio (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_training(log_paths, path):
    style()
    fig, (ax, task_ax) = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True)
    traces = []
    task_rows = []
    for index, log in enumerate(log_paths):
        with Path(log).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        y = np.array([float(row["mean_reward"]) for row in rows])
        ax.plot(np.arange(1, len(y) + 1), y, alpha=0.22, linewidth=0.5, label=f"Seed {index + 1}")
        traces.append(y)
        task_rows.extend(rows)
    average = np.mean(np.array(traces), axis=0)
    window = min(16, len(average))
    smooth = np.convolve(average, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(average) + 1), smooth, color="#C43B2B", linewidth=1.5, label=f"Mean ({window}-ep)")
    labels = ["Low density, T1", "Medium density, T1", "High density, T1", "T1:T2=50:50", "All T2"]
    colors = ["#0072BD", "#D95319", "#609225", "#7E2F8E", "#AE8510"]
    for task in range(5):
        xs, ys = [], []
        for begin in range(0, len(average), 50):
            values = [float(r["mean_reward"]) for r in task_rows
                      if int(r.get("task_index", -1)) == task and begin < int(r["episode"]) <= begin + 50]
            if values:
                xs.append(begin + 25)
                ys.append(np.mean(values))
        task_ax.plot(xs, ys, color=colors[task], label=labels[task], marker="o", markersize=2)
    ax.set_title("Mixed-task episode rewards")
    task_ax.set_title("Per-task means (50-ep bins)")
    ax.set_xlabel("Training episode")
    task_ax.set_xlabel("Training episode")
    ax.set_ylabel("Reward per step")
    ax.grid(True)
    ax.legend(loc="best", framealpha=0.9)
    task_ax.grid(True)
    task_ax.legend(loc="best", framealpha=0.9, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_all(arrays, output, logs):
    save_training(logs, output / "fig03_ppo_training.png")
    save_cdf(arrays["density"], output / "fig04_inr_density.png", "density")
    save_cdf(arrays["mix"], output / "fig05_inr_user_types.png", "mix")
    save_active(arrays["density"], output / "fig06_active_bs_ratio.png")
    save_reference_comparison(output)


def save_reference_comparison(output):
    """展示逐曲线中位差距；参考仅用于评估，未进入控制器。"""
    style()
    with (output / "paper_comparison.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), sharex=True, sharey=True)
    all_values = [float(row[key]) for row in rows for key in ("paper_plotted_median_db", "model_median_db")]
    lo, hi = min(all_values) - 3, max(all_values) + 3
    grid = np.array([lo, hi])
    for ax, figure in zip(axes, ("fig4", "fig5")):
        ax.fill_between(grid, grid - 3, grid + 3, color="0.9", label="Within 3 dB")
        ax.plot(grid, grid, "k:", linewidth=0.8)
        for index, method in enumerate(METHODS):
            subset = [r for r in rows if r["figure"] == figure and r["method"] == method]
            ax.scatter([float(r["paper_plotted_median_db"]) for r in subset],
                       [float(r["model_median_db"]) for r in subset],
                       s=24, marker=["o", "s", "^"][index],
                       color=["#0072BD", "#D95319", "#609225"][index], label=method)
        ax.set_title("Fig. 4: densities" if figure == "fig4" else "Fig. 5: terminal types")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Paper median INR (dB)")
        ax.grid(True)
    axes[0].set_ylabel("Model median INR (dB)")
    axes[1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "fig07_reference_medians.png", dpi=300)
    plt.close(fig)
