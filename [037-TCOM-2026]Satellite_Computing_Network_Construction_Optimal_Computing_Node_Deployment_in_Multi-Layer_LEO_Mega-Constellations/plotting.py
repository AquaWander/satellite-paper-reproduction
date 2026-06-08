"""
plotting.py -- Plotting module (IEEE journal style)
Fig.4: N_C and delay vs network size
Fig.5: Delay comparison: LEO vs MEO computing nodes
Fig.6: N_C comparison across schemes
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import (
    COLORS, MARKERS, LINESTYLES,
    FIG_SINGLE, FIG_DOUBLE,
    HOP_RANGE, NETWORK_SIZES, DEFAULT_L,
)


def plot_fig4(nc_data, delay_data, output_dir):
    """
    Fig.4 combined: (a) N_C vs network size + (b) delay vs network size
    """
    fig, (ax1_left, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # === (a) Computing nodes vs network size ===
    for idx, J in enumerate(HOP_RANGE):
        J = int(J)
        nc = nc_data["nc_data"][J]
        ax1_left.plot(NETWORK_SIZES, nc,
                      color=COLORS[idx], marker=MARKERS[idx],
                      linestyle=LINESTYLES[idx], markersize=5,
                      label=f"$J={J}$")

    ax1_left.set_xlabel("Network size $N$ ($N=M$)")
    ax1_left.set_ylabel("Number of computing nodes $N_C$")
    ax1_left.grid(True, alpha=0.3)
    ax1_left.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax1_left.set_title("(a) Computing nodes vs. network size")

    # === (b) Average delay vs network size ===
    for idx, J in enumerate(HOP_RANGE):
        J = int(J)
        delay = delay_data["delay_data"][J]
        ax2.plot(NETWORK_SIZES, delay,
                 color=COLORS[idx], marker=MARKERS[idx],
                 linestyle=LINESTYLES[idx], markersize=5,
                 label=f"$J={J}$")

    ax2.set_xlabel("Network size $N$ ($N=M$)")
    ax2.set_ylabel("Signaling distribution delay (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax2.set_title("(b) Average delay vs. network size")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig4.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Fig.4 saved to {output_dir}/fig4.png")


def plot_fig5(delay_comparison, output_dir):
    """
    Fig.5: Signaling distribution delay comparison
    LEO Computing Node vs MEO Computing Node
    Each method shows max/min/avg delay band
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    J_vals = delay_comparison["J_values"]
    methods = delay_comparison["methods"]

    method_styles = {
        "LEO Computing Node": {
            "color": COLORS[0], "marker": MARKERS[0],
            "ls_avg": "-", "ls_range": "--",
        },
        "MEO Computing Node": {
            "color": COLORS[1], "marker": MARKERS[1],
            "ls_avg": "-", "ls_range": "--",
        },
    }

    for method_name, data in methods.items():
        style = method_styles.get(method_name, {
            "color": COLORS[0], "marker": MARKERS[0],
            "ls_avg": "-", "ls_range": "--",
        })
        color = style["color"]
        marker = style["marker"]

        # Average delay (solid line with markers)
        ax.plot(J_vals, data["avg"],
                color=color, marker=marker, linestyle=style["ls_avg"],
                markersize=6, linewidth=1.5,
                label=f"{method_name} (Avg)")

        # Max delay (dashed, smaller markers)
        ax.plot(J_vals, data["max"],
                color=color, marker=marker, linestyle=style["ls_range"],
                markersize=4, linewidth=1.0, alpha=0.7,
                label=f"{method_name} (Max)")

        # Min delay (dotted, smaller markers)
        ax.plot(J_vals, data["min"],
                color=color, marker=marker, linestyle=":",
                markersize=4, linewidth=1.0, alpha=0.7,
                label=f"{method_name} (Min)")

        # Shaded band between min and max
        ax.fill_between(J_vals, data["min"], data["max"],
                        color=color, alpha=0.1)

    ax.set_xlabel("Reachable hop count $J$")
    ax.set_ylabel("Signaling distribution delay (ms)")
    ax.set_title("Signaling distribution delay comparison ($L=7$, $N=M=50$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9, ncol=2, fontsize=7)
    ax.set_xticks(J_vals)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig5.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Fig.5 saved to {output_dir}/fig5.png")


def plot_fig6(nc_comparison, output_dir):
    """
    Fig.6: Computing nodes required for different deployment schemes
    Three subplots (one per constellation scenario)
    Each subplot has three curves: LEO Spot, LEO Polygon, MEO
    """
    scenarios = nc_comparison["scenarios"]
    J_vals = nc_comparison["J_values"]

    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=FIG_DOUBLE, sharey=False)

    if n_scenarios == 1:
        axes = [axes]

    method_styles = {
        "LEO Spot Beam": {"color": COLORS[0], "marker": MARKERS[0], "ls": "-"},
        "LEO Polygon Beam": {"color": COLORS[1], "marker": MARKERS[1], "ls": "--"},
        "MEO Computing": {"color": COLORS[2], "marker": MARKERS[2], "ls": "-."},
    }

    for ax_idx, (sc_name, sc_data) in enumerate(scenarios.items()):
        ax = axes[ax_idx]

        for method_name, nc_arr in sc_data.items():
            style = method_styles[method_name]
            ax.plot(J_vals, nc_arr,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["ls"], markersize=5,
                    label=method_name)

        ax.set_xlabel("Reachable hop count $J$")
        if ax_idx == 0:
            ax.set_ylabel("Number of computing nodes $N_C$")
        ax.set_title(sc_name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(J_vals)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7)

    fig.suptitle("Computing nodes required for different deployment schemes",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig6.png", dpi=300)
    plt.close(fig)
    print(f"  [OK] Fig.6 saved to {output_dir}/fig6.png")


