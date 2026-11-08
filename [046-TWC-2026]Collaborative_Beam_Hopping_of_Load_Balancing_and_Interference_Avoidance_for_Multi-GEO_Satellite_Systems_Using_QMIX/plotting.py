"""
plotting.py — IEEE 期刊风格绘图
实现: Fig.9 (吞吐量), Fig.10 (延迟偏移), Fig.11 (满意度率+负载偏移)
"""
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import numpy as np
import os
import config as cfg

# 设置 IEEE 风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': cfg.FONT_SIZE,
    'axes.linewidth': cfg.AXIS_LINE_WIDTH,
    'lines.linewidth': cfg.LINE_WIDTH,
    'lines.markersize': cfg.MARKER_SIZE,
    'figure.dpi': cfg.DPI,
    'savefig.dpi': cfg.DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUTPUT_DIR = 'C:/Users/windows/Desktop/文章复现/[046-TWC-2026]Collaborative_Beam_Hopping_of_Load_Balancing_and_Interference_Avoidance_for_Multi-GEO_Satellite_Systems_Using_QMIX/output'


def _get_style(name):
    """获取曲线样式"""
    return cfg.CURVE_STYLES.get(name, {'color': '#000000', 'marker': 'o', 'linestyle': '-', 'label': name})


def setup_axis(ax, xlabel, ylabel, xlim=None, ylim=None):
    """设置坐标轴格式"""
    ax.set_xlabel(xlabel, fontsize=cfg.FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.FONT_SIZE)
    ax.tick_params(direction='in', length=4, width=0.8, labelsize=cfg.FONT_SIZE-1)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, linestyle=':')


def plot_fig9(traffic_demands, results, save=True):
    """Fig.9: 吞吐量 vs 流量需求
    X: Traffic demand (Mbps), Y: Throughput (Mbps)
    7 条曲线: RBH, PBH, DFBH, GABH, SADRL-BH, MADRL-BH, CMSBH
    """
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_DOUBLE)

    algorithms = ['RBH', 'PBH', 'DFBH', 'GABH', 'SADRL-BH', 'MADRL-BH', 'CMSBH']

    for name in algorithms:
        style = _get_style(name)
        y = np.array(results[name]['throughput'])
        ax.plot(traffic_demands, y,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                label=style['label'],
                linewidth=cfg.LINE_WIDTH,
                markersize=cfg.MARKER_SIZE)

    # 理想吞吐量线 (y=x)
    ax.plot(traffic_demands, traffic_demands,
            color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Ideal')

    setup_axis(ax,
               xlabel='Traffic demand (Mbps)',
               ylabel='Throughput (Mbps)',
               xlim=(min(traffic_demands), max(traffic_demands)))

    ax.legend(loc='lower right', framealpha=0.9, fontsize=cfg.FONT_SIZE-2,
              ncol=2)

    fig.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'fig9_throughput.png')
        fig.savefig(filepath, dpi=cfg.DPI)
        print(f"Fig.9 已保存: {filepath}")
    plt.close(fig)
    return fig


def plot_fig10(traffic_demands, results, save=True):
    """Fig.10: 延迟偏移 vs 流量需求
    X: Traffic demand (Mbps), Y: Delay offset (ms)
    7 条曲线: RBH, PBH, DFBH, GABH, SADRL-BH, MADRL-BH, CMSBH
    """
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_DOUBLE)

    algorithms = ['RBH', 'PBH', 'DFBH', 'GABH', 'SADRL-BH', 'MADRL-BH', 'CMSBH']

    for name in algorithms:
        style = _get_style(name)
        y = np.array(results[name]['delay_offset'])
        ax.plot(traffic_demands, y,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                label=style['label'],
                linewidth=cfg.LINE_WIDTH,
                markersize=cfg.MARKER_SIZE)

    setup_axis(ax,
               xlabel='Traffic demand (Mbps)',
               ylabel='Delay offset (ms)',
               xlim=(min(traffic_demands), max(traffic_demands)))

    ax.legend(loc='upper left', framealpha=0.9, fontsize=cfg.FONT_SIZE-2,
              ncol=2)

    fig.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'fig10_delay_offset.png')
        fig.savefig(filepath, dpi=cfg.DPI)
        print(f"Fig.10 已保存: {filepath}")
    plt.close(fig)
    return fig


def plot_fig11(traffic_demands, results, save=True):
    """Fig.11: (a) 平均流量满意度率, (b) 归一化负载偏移
    X: Traffic demand (Mbps)
    7 条曲线: RBH, PBH, LBGT-MSBH, PH-MSBH, I-MSBH, CMARL-MSBH, CMSBH
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cfg.FIGSIZE_DOUBLE[0], cfg.FIGSIZE_DOUBLE[1]*0.85))

    algorithms = ['RBH', 'PBH', 'LBGT-MSBH', 'PH-MSBH', 'I-MSBH', 'CMARL-MSBH', 'CMSBH']

    # (a) 平均流量满意度率
    for name in algorithms:
        style = _get_style(name)
        y = np.array(results[name]['satisfaction'])
        ax1.plot(traffic_demands, y,
                 color=style['color'],
                 marker=style['marker'],
                 linestyle=style['linestyle'],
                 label=style['label'],
                 linewidth=cfg.LINE_WIDTH,
                 markersize=cfg.MARKER_SIZE)

    setup_axis(ax1,
               xlabel='Traffic demand (Mbps)',
               ylabel='Avg traffic satisfaction rate',
               xlim=(min(traffic_demands), max(traffic_demands)),
               ylim=(0, 1.05))
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=cfg.FONT_SIZE-2,
               ncol=1)
    ax1.set_title('(a)', fontsize=cfg.FONT_SIZE, loc='left')

    # (b) 归一化负载偏移
    for name in algorithms:
        style = _get_style(name)
        y = np.array(results[name]['norm_load_offset'])
        ax2.plot(traffic_demands, y,
                 color=style['color'],
                 marker=style['marker'],
                 linestyle=style['linestyle'],
                 label=style['label'],
                 linewidth=cfg.LINE_WIDTH,
                 markersize=cfg.MARKER_SIZE)

    setup_axis(ax2,
               xlabel='Traffic demand (Mbps)',
               ylabel='Normalized load offset',
               xlim=(min(traffic_demands), max(traffic_demands)))
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=cfg.FONT_SIZE-2,
               ncol=1)
    ax2.set_title('(b)', fontsize=cfg.FONT_SIZE, loc='left')

    fig.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'fig11_satisfaction_load.png')
        fig.savefig(filepath, dpi=cfg.DPI)
        print(f"Fig.11 已保存: {filepath}")
    plt.close(fig)
    return fig


def plot_training_loss(losses, save=True):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_SINGLE)
    ax.plot(losses, color='#000000', linewidth=cfg.LINE_WIDTH)
    setup_axis(ax, xlabel='Epoch', ylabel='Loss')
    fig.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'training_loss.png')
        fig.savefig(filepath, dpi=cfg.DPI)
        print(f"训练损失已保存: {filepath}")
    plt.close(fig)


def plot_all(traffic_demands, fig9_data, fig11_data):
    """绘制所有图表"""
    plot_fig9(traffic_demands, fig9_data)
    plot_fig10(traffic_demands, fig9_data)
    plot_fig11(traffic_demands, fig11_data)
    print("\n所有图表已生成!")


def validate_results(traffic_demands, fig9_data, fig11_data):
    """数值验证"""
    print("\n" + "="*60)
    print("数值验证")
    print("="*60)

    errors = []
    warnings = []

    # 1. 曲线数量验证
    for name, data in [('Fig.9', fig9_data), ('Fig.11', fig11_data)]:
        n_curves = len(data)
        expected = 7
        if n_curves == expected:
            print(f"[PASS] {name}: {n_curves} 条曲线 (期望 {expected})")
        else:
            msg = f"{name}: {n_curves} 条曲线 (期望 {expected})"
            errors.append(msg)

    # 2. 趋势验证: CMSBH 应最优
    for fig_name, data, metric, maximize in [
        ('Fig.9', fig9_data, 'throughput', True),
        ('Fig.10', fig9_data, 'delay_offset', False),
        ('Fig.11(a)', fig11_data, 'satisfaction', True),
        ('Fig.11(b)', fig11_data, 'norm_load_offset', False),
    ]:
        cmsbh_vals = np.array(data['CMSBH'][metric])
        for algo_name in data:
            if algo_name == 'CMSBH':
                continue
            other_vals = np.array(data[algo_name][metric])
            if maximize:
                # CMSBH 应 >= 其他
                if np.any(cmsbh_vals < other_vals - 1e-6):
                    idx = np.where(cmsbh_vals < other_vals - 1e-6)[0]
                    w = f"[WARN] {fig_name}: CMSBH < {algo_name} at {len(idx)}/{len(cmsbh_vals)} points"
                    warnings.append(w)
            else:
                # CMSBH 应 <= 其他
                if np.any(cmsbh_vals > other_vals + 1e-6):
                    idx = np.where(cmsbh_vals > other_vals + 1e-6)[0]
                    w = f"[WARN] {fig_name}: CMSBH > {algo_name} at {len(idx)}/{len(cmsbh_vals)} points"
                    warnings.append(w)

    if not warnings:
        print("[PASS] 趋势验证: CMSBH 在所有指标上最优")
    else:
        for w in warnings:
            print(w)

    # 3. 数学约束验证
    for fig_name, data in [('Fig.9', fig9_data), ('Fig.11', fig11_data)]:
        for algo_name in data:
            for metric in data[algo_name]:
                vals = np.array(data[algo_name][metric])
                if metric == 'satisfaction':
                    if np.any(vals < 0) or np.any(vals > 1):
                        errors.append(f"{fig_name} {algo_name} {metric}: 超出 [0,1]")
                if np.any(vals < 0):
                    errors.append(f"{fig_name} {algo_name} {metric}: 负值")

    if not errors:
        print("[PASS] 所有数学约束验证通过")
    else:
        print(f"[FAIL] {len(errors)} 个验证错误:")
        for e in errors:
            print(f"  - {e}")

    # 4. 吞吐量 ≤ 流量需求
    for algo_name in fig9_data:
        for i, d in enumerate(traffic_demands):
            if fig9_data[algo_name]['throughput'][i] > d + 1:
                errors.append(f"Fig.9 {algo_name} @ {d}: 吞吐量 > 流量需求")

    if not errors:
        print("[PASS] 吞吐量 ≤ 流量需求")
    else:
        for e in errors:
            print(e)

    return len(errors) == 0