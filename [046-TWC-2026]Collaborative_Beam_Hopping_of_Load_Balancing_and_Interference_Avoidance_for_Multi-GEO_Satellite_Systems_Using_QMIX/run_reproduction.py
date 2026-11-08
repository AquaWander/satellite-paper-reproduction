"""
run_reproduction.py — 一键复现脚本
[046-TWC-2026] Collaborative Beam Hopping using QMIX
"""
import sys
import os
import numpy as np

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from simulation import run_full_simulation
from plotting import plot_all, validate_results
import time


def main():
    """主流程: 仿真 -> 绘图 -> 验证"""
    print("=" * 70)
    print("论文复现: [046-TWC-2026] Collaborative Beam Hopping Using QMIX")
    print("=" * 70)
    print(f"卫星数: {cfg.N_SATELLITES}, 每星波束: {cfg.N_BEAMS}, 每星小区: {cfg.N_CELLS}")
    print(f"QMIX 训练轮数: {cfg.QMIX_N_EPOCHS}, 每轮步数: {cfg.QMIX_N_STEPS_PER_EPOCH}")
    print(f"流量点数: {cfg.N_TRAFFIC_POINTS}, 每点运行: {cfg.N_RUNS} 次")
    print()

    t_start = time.time()

    # Step 1: 运行仿真
    print("\n" + "#" * 70)
    print("Step 1: 运行仿真")
    print("#" * 70)
    fig9_data, fig11_data, traffic_demands = run_full_simulation(
        verbose=True, use_cached=False)

    t_sim = time.time()
    print(f"\n仿真完成, 总耗时: {(t_sim - t_start)/60:.1f} 分钟")

    # Step 2: 绘制图表
    print("\n" + "#" * 70)
    print("Step 2: 绘制图表")
    print("#" * 70)
    plot_all(traffic_demands, fig9_data, fig11_data)

    # Step 3: 数值验证
    print("\n" + "#" * 70)
    print("Step 3: 数值验证")
    print("#" * 70)
    all_pass = validate_results(traffic_demands, fig9_data, fig11_data)

    t_end = time.time()
    print(f"\n总耗时: {(t_end - t_start)/60:.1f} 分钟")

    if all_pass:
        print("\n" + "=" * 70)
        print("所有验证通过! 复现成功!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("部分验证未通过, 请检查输出.")
        print("=" * 70)

    print(f"\n输出文件位置: {os.path.join(os.path.dirname(__file__), 'output')}")


if __name__ == '__main__':
    main()