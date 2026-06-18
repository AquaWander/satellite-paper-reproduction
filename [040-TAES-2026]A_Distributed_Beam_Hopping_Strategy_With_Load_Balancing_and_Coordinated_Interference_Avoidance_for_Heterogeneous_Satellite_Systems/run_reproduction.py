# -*- coding: utf-8 -*-
"""
一键复现脚本: Table III -> Fig.5 -> Fig.7 -> Fig.8。
打印验证表, 保存所有图到 output/。
"""
import os
import time
import numpy as np

import config as C
C.setup_matplotlib()
from simulation import run_one_frame, run_averaged
import rw_lb
from geometry import Geometry
from traffic_model import generate_traffic
import plotting

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

SEP = "=" * 78


def header(t):
    print("\n" + SEP)
    print(t)
    print(SEP)


# =====================================================================
# 目标 1: Table III
# =====================================================================
def run_tableIII():
    header("[Target 1] Table III: Load Disparity (640 Mbps, sweep N_LEO)")
    results = {"RW-LB": [], "Distance-based": []}
    util_store = {n: {"RW-LB": None, "Distance-based": None} for n in C.N_LEO_SWEEP}

    for n_leo in C.N_LEO_SWEEP:
        rw_d, dist_d = [], []
        for r in range(C.N_RUNS_TABLE):
            seed = C.RANDOM_SEED_BASE + r * 17 + n_leo
            geo = Geometry(n_leo, seed=seed)
            D_rt, D_nrt, _ = generate_traffic(geo, C.TOTAL_DEMAND_MBPS, seed=seed)
            cell_total = (D_rt + D_nrt).sum(axis=1)
            rng2 = np.random.default_rng(seed)
            _, _, _, u1 = rw_lb.select_rw_lb(geo, cell_total, rng2)
            _, _, _, u2 = rw_lb.select_distance(geo, cell_total)
            rw_d.append(rw_lb.load_disparity(u1))
            dist_d.append(rw_lb.load_disparity(u2))
            if r == 0:
                util_store[n_leo]["RW-LB"] = u1.copy()
                util_store[n_leo]["Distance-based"] = u2.copy()
        results["RW-LB"].append(float(np.mean(rw_d)))
        results["Distance-based"].append(float(np.mean(dist_d)))

    # 打印验证表
    print(f"\n{'Strategy':<18}" + "".join([f"{n:>8}" for n in C.N_LEO_SWEEP]) +
          "   | status")
    print("-" * 78)
    for strat in ["RW-LB", "Distance-based"]:
        line = f"{strat:<18}"
        ok_all = True
        for i, n in enumerate(C.N_LEO_SWEEP):
            repro = results[strat][i]
            paper = C.TABLEIII_PAPER[strat][i]
            dev = repro - paper
            ok = abs(dev) < 0.03
            ok_all = ok_all and ok
            line += f"{repro:>8.3f}"
        line += f"   | {'OK' if ok_all else 'MISMATCH (see per-point)'}"
        print(line)
    print("-" * 78)
    print("paper values:")
    for strat in ["RW-LB", "Distance-based"]:
        print(f"  {strat:<18}" + "".join(f"{v:>8.3f}" for v in C.TABLEIII_PAPER[strat]))
    print("\nPer-point deviation (repro - paper):")
    for strat in ["RW-LB", "Distance-based"]:
        dev = [results[strat][i] - C.TABLEIII_PAPER[strat][i]
               for i in range(len(C.N_LEO_SWEEP))]
        status = ["OK" if abs(d) < 0.03 else f"MISMATCH({d:+.2f})" for d in dev]
        print(f"  {strat:<18}" + "".join(f"{s:>14}" for s in status))

    plotting.plot_tableIII(results, os.path.join(OUT, "tableIII_load_disparity.png"))
    print(f"\nSaved: output/tableIII_load_disparity.png")
    return results, util_store


# =====================================================================
# 目标 2: Fig.5
# =====================================================================
def run_fig05(util_store):
    header("[Target 2] Fig.5: RW-LB vs Distance Load Balancing (15 LEO)")
    u_rw = util_store[15]["RW-LB"]
    u_dist = util_store[15]["Distance-based"]
    print(f"RW-LB util (15 LEO): {np.round(u_rw, 3)}")
    print(f"RW-LB load disparity L_df = {u_rw.max()-u_rw.min():.3f} (paper 0.11)")
    print(f"Distance util:      {np.round(u_dist, 3)}")
    print(f"Distance L_df = {u_dist.max()-u_dist.min():.3f} (paper 0.72)")
    plotting.plot_fig05(u_rw, u_dist, os.path.join(OUT, "fig05_load_balancing.png"))
    print("Saved: output/fig05_load_balancing.png")


# =====================================================================
# 目标 3: Fig.7 (业务需求扫描, 15 LEO)
# =====================================================================
def run_fig07(n_runs, ga_fast=False):
    header(f"[Target 3] Fig.7: Traffic Demand Sweep (15 LEO, {n_runs} runs)")
    t0 = time.time()
    sweep = {s: {"rt_throughput": [], "total_throughput": [],
                 "rt_satisf": [], "total_satisf": []} for s in C.SCHEMES}
    for d in C.DEMAND_SWEEP:
        for s in C.SCHEMES:
            res = run_averaged(15, d, s, n_runs, C.RANDOM_SEED_BASE, ga_fast=ga_fast)
            for m in sweep[s]:
                sweep[s][m].append(res[m])
        print(f"  demand={d} Mbps done ({time.time()-t0:.0f}s)")
    plotting.plot_fig07(sweep, os.path.join(OUT, "fig07_traffic_demand.png"))
    print(f"Saved: output/fig07_traffic_demand.png  ({time.time()-t0:.0f}s)")

    # 趋势验证: DLBIA-BH RT 吞吐 随需求先增后饱和
    rt = sweep["DLBIA-BH"]["rt_throughput"]
    print(f"\nDLBIA-BH RT throughput vs demand: {[round(x,1) for x in rt]}")
    inc_to = max(range(len(rt)), key=lambda i: rt[i])
    print(f"  peak index (should be last/saturated): {inc_to}, value={rt[inc_to]:.1f}")
    # 方案排序 (在最高需求点)
    order = sorted(C.SCHEMES, key=lambda s: -sweep[s]["rt_throughput"][-1])
    print(f"  scheme ranking @640Mbps (RT): {order}")
    expected_best = "DLBIA-BH"
    print(f"  best == DLBIA-BH? {order[0] == expected_best}")
    return sweep


# =====================================================================
# 目标 4: Fig.8 (卫星数扫描, 640 Mbps)
# =====================================================================
def run_fig08(n_runs, ga_fast=False):
    header(f"[Target 4] Fig.8: Satellite Number Sweep (640 Mbps, {n_runs} runs)")
    t0 = time.time()
    sweep = {s: {"rt_throughput": [], "total_throughput": [],
                 "rt_satisf": [], "total_satisf": []} for s in C.SCHEMES}
    for n_leo in C.N_LEO_SWEEP:
        for s in C.SCHEMES:
            res = run_averaged(n_leo, C.TOTAL_DEMAND_MBPS, s, n_runs,
                               C.RANDOM_SEED_BASE, ga_fast=ga_fast)
            for m in sweep[s]:
                sweep[s][m].append(res[m])
        print(f"  N_LEO={n_leo} done ({time.time()-t0:.0f}s)")
    plotting.plot_fig08(sweep, os.path.join(OUT, "fig08_num_satellites.png"))
    print(f"Saved: output/fig08_num_satellites.png  ({time.time()-t0:.0f}s)")

    rt = {s: sweep[s]["rt_throughput"] for s in C.SCHEMES}
    print("\nRT throughput @ each N_LEO:")
    print(f"  {'scheme':<12}" + "".join(f"{n:>8}" for n in C.N_LEO_SWEEP))
    for s in C.SCHEMES:
        print(f"  {s:<12}" + "".join(f"{v:>8.1f}" for v in rt[s]))
    # 验证 RT 吞吐随卫星数递增
    dl = rt["DLBIA-BH"]
    mono = all(dl[i + 1] >= dl[i] - 5 for i in range(len(dl) - 1))
    print(f"\nDLBIA-BH RT throughput monotonic-increasing (tol 5)? {mono}")
    # 排序恒定
    for i, n in enumerate(C.N_LEO_SWEEP):
        order = sorted(C.SCHEMES, key=lambda s: -rt[s][i])
        print(f"  N_LEO={n}: ranking = {order}")
    return sweep


# =====================================================================
# 数学约束验证
# =====================================================================
def math_checks():
    header("[Math Constraints Check]")
    bits = int(np.ceil(np.log2(C.N_CELLS)))
    print(f"gene bits per beam = ceil(log2({C.N_CELLS})) = {bits}  (expect 9) "
          f"-> {'OK' if bits == 9 else 'FAIL'}")
    res = run_one_frame(15, 640, "DLBIA-BH", seed=42)
    print(f"sample run: rt_thr={res['rt_throughput']:.1f} (>=0? "
          f"{res['rt_throughput']>=0}); rt_satisf={res['rt_satisf']:.3f} in[0,1]? "
          f"{0<=res['rt_satisf']<=1}; L_df={res['L_df']:.3f}>=0? {res['L_df']>=0}")


def main():
    header("DLBIA-BH Reproduction — TAES 2026")
    math_checks()
    results, util_store = run_tableIII()
    run_fig05(util_store)
    # Fig.7/8 用 30 次平均 + GA fast 模式 (趋势验证)
    run_fig07(n_runs=C.N_RUNS_FIG, ga_fast=True)
    run_fig08(n_runs=C.N_RUNS_FIG, ga_fast=True)
    header("All targets complete.")
    print(f"Output dir: {OUT}")


if __name__ == "__main__":
    main()
