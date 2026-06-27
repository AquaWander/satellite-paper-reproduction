"""
run_reproduction.py
=================
一键复现脚本: import 全部模块, 跑仿真, 存图到 output/, 打印验证表.

流程:
  1. 6 算法 × N_SEEDS_ALGO 种子训练  → Fig.3a/b, Fig.4
  2. 5 LLM × N_SEEDS_LLM 种子训练    → Fig.3c, Table II, Table III
  3. 绘图 (IEEE 风格)
  4. 数值验证 (趋势/百分比/约束/t 检验) + 打印对比表

用法:
    python run_reproduction.py            # 完整流程
    python run_reproduction.py --quick    # 快速 (少种子少 episode, 验证流程)
"""
from __future__ import annotations
import os, sys, time, argparse, json
import numpy as np
from scipy import stats

# *** 控时: 每 worker 1 线程 (Phase 1 并行 6 worker × 1 线程; Phase 2 串行 1 进程) ***
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

import config as C
import simulation as S
import plotting as P


def _last_window_mean(arr, frac=0.2):
    """末段 frac 比例窗口的均值 (跨种子+episode)."""
    n = arr.shape[-1]
    k = max(1, int(n * frac))
    return float(arr[..., -k:].mean())


def _per_seed_final(arr, frac=0.2):
    """每种种子的末段均值 (用于 Table II 统计: Mean/Std/Min/Max/Median/IQR)."""
    n = arr.shape[-1]
    k = max(1, int(n * frac))
    return arr[..., -k:].mean(axis=-1)    # (n_seeds,)


def run_algo_phase(n_seeds, n_episodes, verbose=True):
    """阶段 1: 6 算法对比."""
    print("\n" + "=" * 70)
    print("Phase 1: 6-Algorithm Comparison (Fig.3a/b, Fig.4)")
    print("=" * 70)
    t0 = time.time()
    rewards, handovers, f1s = S.run_algo_comparison(
        seeds=list(range(n_seeds)), n_episodes=n_episodes, verbose=verbose,
        parallel=True)
    print(f"Phase 1 total time: {time.time()-t0:.1f}s")
    return rewards, handovers, f1s


def run_llm_phase(n_seeds, n_episodes, verbose=True):
    """阶段 2: 5 LLM 对比 (LTQC-DAM 框架). 用并行 (经验证 spawn pool 工作正常)."""
    print("\n" + "=" * 70)
    print("Phase 2: 5-LLM Comparison (Fig.3c, Table II/III)")
    print("=" * 70)
    t0 = time.time()
    results = S.run_llm_comparison(
        seeds=list(range(n_seeds)), n_episodes=n_episodes, verbose=verbose,
        parallel=True)
    print(f"Phase 2 total time: {time.time()-t0:.1f}s")
    return results


def make_plots(rewards, handovers, f1s, llm_results):
    """生成所有图."""
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)
    # Fig.3(a): 收敛 reward
    P.plot_convergence(rewards, "Episodic reward",
                       "fig03a_convergence_reward.png")
    # Fig.3(b): 收敛 handover (handovers 是 dict[algo]=int array, 转 float)
    handovers_float = {a: handovers[a].astype(float) for a in handovers}
    P.plot_convergence(handovers_float, "Handover count",
                       "fig03b_convergence_handover.png")
    # Fig.4: 双 y 轴柱状图
    f1_means = {a: float(np.mean(f1s[a])) for a in f1s}
    f2_means = {a: _last_window_mean(handovers[a].astype(float))
                for a in handovers}
    P.plot_avg_performance(f1_means, f2_means, "fig04_avg_performance.png")
    # Fig.3(c): 5 LLM 收敛
    P.plot_llm_convergence(llm_results, "fig03c_llm_convergence.png")
    return f1_means, f2_means


def validate_and_report(rewards, handovers, f1s, f1_means, f2_means,
                        llm_results):
    """数值验证 + 打印对比表."""
    print("\n" + "=" * 70)
    print("Validation & Comparison with Paper")
    print("=" * 70)

    # --- 1. 趋势: LTQC-DAM 收敛最快/终值最高; DQN 终值最低 ---
    finals = {a: _last_window_mean(rewards[a]) for a in C.ALGO_LIST}
    print("\n[1] Algorithm final episodic reward (last 20% mean):")
    for a in C.ALGO_LIST:
        print(f"    {a:10s}: {finals[a]:7.3f}")
    best = max(finals, key=finals.get)
    worst = min(finals, key=finals.get)
    print(f"    -> Best: {best} ({finals[best]:.3f}); Worst: {worst} "
          f"({finals[worst]:.3f})")
    ltqc, tqc = finals["LTQC-DAM"], finals["TQC"]
    print(f"    -> LTQC-DAM ({ltqc:.3f}) vs TQC ({tqc:.3f}): "
          f"{'LTQC higher OK' if ltqc > tqc else 'LTQC lower FAIL'}")

    # --- 2. 百分比: f2 降幅 ~17.69%, f1 增幅 ~0.44% ---
    f2_ltqc = _last_window_mean(handovers["LTQC-DAM"].astype(float))
    f2_tqc = _last_window_mean(handovers["TQC"].astype(float))
    f1_ltqc = float(np.mean(f1s["LTQC-DAM"]))
    f1_tqc = float(np.mean(f1s["TQC"]))
    f2_drop = (f2_tqc - f2_ltqc) / f2_tqc * 100 if f2_tqc > 0 else 0
    f1_gain = (f1_ltqc - f1_tqc) / f1_tqc * 100 if f1_tqc > 0 else 0
    print(f"\n[2] LTQC-DAM vs TQC (paper targets: f2 -17.69%, f1 +0.44%):")
    print(f"    f2 (handover): TQC={f2_tqc:.2f}, LTQC={f2_ltqc:.2f}, "
          f"drop={f2_drop:.2f}% (target ~17.69%)")
    print(f"    f1 (rate):     TQC={f1_tqc:.3f}, LTQC={f1_ltqc:.3f}, "
          f"gain={f1_gain:.3f}% (target ~0.44%)")
    f2_ok = 12 < f2_drop < 24
    f1_ok = 0.0 < f1_gain < 1.5
    print(f"    -> f2 drop in [12,24]%: {'OK' if f2_ok else 'FAIL'}; "
          f"f1 gain in [0,1.5]%: {'OK' if f1_ok else 'FAIL'}")

    # --- 3. Table II: 5 LLM 统计 ---
    print(f"\n[3] Table II: 5-LLM statistics (last 20% episodic reward):")
    print(f"    {'LLM':10s} {'Mean':>7s} {'Std':>6s} {'Min':>7s} {'Max':>7s} "
          f"{'Median':>7s} {'IQR':>6s}  | Paper Mean (Std)")
    llm_finals = {}
    for llm in C.LLM_LIST:
        per_seed = _per_seed_final(llm_results[llm])
        llm_finals[llm] = per_seed
        mean = float(np.mean(per_seed))
        std = float(np.std(per_seed))
        mn, mx = float(per_seed.min()), float(per_seed.max())
        med = float(np.median(per_seed))
        q1, q3 = np.percentile(per_seed, [25, 75])
        iqr = float(q3 - q1)
        paper = C.LLM_QUALITY[llm]
        print(f"    {llm:10s} {mean:7.3f} {std:6.3f} {mn:7.3f} {mx:7.3f} "
              f"{med:7.3f} {iqr:6.3f}  | {paper['target_mean']:.2f} "
              f"({paper['target_std']:.2f})")

    # 排序检查
    means = {llm: float(np.mean(llm_finals[llm])) for llm in C.LLM_LIST}
    order = sorted(means, key=means.get, reverse=True)
    expected = ["DeepSeek", "Claude", "Grok", "ChatGPT", "Qwen"]
    print(f"    -> Reproduced order: {order}")
    print(f"    -> Expected order:   {expected}")
    order_ok = (order == expected)
    print(f"    -> Order matches: {'OK' if order_ok else 'FAIL'}")

    # --- 4. Table III: 配对 t 检验 (ttest_rel, 因为 5 LLM 用相同 seeds 是配对设计) ---
    print(f"\n[4] Table III: paired t-test p-values (all should be < 0.05):")
    n_llm = len(C.LLM_LIST)
    p_matrix = np.zeros((n_llm, n_llm))
    all_significant = True
    header = "    " + " " * 10 + "".join(f"{l[:6]:>9s}" for l in C.LLM_LIST)
    print(header)
    for i, li in enumerate(C.LLM_LIST):
        row = f"    {li:10s}"
        for j, lj in enumerate(C.LLM_LIST):
            if i == j:
                row += f"{'1.000':>9s}"
                continue
            t_stat, p_val = stats.ttest_rel(llm_finals[li], llm_finals[lj])
            p_matrix[i, j] = p_val
            row += f"{p_val:>9.4f}"
            if p_val >= 0.05:
                all_significant = False
        print(row)
    print(f"    -> All p < 0.05: {'OK' if all_significant else 'FAIL'}")

    # --- 5. 约束检查 ---
    print(f"\n[5] Constraint checks:")
    constraints_ok = True
    for llm in C.LLM_LIST:
        per = llm_finals[llm]
        m = float(np.mean(per))
        ok = per.min() <= m <= per.max()
        if not ok:
            constraints_ok = False
    ho_nonneg = all((handovers[a] >= 0).all() for a in handovers)
    print(f"    min <= mean <= max for all LLM: {'OK' if constraints_ok else 'FAIL'}")
    print(f"    handover counts >= 0 (integer): {'OK' if ho_nonneg else 'FAIL'}")

    return dict(finals=finals, f2_drop=f2_drop, f1_gain=f1_gain,
                order_ok=order_ok, all_p_significant=all_significant,
                constraints_ok=constraints_ok and ho_nonneg,
                llm_finals=llm_finals, p_matrix=p_matrix)


def save_results(rewards, handovers, f1s, llm_results, report):
    """存训练数据 + 报告 (便于复用)."""
    out = C.OUTPUT_DIR
    np.savez(os.path.join(out, "algo_rewards.npz"), **rewards)
    np.savez(os.path.join(out, "algo_handovers.npz"),
             **{k: v.astype(np.int16) for k, v in handovers.items()})
    np.savez(os.path.join(out, "algo_f1.npz"), **f1s)
    np.savez(os.path.join(out, "llm_rewards.npz"), **llm_results)
    # 报告
    serial = {k: (v.tolist() if hasattr(v, "tolist") else v)
              for k, v in report.items()}
    with open(os.path.join(out, "report.json"), "w") as fp:
        json.dump(serial, fp, indent=2, default=str)
    print(f"\nResults saved to {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 少种子少 episode")
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--n_seeds_algo", type=int, default=None)
    parser.add_argument("--n_seeds_llm", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        n_ep = args.n_episodes or 80
        n_sa = args.n_seeds_algo or 2
        n_sl = args.n_seeds_llm or 3
    else:
        n_ep = args.n_episodes or C.N_EPISODES
        n_sa = args.n_seeds_algo or C.N_SEEDS_ALGO
        n_sl = args.n_seeds_llm or C.N_SEEDS_LLM

    print(f"Config: n_episodes={n_ep}, n_seeds_algo={n_sa}, "
          f"n_seeds_llm={n_sl}")
    print(f"Output dir: {C.OUTPUT_DIR}")

    t_total = time.time()
    rewards, handovers, f1s = run_algo_phase(n_sa, n_ep)
    llm_results = run_llm_phase(n_sl, n_ep)
    # 先保存原始训练数据 (即使后续绘图/report 失败, 数据不丢)
    save_results(rewards, handovers, f1s, llm_results,
                 dict(finals={}, f2_drop=0, f1_gain=0, order_ok=False,
                      all_p_significant=False, constraints_ok=False,
                      llm_finals={}, p_matrix=[]))
    # 绘图 + 报告
    f1_means, f2_means = make_plots(rewards, handovers, f1s, llm_results)
    report = validate_and_report(rewards, handovers, f1s, f1_means, f2_means,
                                 llm_results)
    # 用完整 report 再保存一次
    save_results(rewards, handovers, f1s, llm_results, report)

    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {time.time()-t_total:.1f}s "
          f"({(time.time()-t_total)/60:.1f} min)")
    print(f"{'='*70}")
    print("\nOutput files:")
    for f in sorted(os.listdir(C.OUTPUT_DIR)):
        path = os.path.join(C.OUTPUT_DIR, f)
        sz = os.path.getsize(path)
        print(f"  {f}  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
