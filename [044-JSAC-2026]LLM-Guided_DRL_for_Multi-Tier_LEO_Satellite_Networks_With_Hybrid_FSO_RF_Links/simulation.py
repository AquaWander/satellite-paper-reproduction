"""
simulation.py
==========
训练循环: 跑 6 算法对比 (Fig.3a/b, Fig.4) 与 5 LLM 对比 (Fig.3c, Table II/III).

返回结构:
  results_algo[algo_name] = np.ndarray (n_seeds, n_episodes)   episodic reward
  handover_algo[algo_name] = np.ndarray (n_seeds, n_episodes)  每集切换数
  f1_algo[algo_name]       = np.ndarray (n_seeds,)             每集速率贡献 (末期均值)
  results_llm[llm_name]    = np.ndarray (n_seeds, n_episodes)  LTQC-DAM 各 LLM 收敛
"""
from __future__ import annotations
import os, time
import numpy as np
import torch

import config as C
from environment import SatelliteHAPEnv
import agents as A


def make_agent(name: str, state_dim: int, action_dim: int,
               llm_name: str = "DeepSeek", seed: int = None):
    """工厂: 按 name 创建对应 agent."""
    if name == "DQN":
        return A.DQNAgent(state_dim, action_dim)
    if name == "PPO":
        return A.PPOAgent(state_dim, action_dim)
    if name == "SAC":
        return A.SACAgent(state_dim, action_dim)
    if name == "TD3":
        return A.TD3Agent(state_dim, action_dim)
    if name == "TQC":
        return A.TQCAgent(state_dim, action_dim)
    if name == "LTQC-DAM":
        # *** 根因3: 传 seed 给 LLM (配对扰动流) ***
        return A.LTQCDAMAgent(state_dim, action_dim, llm_name=llm_name, seed=seed)
    raise ValueError(name)


def train_one_run(algo_name: str, seed: int,
                  n_episodes: int = C.N_EPISODES,
                  llm_name: str = "DeepSeek",
                  verbose: bool = False):
    """
    单种子训练. 返回:
      rewards (n_episodes,), handovers (n_episodes,), f1_last (float)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = SatelliteHAPEnv(seed=seed)
    agent = make_agent(algo_name, env.state_dim, env.action_dim, llm_name=llm_name,
                       seed=seed)
    # 设置实际 episode 总数 (用于 LTQC-DAM 的 ε 衰减计算, Eq.26)
    if hasattr(agent, "set_total_episodes"):
        agent.set_total_episodes(n_episodes)
    rewards = np.zeros(n_episodes)
    handovers = np.zeros(n_episodes, dtype=int)
    f1s = np.zeros(n_episodes)

    t_start = time.time()
    for ep in range(n_episodes):
        s = env.reset()
        ep_r, ep_ho, ep_f1 = 0.0, 0, 0.0
        done = False
        while not done:
            mask = env.action_mask()
            a = agent.act(s, mask, explore=True, current_a=env.cur_sat)
            s2, r, done, info = env.step(a)
            next_mask = env.action_mask() if not done else mask
            if algo_name == "PPO":
                agent.store(s, a, r, s2, float(done), mask)
            else:
                agent.store(s, a, r, s2, float(done), next_mask)
                agent.train_step()
            s = s2
            ep_r += r
            ep_f1 += info["f1_increment"]
            ep_ho = info["n_handover"]
        if algo_name == "PPO":
            agent.finish_episode(last_val=0.0)
        if algo_name == "LTQC-DAM":
            agent.end_episode(ep_r)
        rewards[ep] = ep_r
        handovers[ep] = ep_ho
        f1s[ep] = ep_f1
        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"  [{algo_name}{'/'+llm_name if algo_name=='LTQC-DAM' else ''}] "
                  f"seed={seed} ep={ep+1}/{n_episodes} "
                  f"reward={ep_r:.2f} ho={ep_ho} f1={ep_f1:.2f} "
                  f"({time.time()-t_start:.1f}s)")
    return rewards, handovers, f1s


def _worker(args):
    """multiprocessing worker: 跑单 (algo, seed, n_episodes, llm_name) 返回结果.
    顺序模式下在主进程调用, 不改线程数 (用 torch 默认多线程)."""
    algo, seed, n_ep, llm = args
    r, h, f1 = train_one_run(algo, seed, n_episodes=n_ep, llm_name=llm,
                             verbose=False)
    return (algo, llm, seed, r, h, f1)


def run_algo_comparison(seeds=None, n_episodes=None, verbose=True,
                        parallel=False):
    """跑 6 算法对比 (Fig.3a/b, Fig.4)."""
    seeds = list(range(C.N_SEEDS_ALGO)) if seeds is None else seeds
    n_episodes = C.N_EPISODES if n_episodes is None else n_episodes
    out_r, out_h, out_f1 = {}, {}, {}
    # 构造所有 (algo, seed) 任务
    tasks = [(algo, sd, n_episodes, "DeepSeek") for algo in C.ALGO_LIST
             for sd in seeds]
    t0 = time.time()
    if parallel and len(tasks) > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            # 限制 worker 数避免过度竞争 (12 核跑 6 worker × 1 线程)
            nproc = min(len(tasks), 6)
            with ctx.Pool(nproc) as pool:
                results = pool.map(_worker, tasks)
        except Exception as e:
            if verbose:
                print(f"  parallel failed ({e}), fallback to serial")
            results = [_worker(t) for t in tasks]
    else:
        results = []
        for i, t in enumerate(tasks):
            res = _worker(t)
            results.append(res)
            if verbose:
                a, l, sd, r, h, f1 = res
                print(f"  [{i+1}/{len(tasks)}] {a}/{l} seed={sd} "
                      f"done: reward={r[-15:].mean():.2f} ho={h[-15:].mean():.1f}",
                      flush=True)
    # 整理
    for algo in C.ALGO_LIST:
        Rs, Hs, F1s = [], [], []
        for (a, llm, sd, r, h, f1) in results:
            if a == algo:
                Rs.append(r); Hs.append(h); F1s.append(f1[-1])
        out_r[algo] = np.array(Rs)
        out_h[algo] = np.array(Hs)
        out_f1[algo] = np.array(F1s)
    if verbose:
        for algo in C.ALGO_LIST:
            print(f"[{algo}] {len(seeds)} seeds; "
                  f"final reward={out_r[algo][:, -50:].mean():.2f} "
                  f"ho={out_h[algo][:, -50:].mean():.2f}")
        print(f"  Phase 1 total: {time.time()-t0:.1f}s")
    return out_r, out_h, out_f1


def run_llm_comparison(seeds=None, n_episodes=None, verbose=True, parallel=False):
    """跑 5 LLM 对比 (Fig.3c, Table II/III) - 全部用 LTQC-DAM 框架."""
    seeds = list(range(C.N_SEEDS_LLM)) if seeds is None else seeds
    n_episodes = C.N_EPISODES if n_episodes is None else n_episodes
    tasks = [("LTQC-DAM", sd, n_episodes, llm) for llm in C.LLM_LIST
             for sd in seeds]
    t0 = time.time()
    if parallel and len(tasks) > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            nproc = min(len(tasks), 6)
            with ctx.Pool(nproc) as pool:
                results = pool.map(_worker, tasks)
        except Exception as e:
            if verbose:
                print(f"  parallel failed ({e}), fallback to serial")
            results = [_worker(t) for t in tasks]
    else:
        results = []
        for i, t in enumerate(tasks):
            res = _worker(t)
            results.append(res)
            if verbose:
                a, l, sd, r, h, f1 = res
                print(f"  [{i+1}/{len(tasks)}] {a}/{l} seed={sd} "
                      f"done: reward={r[-15:].mean():.2f} ho={h[-15:].mean():.1f}",
                      flush=True)
    out = {}
    for llm in C.LLM_LIST:
        Rs = []
        for (a, l, sd, r, h, f1) in results:
            if l == llm:
                Rs.append(r)
        out[llm] = np.array(Rs)
    if verbose:
        for llm in C.LLM_LIST:
            final = out[llm][:, -50:].mean()
            print(f"[LLM={llm}] {len(seeds)} seeds; "
                  f"final reward={final:.2f} "
                  f"(target {C.LLM_QUALITY[llm]['target_mean']})")
        print(f"  Phase 2 total: {time.time()-t0:.1f}s")
    return out
