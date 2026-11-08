# -*- coding: utf-8 -*-
"""
eval_rl.py
================================================================================
Fig.6/9 用训练后 DQN 评估。

算法映射（Phase1 仅空间层，时间层暂用规则）：
  MO（均衡）= DQN(0.4,0.3,0.3) 选目标
  SL       = DQN(0.8,0.1,0.1)
  SQ       = DQN(0.1,0.8,0.1)
  ST       = DQN(0.1,0.1,0.8)
  Phase1 中 MO-HRL 与 MO-RL 空间层相同（同一 DQN），差异留 Phase2 时间层。

策略注入：构造 policy_fn 闭包，在 decide_one_step 的"选目标候选"位置用 DQN
greedy 选；触发/成败逻辑完全不变。

【诚实保真度边界】DQN 在 20 用户训练几何上训，评估用 100 用户全场景，是
泛化（非过拟合）。论文无种子/无代码，定性一致（趋势/排序）即可。
================================================================================
"""

import os
import sys
import time
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import SEED, USER, OUTPUT_DIR
from dqn_spatial import QNet
import simulation as sim
import plotting as plot
import train_rl


# 算法名 → DQN 权重 tag（与 train_rl.FIG5_WEIGHTS 对齐）
# Phase1：MO-HRL 与 MO-RL 用同一均衡 DQN，时间层差异留 Phase2。
# 单因子用倾斜权重 DQN（论文 SL/SQ/ST 是单因子奖励 RL 智能体）。
# 注意：train_rl 训了 (0.6,0.2,0.2)/(0.2,0.6,0.2)/(0.2,0.2,0.6)/(0.4,0.3,0.3)，
# 这里 SL/SQ/ST 用倾斜权重 (0.8/0.1/0.1) 等——若该 tag 没训则回退到最接近的
# 已训权重，并在控制台诚实标注。
ALGO_DQN_WEIGHTS = {
    'MO-HRL': (0.4, 0.3, 0.3),
    'MO-RL':  (0.4, 0.3, 0.3),   # Phase1 空间层同 MO-HRL
    'SL-RL':  (0.6, 0.2, 0.2),   # 负载主导（用最接近的已训 0.6/0.2/0.2）
    'SQ-RL':  (0.2, 0.6, 0.2),   # 信号主导
    'ST-RL':  (0.2, 0.2, 0.6),   # 时间主导
}


def _weights_tag(w):
    return f'{w[0]:.1f}_{w[1]:.1f}_{w[2]:.1f}'


def load_all_dqns(ckpt_dir=None):
    """加载所有已训练 DQN。返回 dict {tag: (qnet, max_vis, state_dim)}。
    若某 tag 的 ckpt 不存在跳过（回退到贪心）。
    """
    from train_rl import CKPT_DIR, load_qnet
    ckpt_dir = ckpt_dir or CKPT_DIR
    dqns = {}
    for name, w in ALGO_DQN_WEIGHTS.items():
        tag = _weights_tag(w)
        path = os.path.join(ckpt_dir, f'dqn_{tag}.pt')
        if os.path.exists(path):
            qnet, mv, sd = load_qnet(tag)
            dqns[name] = (qnet, mv, sd)
        else:
            dqns[name] = None
    return dqns


def make_policy_fn(qnet, max_vis, state_dim, device='cpu'):
    """构造 decide_one_step 的 policy_fn 回调。
    签名: policy_fn(snap, u, valid_k, H, tau, L, psi, cur_sat, max_vis_cap)
          -> (chosen_local_idx_in_valid_k, chosen_sat)
    """
    qnet.eval()
    device = torch.device(device)

    @torch.no_grad()
    def policy_fn(snap, u, valid_k, H, tau, L, psi, cur_sat, max_vis_cap):
        n_cand = len(valid_k)
        if n_cand == 0:
            return None, None
        # 构造 state（与 rl_env._build_state 一致）
        cand = np.zeros((max_vis_cap, 4), dtype=np.float32)
        n_fill = min(n_cand, max_vis_cap)
        cand[:n_fill, 0] = L[:n_fill]
        cand[:n_fill, 1] = H[:n_fill]
        cand[:n_fill, 2] = tau[:n_fill]
        cand[:n_fill, 3] = 1.0
        cand_flat = cand.flatten()
        # 当前服务星特征
        sat_ids = snap.visible_idx[u][valid_k]
        if cur_sat >= 0 and cur_sat in sat_ids:
            cur_k_local = int(np.where(sat_ids == cur_sat)[0][0])
            cur_feat = np.array([L[cur_k_local], H[cur_k_local],
                                 tau[cur_k_local], 1.0], dtype=np.float32)
        else:
            cur_feat = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = np.concatenate([cand_flat, cur_feat]).astype(np.float32)

        # DQN greedy 选（masked）
        s = torch.from_numpy(state).unsqueeze(0).to(device)
        q = qnet(s)[0].cpu().numpy()
        mask = np.zeros(max_vis_cap, dtype=bool)
        mask[:n_fill] = True
        q_masked = np.where(mask, q, -np.inf)
        a = int(np.argmax(q_masked))
        if a >= n_cand:
            # 越界回退
            return None, None
        chosen_local = a  # valid_k 内下标
        chosen_sat = int(sat_ids[a])
        return chosen_local, chosen_sat

    return policy_fn


def build_policies(dqns):
    """把 {algo: (qnet,mv,sd)} 转成 {algo: policy_fn}。缺 DQN 的算法不加入
    （回退贪心，simulate_trajectory 自动处理）。"""
    policies = {}
    for name, info in dqns.items():
        if info is None:
            continue
        qnet, mv, sd = info
        policies[name] = make_policy_fn(qnet, mv, sd)
    return policies


# ============================================================================
# 时间层（PPO+GRU）策略回调：MO-HRL 分层评估
# ============================================================================
def make_temporal_fn(actor, k_hist=10, ins_dim=5, device='cpu',
                     tau_ref_s=60.0, tau_force_s=3.0, seed=0):
    """构造 decide_one_step 的 temporal_fn 回调。

    内部维护：(1) 每用户服务历史 deque(k_hist, [L̃, H̃, τ̃]) — GRU 输入；
              (2) GRU 无状态（每次用全序列前向，不依赖隐状态跨步——评估场景更稳健）。

    actor : ppo_temporal.GRUActor（已加载权重，eval 模式）
    返回 temporal_fn(ctx) -> int {0=keep, 1=trigger}。ctx 见 decide_one_step。
    """
    import numpy as np
    import torch
    from collections import deque

    actor.eval()
    device = torch.device(device)
    rng = np.random.default_rng(seed)

    # 每用户历史 deque（评估场景用户数可能不同于训练，按 ctx['u'] 索引）
    hist = {}

    def _get_hist(u):
        if u not in hist:
            hist[u] = deque(maxlen=k_hist)
        return hist[u]

    def temporal_fn(ctx):
        u = ctx['u']
        snap = ctx['snap']
        cur_sat = ctx['cur_sat']
        valid_k = ctx['valid_k']
        H = ctx['H']; tau = ctx['tau']; L = ctx['L']; psi = ctx['psi']
        best_psi = ctx['best_psi']
        urgent = ctx.get('urgent', False)

        # 紧急时强制 trigger（安全网，但 decide_one_step 内也会兜底）
        # 这里让 PPO 看到 urgent 后倾向 trigger，但仍由网络决定（PPO 应学到）
        # —— 实际安全网在 decide_one_step 已有（urgent + temporal_fn=0 → force_loss）。

        # 构造 o_seq：历史 deque 末尾对齐
        h = _get_hist(u)
        seq = np.zeros((k_hist, 3), dtype=np.float32)
        h_arr = np.array(list(h), dtype=np.float32) if len(h) > 0 \
            else np.zeros((0, 3), dtype=np.float32)
        if h_arr.shape[0] > 0:
            seq[-h_arr.shape[0]:] = h_arr

        # 构造 o_ins：5 维（与训练一致）
        # [θ(当前仰角归一), ψ(当前), Δψ(增益), 距上次切换/τ_ref, τ̃(当前)]
        sat_ids = snap.visible_idx[u][valid_k]
        if cur_sat >= 0 and cur_sat in sat_ids:
            cur_k_local = int(np.where(sat_ids == cur_sat)[0][0])
        else:
            cur_k_local = int(np.argmax(psi))
        cur_elev = float(snap.visible_elev[u, valid_k[cur_k_local]])
        theta_norm = float(np.clip(np.sin(np.deg2rad(cur_elev)), 0.0, 1.0))
        cur_psi = float(ctx['cur_psi']) if 'cur_psi' in ctx else float(psi[cur_k_local])
        gain_psi = max(0.0, best_psi - cur_psi)
        t_since_ho = ctx.get('t_since_ho', 1e9)
        t_since_norm = float(np.clip(t_since_ho / tau_ref_s, 0.0, 5.0)) / 5.0
        cur_tau = float(tau[cur_k_local])
        ins = np.array([theta_norm, cur_psi, gain_psi, t_since_norm, cur_tau],
                       dtype=np.float32)

        # GRU actor greedy（argmax 动作）
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
        ins_t = torch.from_numpy(ins).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = actor(seq_t, ins_t)
            a = int(torch.argmax(logits, dim=1).item())

        # 更新该用户历史（用当前服务星特征，下一时隙 GRU 输入）
        cur_L = float(L[cur_k_local]); cur_H = float(H[cur_k_local])
        h.append([cur_L, cur_H, cur_tau])

        return a

    return temporal_fn


def build_temporals(actors_meta, k_hist=10, ins_dim=5):
    """把 {algo: (actor, meta)} 转成 {algo: temporal_fn}。"""
    temporals = {}
    for name, (actor, meta) in actors_meta.items():
        temporals[name] = make_temporal_fn(actor, k_hist=k_hist, ins_dim=ins_dim)
    return temporals


def run_eval(seed=SEED, n_users=None, max_vis_cap=20, save_figs=True,
             reuse_dqns=None):
    """用训练后 DQN 跑 Fig.6/7/9。
    reuse_dqns: 若提供 {algo:(qnet,mv,sd)} 则复用，否则从磁盘加载。
    返回 (f6, f7, f9)。
    """
    n_users = n_users or USER['num_users']
    print('=' * 78)
    print('  Fig.6/7/9 评估（DQN 空间层 + 规则时间层）')
    print('=' * 78)

    dqns = reuse_dqns or load_all_dqns()
    # 打印 DQN 加载状态
    print('  DQN 加载状态:')
    for name in ALGO_DQN_WEIGHTS:
        info = dqns.get(name)
        tag = _weights_tag(ALGO_DQN_WEIGHTS[name])
        if info is not None:
            print(f'    {name:8} <- DQN({tag})  [已加载]')
        else:
            print(f'    {name:8} <- 贪心(回退)  [DQN {tag} 未找到]')
    policies = build_policies(dqns)

    t0 = time.time()
    print('  [Fig.6] 参数化场景扫描（DQN）...')
    f6 = sim.run_fig6(n_users=n_users, seed=seed,
                      policies=policies, max_vis_cap=max_vis_cap)
    print('  [Fig.7] 3 纬度带（DQN）...')
    f7 = sim.run_fig7(n_users=n_users, seed=seed,
                      policies=policies, max_vis_cap=max_vis_cap)
    print('  [Fig.9] θ_min 扫描（DQN）...')
    f9 = sim.run_fig9(n_users=n_users, seed=seed,
                      policies=policies, max_vis_cap=max_vis_cap)
    print(f'  评估总耗时: {time.time()-t0:.1f}s')

    if save_figs:
        # 覆盖之前贪心版（同名）
        p6 = plot.plot_fig6(f6, os.path.join(OUTPUT_DIR, 'fig06_handover_factors.png'))
        p7 = plot.plot_fig7(f7, os.path.join(OUTPUT_DIR, 'fig07_success_rate_latitude.png'))
        p9 = plot.plot_fig9(f9, os.path.join(OUTPUT_DIR, 'fig09_transmission_rate.png'))
        print(f'  Fig.6 -> {os.path.basename(p6)}')
        print(f'  Fig.7 -> {os.path.basename(p7)}')
        print(f'  Fig.9 -> {os.path.basename(p9)}')

    # 打印关键涌现值 vs 论文
    print()
    print('  [Fig.6(a) 成功率 vs H̃]  涌现首(劣)→末(优):')
    for name in ['MO-HRL', 'MO-RL', 'SL-RL', 'SQ-RL', 'ST-RL']:
        y = f6['a']['curves'][name]
        yv = y[~np.isnan(y)]
        if len(yv):
            print(f'    {name:8}: {yv[0]:.3f} -> {yv[-1]:.3f}  (均值 {yv.mean():.3f})')
    print('  [Fig.6(b) 成功率 vs L̃]:')
    for name in ['MO-HRL', 'MO-RL', 'SL-RL', 'SQ-RL', 'ST-RL']:
        y = f6['b']['curves'][name]
        yv = y[~np.isnan(y)]
        if len(yv):
            print(f'    {name:8}: {yv[0]:.3f} -> {yv[-1]:.3f}  (均值 {yv.mean():.3f})')
    print('  [Fig.9 n=20 速率 Mbps]:')
    for name in ['MO-HRL', 'MO-RL', 'SL-RL', 'SQ-RL', 'ST-RL']:
        y = f9['curves'][name][-1]
        print(f'    {name:8}: {y:.0f}')

    return f6, f7, f9


# ============================================================================
# MO-HRL 分层评估（时间层 PPO+GRU + 空间层 DQN）
#   MO-HRL = 训练后 PPO+GRU(case234) 决定"何时切" + 训练后 DQN(0.4,0.3,0.3) 决定"切到哪"
#   其余算法（MO-RL/SL/SQ/ST）仍用 Phase1 的 DQN 空间层（规则触发）。
# ============================================================================
def run_eval_hierarchical(seed=SEED, n_users=None, max_vis_cap=20,
                          save_figs=True, reuse_dqns=None,
                          ppo_case='case234', fallback_to_phase1=True):
    """MO-HRL 分层评估：重生成 Fig.6/7（MO-HRL 用 PPO+GRU 时间层 + DQN 空间层）。

    fallback_to_phase1 : 若 PPO+GRU ckpt 不存在，回退到 Phase1（DQN 空间层 + 规则
                        时间层），并诚实标注。
    返回 (f6, f7, hierarchical_ok) —— hierarchical_ok=True 表示真分层跑通。
    """
    from train_rl import load_ppo
    n_users = n_users or USER['num_users']
    print('=' * 78)
    print('  Fig.6/7 MO-HRL 分层评估（时间层 PPO+GRU + 空间层 DQN）')
    print('=' * 78)

    dqns = reuse_dqns or load_all_dqns()
    policies = build_policies(dqns)

    # 加载 PPO+GRU（MO-HRL 时间层）
    try:
        actor, critic, meta = load_ppo(ppo_case)
        k_hist = meta['k_hist']; ins_dim = meta['ins_dim']
        print(f'  PPO+GRU 加载: case={ppo_case}, '
              f'k_hist={k_hist}, ins_dim={ins_dim}  [OK]')
        temporals = {'MO-HRL': make_temporal_fn(actor, k_hist=k_hist,
                                                ins_dim=ins_dim)}
        hierarchical_ok = True
    except FileNotFoundError as e:
        print(f'  [WARN] PPO+GRU 未找到: {e}')
        if fallback_to_phase1:
            print(f'  -> 回退 Phase1（DQN 空间层 + 规则时间层），MO-HRL 退化为 DQN 版')
            temporals = {}
            hierarchical_ok = False
        else:
            raise

    t0 = time.time()
    print('  [Fig.6] 参数化场景扫描（分层 MO-HRL）...')
    f6 = sim.run_fig6(n_users=n_users, seed=seed,
                      policies=policies, max_vis_cap=max_vis_cap,
                      temporals=temporals)
    print('  [Fig.7] 3 纬度带（分层 MO-HRL）...')
    f7 = sim.run_fig7(n_users=n_users, seed=seed,
                      policies=policies, max_vis_cap=max_vis_cap,
                      temporals=temporals)
    print(f'  评估总耗时: {time.time()-t0:.1f}s')

    if save_figs:
        tag = '_hier' if hierarchical_ok else '_phase1fallback'
        p6 = plot.plot_fig6(f6, os.path.join(OUTPUT_DIR,
                            f'fig06_handover_factors{tag}.png'))
        p7 = plot.plot_fig7(f7, os.path.join(OUTPUT_DIR,
                            f'fig07_success_rate_latitude{tag}.png'))
        print(f'  Fig.6 -> {os.path.basename(p6)}')
        print(f'  Fig.7 -> {os.path.basename(p7)}')

    # 打印 MO-HRL 分层涌现值
    print()
    print(f'  [MO-HRL {"分层" if hierarchical_ok else "Phase1回退"}] Fig.6(a) 成功率 vs H̃:')
    y = f6['a']['curves']['MO-HRL']
    yv = y[~np.isnan(y)]
    if len(yv):
        print(f'    MO-HRL: {yv[0]:.3f} -> {yv[-1]:.3f}  (均值 {yv.mean():.3f})')
    print(f'  [MO-HRL] Fig.7 成功率 低/中/中高纬:')
    y = f7['curves']['MO-HRL']
    print(f'    MO-HRL: {y[0]:.3f}/{y[1]:.3f}/{y[2]:.3f}')

    return f6, f7, hierarchical_ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hier', action='store_true',
                    help='跑分层 MO-HRL 评估（需先训 PPO+GRU）')
    args = ap.parse_args()
    if args.hier:
        run_eval_hierarchical()
    else:
        run_eval()
    run_eval()
