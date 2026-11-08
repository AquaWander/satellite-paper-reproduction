# -*- coding: utf-8 -*-
"""
train_rl.py
================================================================================
Fig.5 编排：4 组 ψ 权重各训一 DQN，记录 reward 曲线。

  Fig.5 = 空间层收敛，4 组权重：
    (0.6,0.2,0.2) / (0.2,0.6,0.2) / (0.2,0.2,0.6) / (0.4,0.3,0.3)
  论文 Fig.5 参考形状（仅定性）：(0.4,0.3,0.3) 收敛最高，其余依次低。

输出：
  output/rl_ckpt/fig5_curves.json  — 4 条 reward-vs-episode 曲线
  output/rl_ckpt/dqn_<wL>_<wH>_<wT>.pt — 训练后的 Q 网络（供 eval_rl 用）

【诚实保真度边界】论文无种子/无代码，真实训练 + 定性一致（曲线形状/排序），
不查表/硬编码输出。
================================================================================
"""

import os
import json
import time
import numpy as np
import torch

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import SEED, OUTPUT_DIR
from dqn_spatial import train


# Fig.5 的 4 组权重配置（与 eval_rl 的算法映射对应）
FIG5_WEIGHTS = [
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6),
    (0.4, 0.3, 0.3),  # MO-HRL/MO-RL 均衡权重，论文中收敛最高
]

CKPT_DIR = os.path.join(OUTPUT_DIR, 'rl_ckpt')
os.makedirs(CKPT_DIR, exist_ok=True)


def _weights_tag(w):
    return f'{w[0]:.1f}_{w[1]:.1f}_{w[2]:.1f}'


def run_fig5(n_episodes=300, seed=SEED, n_users=20, episode_s=200.0,
             save=True):
    """训练 4 组权重的 DQN，返回 dict {weights_tag: reward_curve}。"""
    print('=' * 78)
    print('  Fig.5 空间层 DQN 训练（4 组 ψ 权重）')
    print('=' * 78)
    print(f'  配置: n_users={n_users}, episode={episode_s}s, '
          f'n_episodes={n_episodes}/权重, seed={seed}')
    print(f'  权重配置: {FIG5_WEIGHTS}')
    print()

    curves = {}
    qnets = {}
    t_total = time.time()

    for i, w in enumerate(FIG5_WEIGHTS):
        tag = _weights_tag(w)
        print(f'  [{i+1}/4] 训练 DQN, 权重 w=(L={w[0]}, H={w[1]}, τ={w[2]})  '
              f'[{tag}]')
        env_kwargs = dict(n_users=n_users, episode_s=episode_s)
        curve, qnet, env, losses = train(
            weights=w, n_episodes=n_episodes, seed=seed + i,
            env_kwargs=env_kwargs, log_every=max(n_episodes // 15, 10))
        curves[tag] = [float(x) for x in curve]
        qnets[tag] = (qnet, env.max_vis, env.state_dim)

        # 存 Q 网络
        if save:
            ckpt_path = os.path.join(CKPT_DIR, f'dqn_{tag}.pt')
            torch.save({
                'state_dict': qnet.state_dict(),
                'max_vis': env.max_vis,
                'state_dim': env.state_dim,
                'weights': list(w),
            }, ckpt_path)
            print(f'    Q 网络已存: {os.path.basename(ckpt_path)}')
        print()

    elapsed = time.time() - t_total
    print(f'  Fig.5 训练总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')

    # 打印收敛值对比（论文 Fig.5 定性参考：均衡权重最高）
    print()
    print('  各权重收敛 reward（末 20 episode 均值）:')
    final_vals = {}
    for tag, curve in curves.items():
        v = float(np.mean(curve[-20:]))
        final_vals[tag] = v
        print(f'    w=({tag.replace("_", ",")}): {v:8.2f}')
    eq = final_vals[_weights_tag((0.4, 0.3, 0.3))]
    others = [v for t, v in final_vals.items()
              if t != _weights_tag((0.4, 0.3, 0.3))]
    print(f'  均衡权重(0.4,0.3,0.3)={eq:.2f} vs 单因子均值={np.mean(others):.2f}  '
          f'(论文预期: 均衡 ≥ 单因子)')
    print(f'  收敛排序: {sorted(final_vals.items(), key=lambda x: -x[1])}')

    # 存曲线 JSON
    if save:
        out = dict(
            weights=[list(w) for w in FIG5_WEIGHTS],
            tags=[_weights_tag(w) for w in FIG5_WEIGHTS],
            curves=curves,
            config=dict(n_users=n_users, episode_s=episode_s,
                        n_episodes=n_episodes, seed=seed),
            final_values=final_vals,
        )
        out_path = os.path.join(CKPT_DIR, 'fig5_curves.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'  曲线已存: {out_path}')

    return curves, qnets


def load_qnet(tag):
    """加载某权重的训练后 Q 网络。返回 (qnet, max_vis, state_dim)。"""
    from dqn_spatial import QNet
    ckpt_path = os.path.join(CKPT_DIR, f'dqn_{tag}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Q 网络未找到: {ckpt_path}（先跑 train_rl.run_fig5）')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    qnet = QNet(ckpt['state_dim'], ckpt['max_vis'])
    qnet.load_state_dict(ckpt['state_dict'])
    qnet.eval()
    return qnet, ckpt['max_vis'], ckpt['state_dim']


# ============================================================================
# Fig.4：时间层 PPO+GRU 收敛（3 种奖励配置）
#   Fig.4 = 时间层收敛，3 种 reward_case：
#     'case2'   : 只激活 Case1+Case2（紧急切换奖励）
#     'case23'  : Case1+Case2+Case3（+ 改善切换奖励）
#     'case234' : 全部激活（+ 乒乓惩罚）— MO-HRL 完整版，论文中收敛最高
#   每配置训一 PPO+GRU，"切到哪"由已训练的空间层 DQN(0.4,0.3,0.3) greedy 决定。
#   论文 Fig.4 参考形状（仅定性）：case234 收敛最高，case2 最低/最慢。
# ============================================================================
FIG4_CASES = ['case2', 'case23', 'case234']


def _make_spatial_policy_fn(tag='0.4_0.3_0.3'):
    """加载空间层 DQN，构造 eval_rl.make_policy_fn 风格的 greedy 策略。
    返回 (policy_fn, qnet, max_vis, state_dim)。"""
    from eval_rl import make_policy_fn
    qnet, mv, sd = load_qnet(tag)
    pf = make_policy_fn(qnet, mv, sd)
    return pf, qnet, mv, sd


def run_fig4(n_episodes=100, seed=SEED, n_users=20, episode_s=200.0,
             spatial_tag='0.4_0.3_0.3', save=True):
    """训练 3 种 reward_case 的 PPO+GRU，返回 dict {case: reward_curve}。

    spatial_tag : 空间层 DQN 权重 tag（决定 trigger=1 时切到哪颗星）。
    """
    import ppo_temporal as ppo
    print('=' * 78)
    print('  Fig.4 时间层 PPO+GRU 训练（3 种 reward_case）')
    print('=' * 78)
    print(f'  配置: n_users={n_users}, episode={episode_s}s, '
          f'n_episodes={n_episodes}/case, seed={seed}')
    print(f'  reward_case 配置: {FIG4_CASES}')
    print(f'  空间层 DQN: dqn_{spatial_tag}.pt（trigger=1 时 greedy 选目标）')
    print()

    # 加载空间层 DQN（3 case 共用同一空间策略）
    print(f'  加载空间层 DQN({spatial_tag})...')
    spatial_fn, qnet, mv, sd = _make_spatial_policy_fn(spatial_tag)
    print(f'    OK (max_vis={mv}, state_dim={sd})')
    print()

    curves = {}
    actors = {}
    critics = {}
    t_total = time.time()

    for i, case in enumerate(FIG4_CASES):
        print(f'  [{i+1}/3] 训练 PPO+GRU, reward_case={case}')
        curve, actor, critic, env = ppo.train(
            reward_case=case, spatial_policy_fn=spatial_fn,
            n_episodes=n_episodes, seed=seed + i,
            n_users=n_users, episode_s=episode_s,
            log_every=max(n_episodes // 10, 5))
        curves[case] = [float(x) for x in curve]
        actors[case] = actor
        critics[case] = critic

        if save:
            ckpt_path = os.path.join(CKPT_DIR, f'ppo_{case}.pt')
            torch.save({
                'actor_state': actor.state_dict(),
                'critic_state': critic.state_dict(),
                'k_hist': 10,
                'ins_dim': 5,
                'state_flat_dim': env.state_flat_dim,
                'reward_case': case,
                'spatial_tag': spatial_tag,
            }, ckpt_path)
            print(f'    PPO+GRU 已存: {os.path.basename(ckpt_path)}')
        print()

    elapsed = time.time() - t_total
    print(f'  Fig.4 训练总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')

    # 收敛值对比（论文 Fig.4 定性参考：case234 最高，case2 最低）
    print()
    print('  各 reward_case 收敛 reward（末 20 episode 均值）:')
    final_vals = {}
    for case, curve in curves.items():
        v = float(np.mean(curve[-20:]))
        final_vals[case] = v
        print(f'    {case:10}: {v:8.2f}')
    order = sorted(final_vals.items(), key=lambda x: -x[1])
    print(f'  收敛排序(高→低): {[(c, round(v,1)) for c, v in order]}')
    expected = ['case234', 'case23', 'case2']
    got = [c for c, _ in order]
    match = (got == expected)
    print(f'  论文预期排序 case234 ≥ case23 ≥ case2? '
          f'实际 {got} → {"OK" if match else "WARN（PPO 噪声/未完全收敛，已诚实记录）"}')

    if save:
        out = dict(
            cases=list(FIG4_CASES),
            curves=curves,
            config=dict(n_users=n_users, episode_s=episode_s,
                        n_episodes=n_episodes, seed=seed,
                        spatial_tag=spatial_tag),
            final_values=final_vals,
            convergence_order=got,
            expected_order=expected,
            order_matches_paper=bool(match),
        )
        out_path = os.path.join(CKPT_DIR, 'fig4_curves.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'  曲线已存: {out_path}')

    return curves, actors, critics


def load_ppo(case):
    """加载某 reward_case 的训练后 PPO+GRU。返回 (actor, critic, meta)。"""
    from ppo_temporal import GRUActor, Critic
    ckpt_path = os.path.join(CKPT_DIR, f'ppo_{case}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'PPO 未找到: {ckpt_path}（先跑 train_rl.run_fig4）')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    k_hist = ckpt['k_hist']
    ins_dim = ckpt['ins_dim']
    state_flat_dim = ckpt['state_flat_dim']
    actor = GRUActor(k_hist=k_hist, ins_dim=ins_dim)
    actor.load_state_dict(ckpt['actor_state'])
    actor.eval()
    critic = Critic(state_flat_dim)
    critic.load_state_dict(ckpt['critic_state'])
    critic.eval()
    meta = dict(reward_case=ckpt['reward_case'],
                spatial_tag=ckpt['spatial_tag'],
                k_hist=k_hist, ins_dim=ins_dim,
                state_flat_dim=state_flat_dim)
    return actor, critic, meta


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', choices=['5', '4', 'both'], default='5')
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--users', type=int, default=20)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--quick', action='store_true', help='快速验证（100/50 ep）')
    args = ap.parse_args()
    n_ep = 100 if args.quick else args.episodes
    if args.fig == '5':
        run_fig5(n_episodes=n_ep, seed=args.seed, n_users=args.users)
    elif args.fig == '4':
        n4 = 100 if args.quick else n_ep
        run_fig4(n_episodes=n4, seed=args.seed, n_users=args.users)
    else:
        run_fig5(n_episodes=n_ep, seed=args.seed, n_users=args.users)
        n4 = 100 if args.quick else n_ep
        run_fig4(n_episodes=n4, seed=args.seed, n_users=args.users)
