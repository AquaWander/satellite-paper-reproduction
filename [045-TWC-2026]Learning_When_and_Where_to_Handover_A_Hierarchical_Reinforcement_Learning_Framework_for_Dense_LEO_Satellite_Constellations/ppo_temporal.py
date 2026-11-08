# -*- coding: utf-8 -*-
"""
ppo_temporal.py
================================================================================
时间层 PPO + GRU（论文 Eq.22-29 结构）。

  Actor  : GRU(k=10, in=3 → hidden=256 → 128) + FC(128+5_instant → 64 → softmax(2))
           把序列 o_seq (Eq.17) 喂入 GRU，取末态与瞬时 o_ins (Eq.18) 拼接再 FC 出动作分布。
  Critic : FC(state_flat=35 → 256 → 128 → 64 → V)
  PPO    : clip ε=0.2、GAE λ=0.95、γ=0.99、Adam lr=3e-4（论文值），
           entropy bonus、normalize advantage、n_epochs 多轮/episode batch。

【诚实保真度边界 / HONEST FIDELITY BOUNDARY】
论文无种子/无代码，即使训练也不可能逐点一致。目标是：
  真实训练 + 定性一致（Fig.4 收敛曲线形状、Fig.6/7 趋势与算法排序）。
  绝不查表/硬编码输出。所有 reward/transition 由前向物理仿真涌现。
  若 PPO+GRU 在合理超参下不收敛（曲线不升/振荡），诚实记录并分析，不伪造收敛。
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import SEED
from rl_env import TemporalEnv


# ============================================================================
# GRU Actor（论文 Eq.22-26）：序列输入 + 瞬时拼接 → softmax(2)
# ============================================================================
class GRUActor(nn.Module):
    """GRU 编码历史序列 o_seq(k,3)，末态拼瞬时 o_ins(5)，FC 输出二值动作分布。

    结构：GRU(input=3, hidden=256, num_layers=2, batch_first) → 取末时刻隐状态(256)
          → FC(256 → 128) → ReLU
          → 拼接 o_ins(5) → Linear(133 → 64) → ReLU → Linear(64 → 2) → softmax
    """

    def __init__(self, k_hist=10, ins_dim=5, gru_hidden=256,
                 gru_layers=2, fc_hidden=128):
        super().__init__()
        self.k_hist = k_hist
        self.ins_dim = ins_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        # GRU 编码序列 (输入维度=3：[L̃, H̃, τ̃])
        self.gru = nn.GRU(input_size=3, hidden_size=gru_hidden,
                          num_layers=gru_layers, batch_first=True)
        # FC：GRU 末态(gru_hidden) → fc_hidden
        self.fc_seq = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden), nn.ReLU(),
        )
        # FC：拼瞬时(ins_dim) → 64 → 2
        self.fc_out = nn.Sequential(
            nn.Linear(fc_hidden + ins_dim, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, seq, ins):
        """
        seq : (B, k_hist, 3) 历史序列
        ins : (B, ins_dim) 瞬时向量
        返回 : (B, 2) 动作 logits（softmax 前的值）
        """
        # GRU 编码（batch_first=True，输入 (B, k, 3)）
        out, h_n = self.gru(seq)              # out: (B, k, hidden), h_n: (layers, B, hidden)
        # 取最后一层的末时刻隐状态
        last_hidden = h_n[-1]                  # (B, hidden)
        # FC 编码
        feat = self.fc_seq(last_hidden)        # (B, fc_hidden)
        # 拼接瞬时向量
        combined = torch.cat([feat, ins], dim=1)  # (B, fc_hidden + ins_dim)
        logits = self.fc_out(combined)          # (B, 2)
        return logits

    def get_dist(self, seq, ins):
        """返回动作分布（Categorical）。"""
        logits = self.forward(seq, ins)
        return torch.distributions.Categorical(logits=logits)


# ============================================================================
# Critic（论文 Eq.27-28）：FC 扁平状态 → V
# ============================================================================
class Critic(nn.Module):
    """FC(state_flat → 256 → 128 → 64 → V)。
    state_flat = o_seq.flatten(k*3) + o_ins(ins_dim)。
    """

    def __init__(self, state_flat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_flat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state_flat):
        return self.net(state_flat).squeeze(-1)  # (B,)


# ============================================================================
# PPO+GRU 训练循环
# ============================================================================
def _flatten_state(seq, ins):
    """把 (seq(k,3), ins(5)) 展平为 (k*3 + 5,) 向量（给 critic）。"""
    return np.concatenate([seq.flatten(), ins]).astype(np.float32)


def _gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation。
    rewards : list[float]，长度 T（每步奖励）
    values  : list[float]，长度 T（每步 critic V(s_t)）
    last_value : 标量，V(s_{T+1})（终止=0）
    返回 (advantages[T], returns[T])。
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    next_value = last_value
    # 倒序计算 GAE
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


def train(reward_case, spatial_policy_fn, n_episodes=300, seed=SEED,
          n_users=20, episode_s=200.0, k_hist=10,
          lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
          n_epochs=4, batch_size=64, entropy_coef=0.01,
          value_coef=0.5, max_grad_norm=0.5,
          env_reuse=None, log_every=20, device='cpu',
          env_kwargs=None):
    """训练一个时间层 PPO+GRU。

    reward_case       : {'case2','case23','case234'}
    spatial_policy_fn : 空间层 DQN greedy（决定 trigger=1 切到哪）
    返回 (reward_curve, actor, critic, env)。
    """
    import time
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = env_kwargs or {}
    env = TemporalEnv(
        weights=(0.4, 0.3, 0.3), spatial_policy_fn=spatial_policy_fn,
        n_users=n_users, episode_s=episode_s, k_hist=k_hist,
        reward_case=reward_case, seed=seed, env_reuse=env_reuse,
        **env_kwargs,
    )
    state_flat_dim = env.state_flat_dim
    device = torch.device(device)

    actor = GRUActor(k_hist=k_hist, ins_dim=env.ins_dim).to(device)
    critic = Critic(state_flat_dim).to(device)
    opt_a = optim.Adam(actor.parameters(), lr=lr)
    opt_c = optim.Adam(critic.parameters(), lr=lr)

    reward_curve = []
    switch_curve = []
    t_start = time.time()

    for ep in range(n_episodes):
        # ---- 收集一条 episode 轨迹 ----
        seq_buf = []    # list of (k,3)
        ins_buf = []    # list of (5,)
        flat_buf = []   # list of (state_flat_dim,)
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []

        seq, ins = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_switches = 0
        while seq is not None:
            flat = _flatten_state(seq, ins)
            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)   # (1,k,3)
            ins_t = torch.from_numpy(ins).unsqueeze(0).to(device)   # (1,5)
            flat_t = torch.from_numpy(flat).unsqueeze(0).to(device) # (1,D)

            with torch.no_grad():
                dist = actor.get_dist(seq_t, ins_t)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = critic(flat_t)
            a = int(action.item())
            seq_buf.append(seq.copy()); ins_buf.append(ins.copy())
            flat_buf.append(flat); act_buf.append(a)
            logp_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))

            next_seq, next_ins, reward, done, info = env.step(a)
            rew_buf.append(float(reward))
            ep_reward += reward
            ep_steps += 1
            if info.get('switched'):
                ep_switches += 1
            seq, ins = next_seq, next_ins

        reward_curve.append(ep_reward)
        switch_curve.append(ep_switches)

        # ---- GAE + returns ----
        # 终止状态 V=0（episode 自然结束）
        adv_buf, ret_buf = _gae(rew_buf, val_buf, last_value=0.0,
                                gamma=gamma, lam=lam)
        adv_buf = np.array(adv_buf, dtype=np.float32)
        ret_buf = np.array(ret_buf, dtype=np.float32)
        # normalize advantage
        if adv_buf.std() > 1e-6:
            adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        T = len(rew_buf)
        if T == 0:
            continue

        # 转 tensor（整批）
        seq_arr = np.stack(seq_buf)        # (T,k,3)
        ins_arr = np.stack(ins_buf)        # (T,5)
        flat_arr = np.stack(flat_buf)      # (T,D)
        act_arr = np.array(act_buf, dtype=np.int64)  # (T,)
        old_logp_arr = np.array(logp_buf, dtype=np.float32)  # (T,)

        seq_t_all = torch.from_numpy(seq_arr).to(device)
        ins_t_all = torch.from_numpy(ins_arr).to(device)
        flat_t_all = torch.from_numpy(flat_arr).to(device)
        act_t_all = torch.from_numpy(act_arr).to(device)
        old_logp_t_all = torch.from_numpy(old_logp_arr).to(device)
        adv_t_all = torch.from_numpy(adv_buf).to(device)
        ret_t_all = torch.from_numpy(ret_buf).to(device)

        # ---- PPO 多轮更新（mini-batch）----
        for epoch in range(n_epochs):
            # 随机 mini-batch 索引
            idx = np.random.permutation(T)
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                mb = idx[start:end]
                if len(mb) < 4:
                    continue  # 太小跳过
                s_mb = seq_t_all[mb]
                i_mb = ins_t_all[mb]
                f_mb = flat_t_all[mb]
                a_mb = act_t_all[mb]
                old_lp_mb = old_logp_t_all[mb]
                adv_mb = adv_t_all[mb]
                ret_mb = ret_t_all[mb]

                # actor loss（clipped surrogate）
                dist = actor.get_dist(s_mb, i_mb)
                logp_mb = dist.log_prob(a_mb)
                ratio = torch.exp(logp_mb - old_lp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                actor_loss = actor_loss - entropy_coef * entropy

                # critic loss
                value_mb = critic(f_mb)
                critic_loss = F.mse_loss(value_mb, ret_mb)

                opt_a.zero_grad()
                opt_c.zero_grad()
                # 联合 backward（actor 和 critic 独立优化器，但一起反传）
                total_loss = actor_loss + value_coef * critic_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                opt_a.step()
                opt_c.step()

        if (ep + 1) % log_every == 0 or ep == 0:
            elapsed = time.time() - t_start
            avg_r = float(np.mean(reward_curve[-log_every:]))
            avg_sw = float(np.mean(switch_curve[-log_every:]))
            print(f'    [PPO+GRU case={reward_case}] ep {ep+1:3d}/{n_episodes}  '
                  f'reward(近{log_every}均值)={avg_r:8.2f}  '
                  f'switches={avg_sw:5.1f}  '
                  f'buf_T={T}  {elapsed:.1f}s')

    print(f'  [PPO+GRU case={reward_case}] 训练完成: '
          f'初始{reward_curve[0]:.2f} → 末{np.mean(reward_curve[-20:]):.2f}  '
          f'(总耗时 {time.time()-t_start:.1f}s)')
    return reward_curve, actor, critic, env


if __name__ == '__main__':
    # 快速自检：用贪心 spatial stub（无 DQN），看 PPO+GRU 是否在学习
    print('[ppo] 快速自检 (30 episode, case234, 贪心 spatial stub)...')

    def greedy_spatial_stub(snap, u, valid_k, H, tau, L, psi, cur_sat, mv):
        a = int(np.argmax(psi))
        sat_ids = snap.visible_idx[u][valid_k]
        return a, int(sat_ids[a])

    curve, actor, critic, env = train(
        reward_case='case234', spatial_policy_fn=greedy_spatial_stub,
        n_episodes=30, seed=42, log_every=10)
    print(f'[ppo] reward 前5均值={np.mean(curve[:5]):.2f}, '
          f'后5均值={np.mean(curve[-5:]):.2f}')
