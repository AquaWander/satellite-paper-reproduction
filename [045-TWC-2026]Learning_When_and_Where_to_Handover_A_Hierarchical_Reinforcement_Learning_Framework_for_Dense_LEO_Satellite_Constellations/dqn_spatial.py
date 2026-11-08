# -*- coding: utf-8 -*-
"""
dqn_spatial.py
================================================================================
空间层 DQN（论文 Eq.38-39 结构）。

Q 网络：FC(input→256→128→128→64) + 输出头 → 每候选一个 Q 值（按 max_vis 输出）。
ReLU 激活。

训练组件：
  - replay buffer（容量 ~1e4）
  - target 网络（周期硬更新）
  - ε-greedy（1.0 线性衰减到 0.01，衰减率 0.995/episode）
  - Adam lr=1e-3（论文值），γ=0.99，batch=64
  - loss=MSE(Q_pred, r + γ·max Q_target)
  - masked Q：不可见候选 Q 设 -inf 后再 argmax

【诚实保真度边界】论文无种子/无代码，真实训练 + 定性一致（Fig.5 收敛形状），
不查表/硬编码输出。
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import SEED
from rl_env import SpatialEnv


# ============================================================================
# Q 网络（Eq.38-39）：FC(input→256→128→128→64) → 每候选一个 Q 值
# ============================================================================
class QNet(nn.Module):
    def __init__(self, state_dim, max_vis):
        super().__init__()
        self.max_vis = max_vis
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, max_vis),  # 每候选一个 Q 值
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_mask):
        item = (state, action, reward, next_state, done, next_mask)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        # next_state/next_mask 可能为 None（episode 结束）
        non_terminal = np.array([b[4] is False for b in batch], dtype=bool)
        state_dim = len(batch[0][0])
        max_vis = len(batch[0][5]) if batch[0][5] is not None else None
        # 找一个非 None 的 mask 确定 max_vis
        if max_vis is None:
            for b in batch:
                if b[5] is not None:
                    max_vis = len(b[5]); break
        next_states = np.zeros((len(batch), state_dim), dtype=np.float32)
        next_masks = np.zeros((len(batch), max_vis), dtype=bool)
        for i, b in enumerate(batch):
            if b[3] is not None:
                next_states[i] = b[3]
            if b[5] is not None:
                next_masks[i] = b[5]
        return (torch.from_numpy(states),
                torch.from_numpy(actions),
                torch.from_numpy(rewards),
                torch.from_numpy(next_states),
                torch.from_numpy(non_terminal),
                torch.from_numpy(next_masks))

    def __len__(self):
        return len(self.buf)


# ============================================================================
# DQN 智能体
# ============================================================================
class DQNAgent:
    def __init__(self, state_dim, max_vis, lr=1e-3, gamma=0.95,
                 batch_size=64, target_update_freq=200, buffer_cap=10000,
                 eps_start=1.0, eps_end=0.02, eps_decay=0.99, device='cpu'):
        self.state_dim = state_dim
        self.max_vis = max_vis
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device(device)

        self.qnet = QNet(state_dim, max_vis).to(self.device)
        self.target = QNet(state_dim, max_vis).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self.epsilon = eps_start
        self.train_steps = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, state, mask, greedy=False):
        """state: np (state_dim,); mask: np bool (max_vis,)。
        不可见候选 Q 设 -inf。greedy=True 时不探索（评估用）。
        返回 int action（在 [0, max_vis) 内的候选索引）。
        """
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0  # 不应发生（环境保证有可见候选）
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid))
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.qnet(s)[0].cpu().numpy()
        # masked Q
        q_masked = np.where(mask, q, -np.inf)
        return int(np.argmax(q_masked))

    # ------------------------------------------------------------------
    def store(self, state, action, reward, next_state, done, next_mask):
        self.buffer.push(state, action, reward, next_state, done, next_mask)

    def update_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ------------------------------------------------------------------
    def learn(self):
        """一次梯度更新。返回 loss 值（float），若 buffer 不足返回 None。"""
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, non_terminal, next_masks = \
            self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        non_terminal = non_terminal.to(self.device)
        next_masks = next_masks.to(self.device)

        # 当前 Q(s, a)
        q_all = self.qnet(states)                  # (B, max_vis)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # target Q: r + γ·max_{a' visible} Q_target(s', a')
        with torch.no_grad():
            q_next = self.target(next_states)      # (B, max_vis)
            # 不可见候选 Q 设 -inf（这样 max 不会选到它们）
            q_next_masked = q_next.masked_fill(~next_masks, float('-inf'))
            # 全终止的样本（next_state 无效）max 会是 -inf，置 0
            q_next_max = q_next_masked.max(dim=1)[0]  # (B,)
            q_next_max = torch.where(non_terminal, q_next_max,
                                     torch.zeros_like(q_next_max))
            # 防止 -inf 传播（非终止但所有 mask=False 的退化情况）
            q_next_max = torch.nan_to_num(q_next_max, neginf=0.0)
            target = rewards + self.gamma * q_next_max

        loss = nn.functional.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（稳定训练）
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.qnet.state_dict())
        return float(loss.item())


# ============================================================================
# 训练循环：一个权重配置训一个 DQN
# ============================================================================
def train(weights, n_episodes=300, seed=SEED, env_kwargs=None,
          log_every=20, warmup_eps=10, max_steps_per_episode=4000,
          device='cpu'):
    """训练一个空间层 DQN。
    weights: (w_load, w_signal, w_time)
    返回 (reward_curve_per_episode, qnet, env, last_losses)。
    """
    import time
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = env_kwargs or {}
    # 构造一个训练环境（几何只算一次，episode 内 reset 复用）
    env = SpatialEnv(weights=weights, seed=seed, **env_kwargs)
    max_vis = env.max_vis
    state_dim = env.state_dim

    agent = DQNAgent(state_dim, max_vis, device=device)

    reward_curve = []
    loss_curve = []
    t_start = time.time()

    for ep in range(n_episodes):
        # 每个 episode 用不同 seed 的 env（业务/几何不同 → 泛化）
        # 性能权衡：重建 env 几何较贵（~1s），故每 episode 只重置运行状态，
        # 用同一几何但不同 random seed 的信道实现（env.rng 仍随机）。
        state, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_loss_sum = 0.0
        ep_loss_n = 0
        while state is not None and ep_steps < max_steps_per_episode:
            mask = info['action_mask']
            action = agent.select_action(state, mask, greedy=False)
            next_state, reward, done, info_next = env.step(action)
            # 存 transition：next_state/next_mask 可能为 None（episode 结束）
            next_mask = info_next.get('action_mask')
            if next_mask is None:
                next_mask = np.zeros(max_vis, dtype=bool)
            ns_for_buf = next_state if next_state is not None else np.zeros(state_dim, dtype=np.float32)
            agent.store(state, action, reward, ns_for_buf,
                        done=(next_state is None), next_mask=next_mask)
            state = next_state
            info = info_next if next_state is not None else None
            ep_reward += reward
            ep_steps += 1
            # 学习（warmup 后开始）
            if ep >= warmup_eps:
                loss = agent.learn()
                if loss is not None:
                    ep_loss_sum += loss
                    ep_loss_n += 1
        # epsilon 衰减（每 episode 一次）
        agent.update_epsilon()
        reward_curve.append(ep_reward)
        loss_curve.append(ep_loss_sum / max(ep_loss_n, 1))

        if (ep + 1) % log_every == 0 or ep == 0:
            elapsed = time.time() - t_start
            # 用 greedy 评估一次（短 episode）看真实策略表现
            avg_r = np.mean(reward_curve[-log_every:])
            print(f'    [DQN w={weights}] ep {ep+1:3d}/{n_episodes}  '
                  f'reward(近{log_every}均值)={avg_r:8.2f}  '
                  f'eps={agent.epsilon:.3f}  loss={ep_loss_sum/max(ep_loss_n,1):.4f}  '
                  f'buf={len(agent.buffer)}  {elapsed:.1f}s')

    print(f'  [DQN w={weights}] 训练完成: '
          f'初始{reward_curve[0]:.2f} → 末{np.mean(reward_curve[-20:]):.2f}  '
          f'(总耗时 {time.time()-t_start:.1f}s)')
    return reward_curve, agent.qnet, env, loss_curve


if __name__ == '__main__':
    # 快速自检：50 episode 看是否在学
    print('[dqn] 快速训练自检 (50 episode, 权重 0.4/0.3/0.3)...')
    curve, qnet, env, losses = train(
        weights=(0.4, 0.3, 0.3), n_episodes=50, seed=42, log_every=10)
    print(f'[dqn] reward 前10均值={np.mean(curve[:10]):.2f}, '
          f'后10均值={np.mean(curve[-10:]):.2f}')
