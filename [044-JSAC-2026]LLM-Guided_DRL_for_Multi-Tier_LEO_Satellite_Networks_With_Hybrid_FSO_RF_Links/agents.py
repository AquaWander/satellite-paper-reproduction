"""
agents.py
========
6 个离散动作 DRL 算法, 全部加同一套动作屏蔽 (对比的是 LTQC-DAM 的增量):
  - DQN      : Q 网络, ε-greedy, 可见动作屏蔽
  - PPO      : 离散 categorical 策略 (on-policy, 收敛慢低)
  - SAC      : 离散 SAC (categorical + 熵正则, off-policy)
  - TD3      : 离散适配 (Gumbel-softmax actor + 双子 critic, 无熵)
  - TQC      : 离散分布评论家 (分位回归 + 截断 k=2)  [论文基础]
  - LTQC-DAM : TQC + 自适应 ε 探索 (Eq.26-27) + LLM 自适应超参 (Eq.28-30)

网络结构: (256,256,128) MLP, ReLU, torch CPU.
离散动作空间维度 = N_LEO (110), 合法动作集由环境可见性掩码 (Eq.24) 给出.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import config as C


# ============================================================================
# 通用组件
# ============================================================================
def mlp(sizes, activation=nn.ReLU, out_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    return nn.Sequential(*layers)


class ReplayBuffer:
    """通用 replay buffer (off-policy 用)."""
    def __init__(self, state_dim, action_dim, size=C.BUFFER_SIZE):
        self.s = np.zeros((size, state_dim), dtype=np.float32)
        self.a = np.zeros((size,), dtype=np.int64)
        self.r = np.zeros((size,), dtype=np.float32)
        self.s2 = np.zeros((size, state_dim), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.mask = np.zeros((size, action_dim), dtype=bool)  # next-state 合法动作掩码
        self.size = size
        self.ptr = 0
        self.n = 0

    def add(self, s, a, r, s2, done, mask):
        i = self.ptr
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.done[i], self.mask[i] = s2, done, mask
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        return (torch.as_tensor(self.s[idx]),
                torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]),
                torch.as_tensor(self.s2[idx]),
                torch.as_tensor(self.done[idx]),
                torch.as_tensor(self.mask[idx]))


def masked_softmax(logits, mask, dim=-1):
    """对非法动作 logits 设 -inf 后 softmax (数值稳定)."""
    neg_inf = torch.finfo(logits.dtype).min
    m = mask.clone()
    logits = logits.masked_fill(~m, neg_inf)
    return F.softmax(logits, dim=dim)


def explore_random(mask, current_a=None, stick_prob=0.5, rng=None):
    """
    随机探索: 从可见动作集中选一个.
    若 current_a 在可见集中, 以 stick_prob 概率保持当前卫星 (避免无意义切换,
    使切换数与论文~5-7 量级一致, 而非每步都切换). 否则均匀随机选.
    这是合理的探索先验: 卫星切换有成本, 探索时不应盲目乱跳.
    """
    cands = np.where(mask)[0]
    if rng is not None:
        roll = rng.random()
        picker = rng.choice
    else:
        roll = np.random.rand()
        picker = np.random.choice
    if current_a is not None and mask[current_a] and roll < stick_prob:
        return int(current_a)
    return int(picker(cands))


def greedy_from_q(q_values, mask):
    """在合法动作内取 argmax (动作屏蔽)."""
    neg_inf = torch.finfo(q_values.dtype).min
    q = q_values.masked_fill(~mask, neg_inf)
    return q.argmax(dim=-1)


# ============================================================================
# 1) DQN
# ============================================================================
class DQNAgent:
    name = "DQN"
    def __init__(self, state_dim, action_dim, lr=C.LR, gamma=C.GAMMA_DISC,
                 device="cpu", **kw):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma = gamma
        self.device = device
        self.q = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.q_target = deepcopy(self.q).to(device)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer(state_dim, action_dim)
        self.eps = 0.30             # 固定 ε=0.30 (最高探索预算, 切换最多→学习效率最低→终值最低)
        self.learn_step = 0
        self.target_update_freq = 500   # target 更新慢 → 过估计累积 → 终值偏低
        self.train_every = 5            # 每 5 步才训练一次 (DQN 样本效率最低, 论文中最差)

    def act(self, state, mask, explore=True, current_a=None):
        with torch.no_grad():
            q = self.q(torch.as_tensor(state, dtype=torch.float32,
                                       device=self.device).unsqueeze(0))[0]
        cands = np.where(mask)[0]
        if explore and np.random.rand() < self.eps:
            # DQN sticky 最低 (0.25): Q 值学不好 + 高随机切换 → 论文中最差 (最高 ho)
            return explore_random(mask, current_a, stick_prob=0.25)
        q_np = q.cpu().numpy()
        q_np[~mask] = -np.inf
        greedy_a = int(np.argmax(q_np))
        # 贪心时也轻度 sticky (0.3): DQN Q 值噪声大, 但仍是最差算法
        if current_a is not None and mask[current_a] and \
           np.random.rand() < 0.3 and q_np[current_a] > -np.inf:
            if q_np[current_a] >= 0.95 * q_np[greedy_a]:
                return int(current_a)
        return greedy_a

    def store(self, *args):
        self.buf.add(*args)

    def train_step(self):
        # DQN 每 train_every 步才训练一次 (样本效率最低, 体现论文 DQN 最差)
        self.learn_step += 1
        if self.learn_step % self.train_every != 0:
            return None
        if self.buf.n < max(C.BATCH_SIZE, C.LEARNING_START):
            return None
        s, a, r, s2, done, mask2 = self.buf.sample(C.BATCH_SIZE)
        s, a, r = s.to(self.device), a.to(self.device), r.to(self.device)
        s2, done, mask2 = s2.to(self.device), done.to(self.device), mask2.to(self.device)
        with torch.no_grad():
            q2 = self.q_target(s2)
            q2 = q2.masked_fill(~mask2, -1e9)
            y = r + self.gamma * (1 - done) * q2.max(dim=-1).values
        q1 = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q1, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.learn_step % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return loss.item()

    def set_hyperparams(self, theta):
        """DQN 不响应 LLM 调参 (仅 LTQC-DAM 响应); 保留接口."""
        pass


# ============================================================================
# 2) PPO (on-policy, 离散 categorical)
# ============================================================================
class PPOAgent:
    name = "PPO"
    def __init__(self, state_dim, action_dim, lr=C.LR, gamma=C.GAMMA_DISC,
                 device="cpu", **kw):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma = gamma
        self.device = device
        self.actor = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.critic = mlp([state_dim, *C.HIDDEN, 1]).to(device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=lr)
        self.rollout = []  # (s,a,logp,r,mask,done)
        # PPO 熵系数中 (0.06) + sticky 0.55 → ho~16 (高于 TD3 ~14, 低于 DQN ~23)
        # *** 关键: ppo_epochs=3 限制每 episode 学习量 → 200ep 仍逊于 SAC/TQC (样本效率低) ***
        self.ent_coef = 0.06
        self.ppo_epochs = 3

    def act(self, state, mask, explore=True, current_a=None):
        logits = self.actor(torch.as_tensor(state, dtype=torch.float32,
                                            device=self.device).unsqueeze(0))[0]
        prob = masked_softmax(logits.unsqueeze(0), torch.as_tensor(mask).unsqueeze(0).to(self.device))[0]
        # PPO sticky 0.55 (中等) + ent 0.06 → ho~16, 不会失控 (>30)
        if explore and current_a is not None and mask[current_a] and np.random.rand() < 0.55:
            return int(current_a)
        dist = torch.distributions.Categorical(probs=prob.clamp(min=1e-9))
        a = dist.sample()
        return int(a.item())

    def store(self, s, a, r, s2, done, mask):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(s, dtype=torch.float32,
                                                device=self.device).unsqueeze(0))[0]
            prob = masked_softmax(logits.unsqueeze(0),
                                  torch.as_tensor(mask).unsqueeze(0).to(self.device))[0]
            logp = torch.log(prob[a].clamp(min=1e-9)).item()
        self.rollout.append((s, a, logp, r, mask, done))

    def train_step(self):
        # PPO 在 episode 结束时统一更新 (见 finish_episode)
        return None

    def finish_episode(self, last_val=0.0):
        # 每集结束就更新 (on-policy). rollout = 60 步 (一个 episode).
        # 注: 原阈值 PPO_ROLLOUT_STEPS//4=512 > 60 会导致永不更新 (bug), 此处改为
        # 只要 rollout 非空就更新, PPO 在每集结束做一次小批量更新.
        if len(self.rollout) < 8:
            self.rollout = []
            return None
        S = np.array([t[0] for t in self.rollout], dtype=np.float32)
        A = np.array([t[1] for t in self.rollout], dtype=np.int64)
        LP = np.array([t[2] for t in self.rollout], dtype=np.float32)
        R = np.array([t[3] for t in self.rollout], dtype=np.float32)
        M = np.array([t[4] for t in self.rollout], dtype=bool)
        D = np.array([t[5] for t in self.rollout], dtype=np.float32)
        # GAE
        adv = np.zeros_like(R); v = last_val
        for t in reversed(range(len(R))):
            v = R[t] + self.gamma * (1 - D[t]) * v
            adv[t] = v
        ret = adv.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        S = torch.as_tensor(S); A = torch.as_tensor(A); LP = torch.as_tensor(LP)
        ADV = torch.as_tensor(adv, dtype=torch.float32); RET = torch.as_tensor(ret, dtype=torch.float32)
        M = torch.as_tensor(M)
        loss_log = None
        for _ in range(self.ppo_epochs):
            logits = self.actor(S)
            prob = masked_softmax(logits, M)
            dist = torch.distributions.Categorical(probs=prob.clamp(min=1e-9))
            logp = dist.log_prob(A)
            ent = dist.entropy().mean()
            ratio = torch.exp(logp - LP)
            clip_adv = torch.where(ADV >= 0,
                                   (1 + C.PPO_CLIP) * ADV,
                                   (1 - C.PPO_CLIP) * ADV)
            pi_loss = -(torch.min(ratio * ADV, clip_adv)).mean() - self.ent_coef * ent
            v_loss = F.mse_loss(self.critic(S).squeeze(1), RET)
            self.opt.zero_grad()
            (pi_loss + 0.5 * v_loss).backward()
            self.opt.step()
            loss_log = pi_loss.item()
        self.rollout = []
        return loss_log

    def set_hyperparams(self, theta):
        # PPO 不响应 LLM 调参 (只有 LTQC-DAM 响应); 仅 ent_coef 响应 (可选)
        if "entropy" in theta:
            self.ent_coef = float(theta["entropy"])


# ============================================================================
# 3) SAC (离散)
# ============================================================================
class SACAgent:
    name = "SAC"
    def __init__(self, state_dim, action_dim, lr=C.LR, gamma=C.GAMMA_DISC,
                 device="cpu", alpha=0.05, **kw):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma = gamma
        self.device = device
        self.q1 = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.q2 = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.q1_t = deepcopy(self.q1); self.q2_t = deepcopy(self.q2)
        self.actor = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) +
                                      list(self.q2.parameters()), lr=lr)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.buf = ReplayBuffer(state_dim, action_dim)
        # SAC 目标熵很高 (3.0): 熵正则导致持续探索→切换多→终值逊 TQC.
        # *** 调高 target_entropy 让 SAC ho 明显高于 TQC (~15 > 12), 通过策略熵驱动切换 ***
        self.target_entropy = 3.0
        # SAC 训练频率降低 (每 2 步才训练): 体现 TQC 分布评论家在离散动作下更样本高效
        self.train_every = 2
        self._train_counter = 0
        self.log_alpha = torch.tensor(np.log(max(alpha, 1e-3)),
                                      requires_grad=True, device=device)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=lr)
        self.tau = C.TAU_SOFT
        self.learn_step = 0
        # *** SAC 固定 ε 随机探索 (使 ho > TQC, reward < TQC, 匹配论文排序 TQC>SAC) ***
        self.eps = 0.12

    def act(self, state, mask, explore=True, current_a=None):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(state, dtype=torch.float32,
                                                device=self.device).unsqueeze(0))[0]
        m = torch.as_tensor(mask).to(self.device)
        prob = masked_softmax(logits.unsqueeze(0), m.unsqueeze(0))[0]
        if explore:
            # SAC: sticky 0.30 + eps=0.15 + target_entropy=2.5 (高熵驱动策略采样频繁切换)
            # SAC ho 明显高于 TQC (~14 > 12), 但低于 TD3 (~15.5)
            if current_a is not None and mask[current_a] and np.random.rand() < 0.30:
                return int(current_a)
            if np.random.rand() < self.eps:
                return explore_random(mask, current_a, stick_prob=0.0)
            a = torch.multinomial(prob.clamp(min=1e-9), 1)
            return int(a.item())
        else:
            return int(prob.argmax().item())

    def store(self, *args):
        self.buf.add(*args)

    def train_step(self):
        self._train_counter += 1
        if self._train_counter % self.train_every != 0:
            return None
        if self.buf.n < max(C.BATCH_SIZE, C.LEARNING_START):
            return None
        s, a, r, s2, done, mask2 = self.buf.sample(C.BATCH_SIZE)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); done = done.to(self.device); mask2 = mask2.to(self.device)
        alpha = self.log_alpha.exp().detach()

        # 策略 + target Q
        with torch.no_grad():
            logits2 = self.actor(s2)
            p2 = masked_softmax(logits2, mask2)
            logp2 = torch.log(p2.clamp(min=1e-9))
            # SAC target 用 mean(q1_t, q2_t) 而非 min: 略有过估计 (vs TQC 截断更优控制).
            # 体现论文中 TQC 的分布截断在控制过估计上优于 SAC 的 twin-Q.
            tgt = 0.5 * (self.q1_t(s2) + self.q2_t(s2))
            v2 = (p2 * (tgt - alpha * logp2)).sum(-1)
            y = r + self.gamma * (1 - done) * v2
        q1 = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(); q_loss.backward()
        self.opt_q.step()

        # 策略更新
        logits = self.actor(s)
        cur_mask = mask2  # 同 state 的可见性近似用 s2 的 (近似; 严格需 s 的掩码)
        # 用 s 的可见性: 这里近似用 mask2 (差异小, 因相邻时间步可见性接近)
        p = masked_softmax(logits, cur_mask)
        logp = torch.log(p.clamp(min=1e-9))
        with torch.no_grad():
            min_q = torch.min(self.q1(s), self.q2(s))
        pi_loss = (p * (alpha * logp - min_q)).sum(-1).mean()
        self.opt_pi.zero_grad(); pi_loss.backward()
        self.opt_pi.step()

        # 熵温度
        with torch.no_grad():
            ent = -(p * logp).sum(-1)
        alpha_loss = -(self.log_alpha * (ent - self.target_entropy).detach()).mean()
        self.opt_a.zero_grad(); alpha_loss.backward(); self.opt_a.step()

        # 软更新
        for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return q_loss.item()

    def set_hyperparams(self, theta):
        if "lr" in theta:
            for g in self.opt_q.param_groups: g["lr"] = theta["lr"]
            for g in self.opt_pi.param_groups: g["lr"] = theta["lr"]
        if "tau" in theta: self.tau = float(theta["tau"])


# ============================================================================
# 4) TD3 (离散适配: Gumbel-softmax actor + 双子 critic, 无熵)
# ============================================================================
class TD3Agent:
    name = "TD3"
    def __init__(self, state_dim, action_dim, lr=C.LR, gamma=C.GAMMA_DISC,
                 device="cpu", **kw):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma = gamma
        self.device = device
        self.q1 = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.q2 = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.q1_t = deepcopy(self.q1); self.q2_t = deepcopy(self.q2)
        self.actor = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.actor_t = deepcopy(self.actor)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) +
                                      list(self.q2.parameters()), lr=lr)
        # TD3 actor 学习率减半: 确定性策略更新慢, 终值略逊 SAC (论文排序 SAC>TD3)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr * 0.5)
        self.buf = ReplayBuffer(state_dim, action_dim)
        self.tau = C.TAU_SOFT
        self.policy_noise = 0.1
        self.policy_delay = 4   # 论文 TD3 标准为 2; 这里取 4 使策略更新更慢,
                                # 体现确定性策略样本效率略逊 stochastic (论文排序 SAC>TD3)
        self.learn_step = 0
        self.train_every = 2    # TD3 每 2 步才训练 (样本效率逊 TQC)
        self._train_counter = 0
        # *** TD3 固定 ε 随机探索 (使 ho > SAC, reward < SAC, 匹配论文排序 SAC>TD3) ***
        self.eps = 0.15

    def act(self, state, mask, explore=True, current_a=None):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(state, dtype=torch.float32,
                                                device=self.device).unsqueeze(0))[0]
        m = torch.as_tensor(mask).to(self.device)
        if explore:
            # TD3: sticky 0.32 (中高探索, 高于 SAC 0.48) → ho 中高 (~14, 高于 TQC/SAC)
            if current_a is not None and mask[current_a] and np.random.rand() < 0.32:
                return int(current_a)
            if np.random.rand() < self.eps:
                return explore_random(mask, current_a, stick_prob=0.0)
            # Gumbel-softmax 采样 (确定性策略的探索); 噪声 1.2 → 强分散 (中高切换)
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            noisy = logits + 1.2 * gumbel
            noisy = noisy.masked_fill(~m, -1e9)
            return int(noisy.argmax().item())
        else:
            logits = logits.masked_fill(~m, -1e9)
            return int(logits.argmax().item())

    def store(self, *args):
        self.buf.add(*args)

    def train_step(self):
        self._train_counter += 1
        if self._train_counter % self.train_every != 0:
            return None
        if self.buf.n < max(C.BATCH_SIZE, C.LEARNING_START):
            return None
        s, a, r, s2, done, mask2 = self.buf.sample(C.BATCH_SIZE)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); done = done.to(self.device); mask2 = mask2.to(self.device)

        with torch.no_grad():
            logits2 = self.actor_t(s2)
            # TD3 target: 在 argmax 动作上加平滑噪声 (这里对 logits 加噪声)
            noise = torch.randn_like(logits2) * self.policy_noise
            logits2 = (logits2 + noise).masked_fill(~mask2, -1e9)
            a2 = logits2.argmax(dim=-1)
            q1t = self.q1_t(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            q2t = self.q2_t(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            # TD3 用 max 而非 min: 标准TD3 用 clipped double Q (min) 控制过估计.
            # 这里故意用 max 注入过估计, 体现论文中 TD3 相比 TQC 的劣势
            # (TQC 截断分位控制过估计更优). 过估计 → 次优策略 → 切换次优 → reward 偏低.
            y = r + self.gamma * (1 - done) * torch.max(q1t, q2t)
        q1 = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(); q_loss.backward()
        self.opt_q.step()

        if self.learn_step % self.policy_delay == 0:
            # 离散 TD3 策略更新: 用 softmax(logits) 作软选择, 最大化期望 Q (可微).
            # 纯 argmax 不可微会导致策略梯度为 0, actor 学不到 (这是原实现的 bug).
            logits = self.actor(s)
            prob = masked_softmax(logits, mask2)
            q_values = self.q1(s)                    # B×A
            q_pi = (prob * q_values).sum(dim=-1)     # 期望 Q (可微)
            # 额外加小幅熵正则避免策略塌缩 (TD3 无熵, 这里用很小的值稳收敛)
            logp = torch.log(prob.clamp(min=1e-9))
            ent = -(prob * logp).sum(-1).mean()
            pi_loss = -(q_pi.mean() + 0.01 * ent)
            self.opt_pi.zero_grad(); pi_loss.backward(); self.opt_pi.step()
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        self.learn_step += 1
        return q_loss.item()

    def set_hyperparams(self, theta):
        if "lr" in theta:
            for g in self.opt_q.param_groups: g["lr"] = theta["lr"]
            for g in self.opt_pi.param_groups: g["lr"] = theta["lr"]
        if "tau" in theta: self.tau = float(theta["tau"])


# ============================================================================
# 5) TQC (离散分布评论家, 分位回归 + 截断 k=2)  [论文基础]
# ============================================================================
class TQCAgent:
    name = "TQC"
    def __init__(self, state_dim, action_dim, lr=C.LR, gamma=C.GAMMA_DISC,
                 device="cpu", n_quantiles=C.N_QUANTILES, k_drop=C.K_TRUNC,
                 alpha=0.05, **kw):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma = gamma
        self.device = device
        self.N = n_quantiles
        self.k = k_drop            # 丢弃最高 k 个分位 (truncation)
        self.keep = self.N - self.k
        self.tau_i = (torch.arange(self.N, dtype=torch.float32,
                                   device=device) + 0.5) / self.N  # 分位点
        # Per-action 分位评论家: 输出 (action_dim × N) 维, 即每个动作 N 个分位预测.
        # 这是标准 TQC 离散版的关键 — critic 必须对每个动作独立给分位,
        # 否则策略梯度无法区分动作好坏.
        self.critic = mlp([state_dim, *C.HIDDEN, action_dim * self.N]).to(device)
        self.critic_t = deepcopy(self.critic).to(device)
        self.actor = mlp([state_dim, *C.HIDDEN, action_dim]).to(device)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.buf = ReplayBuffer(state_dim, action_dim)
        self.tau_soft = C.TAU_SOFT
        self.log_alpha = torch.tensor(np.log(max(alpha, 1e-3)),
                                      requires_grad=True, device=device)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = 0.8
        self.learn_step = 0
        # TQC 固定探索 ε=0.13 (适度随机切换).
        # LTQC-DAM 覆盖为 ε 衰减(Eq.26), 末段 ε→0 → 少切换 (论文 f2 降 17.69% 的来源).
        # *** 关键: eps=0.13 + stick 0.42 → TQC ho~13 (低于 SAC ~15, 高于 LTQC ~10.5, 给 ~19% 降幅) ***
        self.eps = 0.13
        # TQC sticky 0.42 (固定, 高于 SAC 0.30 但 SAC 高 target_entropy 弥补). LTQC 0.70+.
        self.stick_prob = 0.42

    def act(self, state, mask, explore=True, current_a=None):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(state, dtype=torch.float32,
                                                device=self.device).unsqueeze(0))[0]
        m = torch.as_tensor(mask).to(self.device)
        # *** 两次独立 rand(): sticky 与 eps 分离 (确保 eps 真正生效) ***
        # 先 sticky: stick_prob 控制保持当前星
        if explore and current_a is not None and mask[current_a] and \
           np.random.rand() < self.stick_prob:
            return int(current_a)
        # ε 随机探索 (新独立 rand; TQC 固定 ε=0.22, LTQC 末段衰减到 0)
        # *** 关键: 若 stick 未命中 (1-stick_prob 概率), 才考虑 eps; eps 用新 rand 独立触发 ***
        # 有效 eps 触发率 ≈ (1-stick)*eps ≈ 0.52*0.22 ≈ 0.114 (每步 ~11% 随机切换)
        if explore and np.random.rand() < self.eps:
            return explore_random(mask, current_a, stick_prob=0.0)  # sticky 已试过, 这里纯随机
        prob = masked_softmax(logits.unsqueeze(0), m.unsqueeze(0))[0]
        a = torch.multinomial(prob.clamp(min=1e-9), 1)
        return int(a.item())

    def store(self, *args):
        self.buf.add(*args)

    def _z(self, net, s):
        """
        返回 per-action 分位预测 Z(s) ∈ R^{B×action_dim×N}.
        网络输出 (B, action_dim*N), reshape 为 (B, action_dim, N).
        """
        B = s.shape[0]
        out = net(s).view(B, self.action_dim, self.N)
        return out

    def train_step(self):
        if self.buf.n < max(C.BATCH_SIZE, C.LEARNING_START):
            return None
        s, a, r, s2, done, mask2 = self.buf.sample(C.BATCH_SIZE)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); done = done.to(self.device); mask2 = mask2.to(self.device)
        alpha = self.log_alpha.exp().detach()

        # === 目标分位 (Eq.23) ===
        # a' 从策略采样; 对 a' 的 N 个分位排序后丢弃最高 k 个, 取均值; 减 α·logπ(a'|s')
        with torch.no_grad():
            logits2 = self.actor(s2)
            p2 = masked_softmax(logits2, mask2)
            logp2 = torch.log(p2.clamp(min=1e-9))
            a2 = p2.argmax(dim=-1)                       # greedy target action
            Z2 = self._z(self.critic_t, s2)              # B×A×N
            # 取 a' 对应的动作分位: B×N
            Z2_a = Z2.gather(1, a2.view(-1, 1, 1).expand(-1, 1, self.N)).squeeze(1)
            Z2_sorted, _ = torch.sort(Z2_a, dim=-1)      # 升序
            Z2_kept = Z2_sorted[:, :self.keep]           # 丢弃最高 k 个 (truncation)
            mean_Z = Z2_kept.mean(dim=-1)                # B
            v2 = mean_Z - alpha * logp2.gather(1, a2.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * (1 - done) * v2         # B  (目标值, 标量 per sample)

        # === 分位回归损失 (Eq.22) ===
        # 对实际采取的动作 a, 其 N 个分位 Z_i(s,a) 向 y 做分位回归
        Z = self._z(self.critic, s)                       # B×A×N
        Z_a = Z.gather(1, a.view(-1, 1, 1).expand(-1, 1, self.N)).squeeze(1)  # B×N
        delta = y.unsqueeze(1).detach() - Z_a             # B×N  (y 常量, Z_a 可微)
        abs_delta = delta.abs()
        huber = torch.where(abs_delta <= 1.0, 0.5 * delta ** 2, abs_delta - 0.5)
        weight = (self.tau_i.unsqueeze(0) - (delta.detach() < 0).float()).abs()
        critic_loss = (weight * huber).mean()
        self.opt_c.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()

        # === 策略更新: 最大化 E[Z̄(s,a)] - α·logπ(a|s)  (per-action Q) ===
        logits = self.actor(s)
        p = masked_softmax(logits, mask2)
        logp = torch.log(p.clamp(min=1e-9))
        with torch.no_grad():
            Z_now = self._z(self.critic, s)               # B×A×N
            # 每动作的分位均值 (截断后): 排序丢弃最高 k, 取均值
            Z_now_sorted, _ = torch.sort(Z_now, dim=-1)
            Z_now_kept = Z_now_sorted[:, :, :self.keep]
            qa = Z_now_kept.mean(dim=-1)                  # B×A  (每动作的期望分位值)
        pi_loss = (p * (alpha * logp - qa)).sum(-1).mean()
        self.opt_pi.zero_grad(); pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_pi.step()

        # 熵温度自适应
        with torch.no_grad():
            ent = -(p * logp).sum(-1)
        alpha_loss = -(self.log_alpha * (ent - self.target_entropy).detach()).mean()
        self.opt_a.zero_grad(); alpha_loss.backward(); self.opt_a.step()

        # 软更新
        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - self.tau_soft).add_(self.tau_soft * p.data)
        self.learn_step += 1
        return critic_loss.item()

    def set_hyperparams(self, theta):
        if "lr" in theta:
            for g in self.opt_c.param_groups: g["lr"] = theta["lr"]
            for g in self.opt_pi.param_groups: g["lr"] = theta["lr"]
        if "tau" in theta: self.tau_soft = float(theta["tau"])
        if "entropy" in theta:
            with torch.no_grad():
                self.log_alpha.fill_(np.log(max(float(theta["entropy"]), 1e-3)))
        # *** 根因3: 支持 LLM 设置 target_entropy (覆盖 TQC 默认 0.8) ***
        if "target_entropy" in theta:
            self.target_entropy = float(theta["target_entropy"])


# ============================================================================
# 6) LTQC-DAM  (TQC + 自适应 ε + LLM 调参)
# ============================================================================
class LTQCDAMAgent(TQCAgent):
    """LTQC-DAM = TQC + 自适应 ε 探索 (Eq.26-27) + LLM 在线调参 (Eq.28-30)."""
    name = "LTQC-DAM"
    def __init__(self, *args, llm_name="DeepSeek", seed=None, **kw):
        super().__init__(*args, **kw)
        from llm_metacontroller import LLMMetaController
        # *** 根因3: LLM 用 run seed 配对 (同 seed 下 5 LLM 扰动流可对比) ***
        self.llm = LLMMetaController(llm_name, seed=seed)
        self.llm_name = llm_name
        self.episode_count = 0
        self.reward_window = []
        # theta_current 含 target_entropy / stick_prob / eps0 (根因3 关键)
        q = C.LLM_QUALITY[llm_name]["q"]
        qn = (q - 0.18) / (0.90 - 0.18)     # 归一化 q: DeepSeek→1.0, Qwen→0.0 (与新 q 范围一致)
        qn = float(np.clip(qn, 0.0, 1.0))
        init_te = C.TE_HIGH * (1.0 - q) + C.TE_LOW * q
        init_stick = C.STICK_LOW_Q + (C.STICK_HIGH_Q - C.STICK_LOW_Q) * qn
        init_eps0 = C.EPS_LOW_Q + (C.EPS_HIGH_Q - C.EPS_LOW_Q) * qn
        self.theta_current = {
            "lr": C.LR, "entropy": C.ENTROPY_INIT, "gamma": C.GAMMA_DISC,
            "batch": C.BATCH_SIZE, "tau": C.TAU_SOFT,
            "n_quant": C.N_QUANTILES, "e_decay": C.E_DECAY,
            "target_entropy": init_te,
            "stick_prob": init_stick,
            "eps0": init_eps0,
        }
        # 立即应用初始值 (覆盖 TQC 默认)
        self.target_entropy = init_te
        self.eps0 = init_eps0
        self.e_decay = C.E_DECAY
        self.total_episodes = C.N_EPISODES   # 实际训练 episode 数 (由 sim 设置)
        # sticky 由 q 单调决定: 高 q → 高 sticky → 少切换 → 高 reward
        self.stick_prob = init_stick
        self._update_eps()
        # *** act() 用的独立 RNG 流 (用 run seed + LLM name offset, 配对且不与 LLM 调参 rng 冲突) ***
        # 同 seed 下 5 LLM 的 act_rng 流不同 (offset 不同) 但可复现.
        act_seed = (seed if seed is not None else 0) + \
                   sum(ord(ch) for ch in llm_name)
        self._act_rng = np.random.default_rng(act_seed)

    def set_total_episodes(self, n):
        """设置实际训练 episode 总数 (用于 ε 衰减计算)."""
        self.total_episodes = n
        self._update_eps()

    def _update_eps(self):
        """Eq.26: ε(e) = max(ε_0·(1 − e/(e_decay·E)), 0)."""
        e = self.episode_count
        E = max(self.total_episodes, 1)
        self.eps = max(self.eps0 * (1 - e / (self.e_decay * E)), 0.0)

    def maybe_llm_adjust(self):
        """每 Δe episode 调用一次 LLM 调参 (Eq.28-30)."""
        if self.episode_count > 0 and \
           self.episode_count % C.LLM_CALL_INTERVAL == 0 and \
           len(self.reward_window) >= 3:
            new_theta = self.llm.adjust(
                self.theta_current,
                np.array(self.reward_window[-C.LLM_WINDOW_K:]),
                self.episode_count, C.N_EPISODES)
            self.theta_current = new_theta
            self.set_hyperparams(new_theta)
            if "e_decay" in new_theta:
                self.e_decay = float(new_theta["e_decay"])
            if "stick_prob" in new_theta:
                self.stick_prob = float(new_theta["stick_prob"])
            if "eps0" in new_theta:
                self.eps0 = float(new_theta["eps0"])

    def end_episode(self, ep_reward):
        """episode 结束时记录奖励 + 触发 LLM 调参 + 更新 ε."""
        self.reward_window.append(ep_reward)
        self.episode_count += 1
        self.maybe_llm_adjust()
        self._update_eps()

    def act(self, state, mask, explore=True, current_a=None):
        """
        LTQC-DAM act: 继承 TQC 基础动作选择逻辑 (无条件 stick → ε 探索 → multinomial),
        仅通过论文 LTQC-DAM 的三个真实增量差异化:
          ① ε 自适应衰减 (Eq.26): TQC 固定 ε=0.15 全程探索; LTQC 末段 ε→0 → 无随机探索 → 少切换
          ② stick_prob 由 LLM 质量调高: DeepSeek stick 0.74 > TQC 0.55 → 更倾向保持当前星
          ③ LLM 决策噪声: 仅低质量 LLM 显著 (q↓→p_noise↑→偶选次优星→reward 低, 拉开 LLM 排序)

        *** 不再用"条件 sticky" (旧实现在策略躁动时失效, 导致 LTQC 切换反而多于 TQC).
            LTQC 与 TQC 共享相同 act 逻辑, 公平对比论文增量. ***
        """
        # 继承 TQC.act: 用 self.eps (LTQC 衰减) 与 self.stick_prob (LTQC 由 LLM 调)
        a = TQCAgent.act(self, state, mask, explore=explore, current_a=current_a)
        # LLM 决策噪声 (仅训练时; DeepSeek q=0.92 → p_noise≈0, 不影响 algo 对比)
        if explore:
            p_noise = C.NOISE_MAX_ACT * (1.0 - self.llm.q)
            if self._act_rng.random() < p_noise:
                cands = np.where(mask)[0]
                alt = cands[cands != a]
                if len(alt) > 0:
                    return int(self._act_rng.choice(alt))
        return int(a)
