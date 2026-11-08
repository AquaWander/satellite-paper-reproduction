"""
qmix_agent.py — QMIX 算法实现
实现: 前馈智能体网络(含手动反向传播), Mixing Network + Hypernetwork, 经验回放, 训练
对应论文 Eq.28-46 (QMIX CTDE 框架)
"""
import numpy as np
import config as cfg


class FeedforwardNet:
    """两层前馈网络 (手动反向传播)
    结构: input -> ReLU -> output
    """
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        rng = np.random.RandomState(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # He 初始化
        self.W1 = rng.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(out_dim)
        # 缓存 (用于反向传播)
        self.x = None
        self.h = None

    def forward(self, x):
        """前向传播: x -> Q-values"""
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        q = self.h @ self.W2 + self.b2
        return q

    def backward(self, dq):
        """反向传播: 计算梯度
        :param dq: 损失对输出的梯度 (out_dim,)
        :return: (dW1, db1, dW2, db2)
        """
        dW2 = np.outer(self.h, dq)
        db2 = dq
        dh = dq @ self.W2.T * (self.h > 0)  # ReLU 导数
        dW1 = np.outer(self.x, dh)
        db1 = dh
        return dW1, db1, dW2, db2

    def get_params(self):
        return {'W1': self.W1.copy(), 'b1': self.b1.copy(),
                'W2': self.W2.copy(), 'b2': self.b2.copy()}

    def set_params(self, p):
        self.W1 = p['W1'].copy()
        self.b1 = p['b1'].copy()
        self.W2 = p['W2'].copy()
        self.b2 = p['b2'].copy()


class MixingNetwork:
    """QMIX Mixing Network + Hypernetwork (Eq.35-39)
    输入: 各智能体 Q 值 + 全局状态; 输出: Q_tot
    Hypernetwork 根据状态生成归一化权重 (保证单调性)
    """
    def __init__(self, n_agents, state_dim, hidden_dim=cfg.QMIX_HIDDEN_DIM, seed=0):
        rng = np.random.RandomState(seed)
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # Hypernetwork: state -> W1 (n_agents x hidden_dim)
        self.hyper_w1 = rng.randn(state_dim, n_agents * hidden_dim) * 0.01
        self.hyper_b1 = np.zeros(n_agents * hidden_dim)
        # Hypernetwork: state -> W2 (hidden_dim)
        self.hyper_w2 = rng.randn(state_dim, hidden_dim) * 0.01
        self.hyper_b2 = np.zeros(hidden_dim)
        # 偏置
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(1)
        # 缓存
        self.q_vals = None
        self.state = None
        self.w1 = None
        self.w2 = None
        self.h = None

    def forward(self, q_vals, state):
        """前向传播: Q_tot = Mixing(q_vals, state)"""
        self.q_vals = q_vals
        self.state = state
        # 生成第一层权重 (绝对值保证非负 -> 单调性)
        self.w1 = np.abs((state @ self.hyper_w1 + self.hyper_b1)
                         .reshape(self.n_agents, self.hidden_dim))
        h = np.maximum(0, q_vals @ self.w1 + self.b1)  # ReLU
        self.h = h
        # 生成第二层权重
        self.w2 = np.abs((state @ self.hyper_w2 + self.hyper_b2)
                         .reshape(self.hidden_dim))
        q_tot = h @ self.w2 + self.b2[0]
        return q_tot

    def backward(self, dq_tot):
        """反向传播
        :param dq_tot: 损失对 Q_tot 的梯度 (标量)
        :return: 各参数梯度 dict
        """
        # 对第二层
        dh = dq_tot * self.w2  # (hidden_dim,)
        dW2_hyper = np.outer(self.state, dq_tot * self.h)  # (state_dim, hidden_dim)
        dW2 = dq_tot * self.h
        db2 = dq_tot

        # 对第一层
        dq_qvals = dh @ self.w1.T  # (n_agents,)
        # dw1: dQ_tot/dw1, 形状 (n_agents, hidden_dim)
        # Q_tot = h @ w2, h = relu(q_vals @ w1 + b1)
        # dQ_tot/dw1 = outer(q_vals, relu'(·) * w2) 转置对齐
        dw1 = np.outer(self.q_vals, dq_tot * self.w2 * (self.h > 0))  # (n_agents, hidden_dim)
        # w1 = abs(pre_w1), 梯度经 abs 求导
        pre_w1 = (self.state @ self.hyper_w1 + self.hyper_b1).reshape(self.n_agents, self.hidden_dim)
        dw1 = dw1 * np.sign(pre_w1)
        # hyper_w1 梯度: state(1,state_dim) -> pre_w1(1, n_agents*hidden_dim)
        dhyper_w1 = np.outer(self.state, dw1.flatten())
        dhyper_b1 = dw1.flatten()
        db1 = dh * (self.h > 0)

        # 对 hyper_w2 的梯度
        # w2 = abs(state @ hyper_w2 + hyper_b2)
        pre_w2 = self.state @ self.hyper_w2 + self.hyper_b2
        dhyper_w2 = np.outer(self.state, dq_tot * self.h * np.sign(pre_w2))
        dhyper_b2 = dq_tot * self.h * np.sign(pre_w2)

        return {
            'hyper_w1': dhyper_w1,
            'hyper_b1': dhyper_b1,
            'hyper_w2': dhyper_w2,
            'hyper_b2': dhyper_b2,
            'b1': db1,
            'b2': db2,
            # 返回对 q_vals 的梯度供智能体网络使用
            'dq_qvals': dq_qvals,
        }

    def get_params(self):
        return {'hyper_w1': self.hyper_w1.copy(), 'hyper_b1': self.hyper_b1.copy(),
                'hyper_w2': self.hyper_w2.copy(), 'hyper_b2': self.hyper_b2.copy(),
                'b1': self.b1.copy(), 'b2': self.b2.copy()}

    def set_params(self, p):
        self.hyper_w1 = p['hyper_w1'].copy()
        self.hyper_b1 = p['hyper_b1'].copy()
        self.hyper_w2 = p['hyper_w2'].copy()
        self.hyper_b2 = p['hyper_b2'].copy()
        self.b1 = p['b1'].copy()
        self.b2 = p['b2'].copy()


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity=cfg.QMIX_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, actions, rewards, next_obs, dones, states, next_states):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, actions, rewards, next_obs, dones, states, next_states)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QMIXAgent:
    """QMIX 主智能体 (CTDE 框架)
    训练时集中学习 mixing network, 执行时各智能体独立决策
    """
    def __init__(self, n_agents=cfg.N_SATELLITES, obs_dim=cfg.N_CELLS,
                 n_actions=cfg.N_CELLS, state_dim=cfg.N_SATELLITES * cfg.N_CELLS,
                 hidden_dim=cfg.QMIX_GRU_HIDDEN_DIM):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.state_scale = 500.0  # 归一化系数: 最大流量约 500 Mbps
        self.obs_scale = 500.0

        # 智能体网络 (每个卫星一个)
        self.agents = [FeedforwardNet(obs_dim, hidden_dim, n_actions, seed=i)
                       for i in range(n_agents)]
        self.target_agents = [FeedforwardNet(obs_dim, hidden_dim, n_actions, seed=i + 100)
                              for i in range(n_agents)]
        for i in range(n_agents):
            self.target_agents[i].set_params(self.agents[i].get_params())

        # Mixing Network
        self.mixing = MixingNetwork(n_agents, state_dim, hidden_dim, seed=200)
        self.target_mixing = MixingNetwork(n_agents, state_dim, hidden_dim, seed=300)
        self.target_mixing.set_params(self.mixing.get_params())

        # 经验回放
        self.replay_buffer = ReplayBuffer()

        self.lr = cfg.QMIX_LR
        self.epsilon = cfg.QMIX_EPSILON_START
        self.epsilon_decay = (cfg.QMIX_EPSILON_START - cfg.QMIX_EPSILON_END) / cfg.QMIX_EPSILON_DECAY
        self.training_step = 0

    def _norm_obs(self, obs):
        """归一化观测值到 [0, 1]"""
        return np.asarray(obs, dtype=np.float64) / self.obs_scale

    def _norm_state(self, state):
        """归一化全局状态到 [0, 1]"""
        return np.asarray(state, dtype=np.float64) / self.state_scale

    # ========== 动作选择 ==========

    def select_actions(self, obs_list, epsilon=None):
        """ε-greedy 选择动作 (每个智能体选择 N_BEAMS 个小区)"""
        if epsilon is None:
            epsilon = self.epsilon
        actions = []
        for i in range(self.n_agents):
            obs_norm = self._norm_obs(obs_list[i])
            q = self.agents[i].forward(obs_norm)
            action = np.zeros(self.n_actions)
            if np.random.rand() < epsilon:
                chosen = np.random.choice(self.n_actions, size=cfg.N_BEAMS, replace=False)
            else:
                chosen = np.argsort(-q)[:cfg.N_BEAMS]
            action[chosen] = 1
            actions.append(action)
        return np.array(actions)

    def select_actions_exploit(self, obs_list):
        """贪婪选择 (无探索)"""
        return self.select_actions(obs_list, epsilon=0.0)

    # ========== 经验存储 ==========

    def store_experience(self, obs, actions, rewards, next_obs, dones, state, next_state):
        self.replay_buffer.push(obs, actions, rewards, next_obs, dones, state, next_state)

    def update_epsilon(self):
        self.epsilon = max(cfg.QMIX_EPSILON_END, self.epsilon - self.epsilon_decay)

    # ========== Q_tot 计算 ==========

    def _agent_qs(self, agents, obs_list):
        """计算所有智能体的 Q 值 (输入归一化)"""
        qs = []
        for i in range(self.n_agents):
            obs_norm = self._norm_obs(obs_list[i])
            qs.append(agents[i].forward(obs_norm))
        return qs

    def _q_selected(self, qs, actions):
        """选择动作对应的 Q 值 (对选中动作求平均)"""
        q_selected = []
        for i in range(self.n_agents):
            n_act = max(np.sum(actions[i]), 1)
            q_selected.append(np.sum(qs[i] * actions[i]) / n_act)
        return np.array(q_selected)

    # ========== 训练 ==========

    def train_step(self):
        """训练一步: 采样经验, TD学习, 反向传播更新"""
        if len(self.replay_buffer) < cfg.QMIX_BATCH_SIZE:
            return None

        batch = self.replay_buffer.sample(cfg.QMIX_BATCH_SIZE)
        obs_batch, action_batch, reward_batch, next_obs_batch, \
            done_batch, state_batch, next_state_batch = batch

        total_loss = 0.0
        # 累积梯度
        agent_grads = [{'W1': np.zeros_like(a.W1), 'b1': np.zeros_like(a.b1),
                        'W2': np.zeros_like(a.W2), 'b2': np.zeros_like(a.b2)}
                       for a in self.agents]
        mix_grads = {'hyper_w1': np.zeros_like(self.mixing.hyper_w1),
                     'hyper_b1': np.zeros_like(self.mixing.hyper_b1),
                     'hyper_w2': np.zeros_like(self.mixing.hyper_w2),
                     'hyper_b2': np.zeros_like(self.mixing.hyper_b2),
                     'b1': np.zeros_like(self.mixing.b1),
                     'b2': np.zeros_like(self.mixing.b2)}

        for b in range(len(obs_batch)):
            obs = obs_batch[b]
            actions = action_batch[b]
            reward = reward_batch[b]
            next_obs = next_obs_batch[b]
            done = done_batch[b]
            state = state_batch[b]
            next_state = next_state_batch[b]

            # 当前 Q_tot (online network)
            qs = self._agent_qs(self.agents, obs)
            q_selected = self._q_selected(qs, actions)
            state_norm = self._norm_state(state)
            q_tot = self.mixing.forward(q_selected, state_norm)

            # 目标 Q_tot (target network)
            target_qs = self._agent_qs(self.target_agents, next_obs)
            # max a' Q(s', a'): 每个智能体取最大 Q
            q_max = np.array([np.max(target_qs[i]) for i in range(self.n_agents)])
            next_state_norm = self._norm_state(next_state)
            q_tot_target = self.target_mixing.forward(q_max, next_state_norm)

            # TD 目标
            target = reward + cfg.QMIX_GAMMA * q_tot_target * (1 - done)
            td = q_tot - target
            loss = 0.5 * td ** 2
            total_loss += loss

            # 反向传播: 损失对 Q_tot 的梯度
            dq_tot = td  # dL/dQ_tot = (Q_tot - target)

            # Mixing network 反向传播
            mix_grad_b = self.mixing.backward(dq_tot)
            for k in mix_grads:
                mix_grads[k] += mix_grad_b[k]

            # 对每个智能体: 通过动作选择传播梯度
            dq_agent = mix_grad_b['dq_qvals']  # (n_agents,)
            for i in range(self.n_agents):
                n_act = max(np.sum(actions[i]), 1)
                # dQ_select/dQ_i = actions[i] / n_act
                dq_i = dq_agent[i] * actions[i] / n_act
                dW1, db1, dW2, db2 = self.agents[i].backward(dq_i)
                agent_grads[i]['W1'] += dW1
                agent_grads[i]['b1'] += db1
                agent_grads[i]['W2'] += dW2
                agent_grads[i]['b2'] += db2

        # 平均梯度并更新 (带梯度裁剪)
        n = len(obs_batch)
        lr = self.lr
        grad_norm_max = 10.0
        for i in range(self.n_agents):
            gn = np.sqrt(np.sum(agent_grads[i]['W1']**2) + np.sum(agent_grads[i]['W2']**2))
            if gn > grad_norm_max:
                scale = grad_norm_max / (gn + 1e-8)
                agent_grads[i]['W1'] *= scale
                agent_grads[i]['W2'] *= scale
            self.agents[i].W1 -= lr * agent_grads[i]['W1'] / n
            self.agents[i].b1 -= lr * agent_grads[i]['b1'] / n
            self.agents[i].W2 -= lr * agent_grads[i]['W2'] / n
            self.agents[i].b2 -= lr * agent_grads[i]['b2'] / n
        mg = np.sqrt(np.sum(mix_grads['hyper_w1']**2) + np.sum(mix_grads['hyper_w2']**2))
        if mg > grad_norm_max:
            mscale = grad_norm_max / (mg + 1e-8)
            for k in mix_grads:
                mix_grads[k] *= mscale
        self.mixing.hyper_w1 -= lr * mix_grads['hyper_w1'] / n
        self.mixing.hyper_b1 -= lr * mix_grads['hyper_b1'] / n
        self.mixing.hyper_w2 -= lr * mix_grads['hyper_w2'] / n
        self.mixing.hyper_b2 -= lr * mix_grads['hyper_b2'] / n
        self.mixing.b1 -= lr * mix_grads['b1'] / n
        self.mixing.b2 -= lr * mix_grads['b2'] / n

        # 软更新目标网络
        self._soft_update_target()

        self.training_step += 1
        return total_loss / n

    def _soft_update_target(self):
        """软更新 (Eq.46): θ' = τ·θ + (1-τ)·θ'"""
        tau = cfg.QMIX_TAU
        for i in range(self.n_agents):
            ps = self.agents[i].get_params()
            pt = self.target_agents[i].get_params()
            new_p = {k: tau * ps[k] + (1 - tau) * pt[k] for k in ps}
            self.target_agents[i].set_params(new_p)
        ps = self.mixing.get_params()
        pt = self.target_mixing.get_params()
        new_p = {k: tau * ps[k] + (1 - tau) * pt[k] for k in ps}
        self.target_mixing.set_params(new_p)


class SADRLAgent:
    """单智能体 DRL (SADRL-BH) - 独立 DQN"""
    def __init__(self, obs_dim, n_actions, hidden_dim=cfg.QMIX_GRU_HIDDEN_DIM):
        self.net = FeedforwardNet(obs_dim, hidden_dim, n_actions, seed=0)
        self.n_actions = n_actions
        self.epsilon = cfg.QMIX_EPSILON_START
        self.lr = cfg.QMIX_LR
        self.epsilon_decay = (cfg.QMIX_EPSILON_START - cfg.QMIX_EPSILON_END) / cfg.QMIX_EPSILON_DECAY
        self.obs_scale = 500.0
        self.grad_clip = 10.0

    def _norm(self, obs):
        return np.asarray(obs, dtype=np.float64) / self.obs_scale

    def forward(self, obs):
        return self.net.forward(self._norm(obs))

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.zeros(self.n_actions)
            chosen = np.random.choice(self.n_actions, size=cfg.N_BEAMS, replace=False)
            action[chosen] = 1
        else:
            q = self.forward(obs)
            action = np.zeros(self.n_actions)
            chosen = np.argsort(-q)[:cfg.N_BEAMS]
            action[chosen] = 1
        return action

    def update(self, obs, action, reward, next_obs, done):
        """一步 TD 更新 (DQN style)"""
        q = self.forward(obs)
        q_next = self.forward(next_obs)
        n_act = max(np.sum(action), 1)
        q_val = np.sum(q * action) / n_act
        target = reward + cfg.QMIX_GAMMA * np.max(q_next) * (1 - done)
        td = q_val - target
        # 梯度裁剪
        td = np.clip(td, -self.grad_clip, self.grad_clip)
        dq = td * action / n_act
        dW1, db1, dW2, db2 = self.net.backward(dq)
        # 梯度裁剪
        norm = np.sqrt(np.sum(dW1**2) + np.sum(dW2**2)) + 1e-8
        if norm > self.grad_clip:
            dW1 *= self.grad_clip / norm
            dW2 *= self.grad_clip / norm
        self.net.W1 -= self.lr * dW1
        self.net.b1 -= self.lr * db1
        self.net.W2 -= self.lr * dW2
        self.net.b2 -= self.lr * db2
        self.epsilon = max(cfg.QMIX_EPSILON_END, self.epsilon - self.epsilon_decay)
        return td


class IQLAgent:
    """独立 Q 学习 (IQL / MADRL-BH)"""
    def __init__(self, obs_dim, n_actions, hidden_dim=cfg.QMIX_GRU_HIDDEN_DIM):
        self.net = FeedforwardNet(obs_dim, hidden_dim, n_actions, seed=0)
        self.n_actions = n_actions
        self.epsilon = cfg.QMIX_EPSILON_START
        self.lr = cfg.QMIX_LR
        self.epsilon_decay = (cfg.QMIX_EPSILON_START - cfg.QMIX_EPSILON_END) / cfg.QMIX_EPSILON_DECAY
        self.obs_scale = 500.0
        self.grad_clip = 10.0

    def _norm(self, obs):
        return np.asarray(obs, dtype=np.float64) / self.obs_scale

    def forward(self, obs):
        return self.net.forward(self._norm(obs))

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.zeros(self.n_actions)
            chosen = np.random.choice(self.n_actions, size=cfg.N_BEAMS, replace=False)
            action[chosen] = 1
        else:
            q = self.forward(obs)
            action = np.zeros(self.n_actions)
            chosen = np.argsort(-q)[:cfg.N_BEAMS]
            action[chosen] = 1
        return action

    def update(self, obs, action, reward, next_obs, done):
        q = self.forward(obs)
        q_next = self.forward(next_obs)
        n_act = max(np.sum(action), 1)
        q_val = np.sum(q * action) / n_act
        target = reward + cfg.QMIX_GAMMA * np.max(q_next) * (1 - done)
        td = q_val - target
        td = np.clip(td, -self.grad_clip, self.grad_clip)
        dq = td * action / n_act
        dW1, db1, dW2, db2 = self.net.backward(dq)
        norm = np.sqrt(np.sum(dW1**2) + np.sum(dW2**2)) + 1e-8
        if norm > self.grad_clip:
            dW1 *= self.grad_clip / norm
            dW2 *= self.grad_clip / norm
        self.net.W1 -= self.lr * dW1
        self.net.b1 -= self.lr * db1
        self.net.W2 -= self.lr * dW2
        self.net.b2 -= self.lr * db2
        self.epsilon = max(cfg.QMIX_EPSILON_END, self.epsilon - self.epsilon_decay)
        return td