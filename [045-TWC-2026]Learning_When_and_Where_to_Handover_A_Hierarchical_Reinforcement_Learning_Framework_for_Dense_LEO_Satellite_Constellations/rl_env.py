# -*- coding: utf-8 -*-
"""
rl_env.py
================================================================================
空间层 DQN 训练环境（论文 Eq.30-37）。

【设计目标】
把"切换目标选择"封装为 RL 决策：
  - 决策时机：沿用 handover.py 的触发逻辑（强制切换/ψ迟滞自愿切换）判定
    "是否要切换"；当要切换时由 DQN 选目标星（替换原来的 argmax ψ）。
  - state  (Eq.30-32)：候选星特征矩阵 padding 到 max_vis，每候选
              [L̃, H̃, τ̃, visibility_mask] + 当前服务星特征。
  - action  (Eq.33-36)：选一个候选星索引；不可见/越界用掩码 mask
              （masked Q，不可选的 Q 设 -inf）。
  - reward  (Eq.37 前向、不止瞬时 ψ)：
              r = ψ(选中星；当前权重) − ξ(切换代价) − (失败惩罚) + (速率奖励)
    关键：reward 反映真实切换后果（成败/速率），让 DQN 学到比瞬时 ψ-贪心更好的
    前向策略（例如避开高 ψ 但即将过载的星）。

【诚实保真度边界 / HONEST FIDELITY BOUNDARY】
论文无种子/无代码，即使训练也不可能逐点一致。目标是：
  真实训练 + 定性一致（Fig.5 收敛曲线形状、Fig.6/9 趋势与算法排序）。
  绝不查表/硬编码输出。所有 reward/transition 由前向物理仿真涌现。

训练用简化配置：20 用户 / 200s episode（性能：~2.5s/episode）。
================================================================================
"""

import numpy as np

from config import (CONSTELLATION, USER, RF, DECISION, SEED,
                    compute_sinr_floor_db)
from environment import Environment
from handover import qos_utility


# 训练专用默认配置：20 用户 / 200s
TRAIN_USER_OVERRIDE = dict(
    num_users=20,
    traffic_poisson_lambda=USER['traffic_poisson_lambda'],  # 保留论文业务压力
    mean_session_s=USER['mean_session_s'],
    sim_duration_s=200.0,
    dt_s=USER['dt_s'],
    fine_step_s=USER['fine_step_s'],
)


# ============================================================================
# 用户位置（中纬区，与仿真一致）
# ============================================================================
def _gen_users(lat_range, lon_range, n_users, seed):
    rng = np.random.default_rng(seed)
    lats = rng.uniform(lat_range[0], lat_range[1], n_users)
    lons = rng.uniform(lon_range[0], lon_range[1], n_users)
    return lats, lons


# ============================================================================
# 空间层 RL 环境
# ============================================================================
class SpatialEnv:
    """空间层 DQN 训练环境。

    一个 episode = 一条 200s trajectory。每个时隙对每 active 用户：
      1) 用 handover 触发逻辑判定是否要切换
      2) 要切换 → 调 step(user) 由策略选目标星，得 reward，转移到下一时隙
    """

    def __init__(self, weights, n_users=20, episode_s=200.0,
                 max_vis_cap=20, seed=SEED, env_reuse=None,
                 xi_ho=0.05, p_fail=0.5, rate_bonus=0.3,
                 traffic_lambda_mult=None, extra_loss_dB=None,
                 effective_user_mult=None, visibility_fragmentation=None,
                 rf_cfg=None):
        """
        weights : (w_load, w_signal, w_time) 三元组，决定 ψ 权重（reward 用）
        max_vis_cap : 候选维度的固定上限（pad 到此）。20 对中纬密集星座够用。
        xi_ho  : 切换代价 ξ（每次成功切换的固定成本，对应信令开销）
        p_fail : 失败惩罚（切换后 SINR<SINR_floor 或过载）
        rate_bonus : 速率奖励系数（选中星可达 Shannon 速率归一）
        env_reuse : 若提供已构造的 Environment，复用其几何（reuse_geometry_from）
        """
        self.weights = weights
        self.wL, self.wH, self.wT = weights
        self.n_users = n_users
        self.episode_s = episode_s
        self.max_vis = max_vis_cap
        self.xi_ho = xi_ho
        self.p_fail = p_fail
        self.rate_bonus = rate_bonus
        self.seed = seed

        # 训练用 USER 配置（覆盖 num_users / 时长，其余物理参数沿用 USER）
        user_cfg = dict(USER)
        user_cfg['num_users'] = n_users
        user_cfg['sim_duration_s'] = episode_s

        lats, lons = _gen_users(USER['region_lat'], USER['region_lon'],
                                n_users, seed)
        # 构造训练 Environment（可能复用 env_reuse 的几何）
        self.env = Environment(
            lats, lons,
            user_cfg=user_cfg,
            rf_cfg=rf_cfg or RF,
            seed=seed,
            traffic_lambda_mult=traffic_lambda_mult,
            extra_loss_dB=extra_loss_dB,
            effective_user_mult=effective_user_mult,
            visibility_fragmentation=visibility_fragmentation,
            reuse_geometry_from=env_reuse,
        )

        # 决策参数（从 DECISION 取，与 handover.py 一致）
        self.tau_force = DECISION['tau_force_s']
        self.tau_ref = DECISION['tau_ref_s']
        self.delta_psi = DECISION['delta_psi_nonhrl']  # 训练用非分层触发
        self.T_dwell = DECISION['T_dwell_nonhrl']
        self.sinr_floor_db = compute_sinr_floor_db()
        self.sinr_floor_lin = 10 ** (self.sinr_floor_db / 10.0)
        self.R_min = RF['min_rate_Mbps']
        self.N_beam = RF.get('sat_beam_max', 40)
        self.kappa_load = RF.get('load_share_kappa', 0.12)
        self.shadow_sigma = RF.get('shadow_sigma_dB', 5.0)
        self.load_penalty_coef = DECISION.get('load_penalty_db_per_L', 7.0)

        # 速率归一化参考（用于 rate_bonus 标度，让 reward 量级合理）
        # 取环境典型好链路 Shannon 速率作为参考
        self._rate_ref_mbps = max(self.env.ch.rate_mbps(
            np.array([self.sinr_floor_lin * 4.0]))[0], 1.0)

        # 状态维度（展平候选矩阵 + 当前服务星特征）
        # 候选矩阵：max_vis × 4 ([L̃, H̃, τ̃, mask])
        # 当前服务星特征：4 ([L̃_cur, H̃_cur, τ̃_cur, has_cur])
        self.state_dim = self.max_vis * 4 + 4
        self.action_dim = self.max_vis

        # episode 运行状态
        self._reset_runstate()

    # ------------------------------------------------------------------
    def _reset_runstate(self):
        """重置 episode 运行状态（不重建 env）。"""
        N = self.env.const.N
        self.serving = -np.ones(self.env.U, dtype=np.int64)
        self.last_ho_time = -np.full(self.env.U, 1e9)
        self.load_view = np.zeros(N)
        self.t_idx = 0
        self._snap = None
        self._pending_user = None  # 当前等待决策的用户（None=本时隙已处理完）
        self.episode_reward = 0.0
        self.episode_steps = 0

    def reset(self):
        """重置 episode。返回首个 state（等待第一个切换决策）。"""
        self._reset_runstate()
        # 推进到第一个需要决策的时隙/用户
        self._snap = self.env.snapshot(self.t_idx)
        state, user, mask = self._advance_to_next_decision()
        return state, dict(user=user, action_mask=mask)

    # ------------------------------------------------------------------
    # 在当前快照上扫描所有 active 用户，找第一个触发切换的用户
    # ------------------------------------------------------------------
    def _check_user_trigger(self, snap, u, serving_u):
        """复用 handover.py 触发逻辑，返回 (need_switch, best_psi_for_baseline, info)。
        need_switch=True 表示该用户需要做切换决策。
        info 含候选星数组 + 当前服务星特征。
        """
        idx_row = snap.visible_idx[u]
        valid_k = np.where(idx_row >= 0)[0]
        if len(valid_k) == 0:
            return False, None
        sat_ids = idx_row[valid_k]
        H = snap.H_norm_cand[u, valid_k]
        tau_raw = snap.tau_cand_s[u, valid_k]
        tau = np.clip(tau_raw / self.tau_ref, 0, 1)
        L = np.clip(self.load_view[sat_ids], 0, 1)
        psi = qos_utility(L, H, tau, self.wL, self.wH, self.wT)
        best_k_local = int(np.argmax(psi))
        best_psi = float(psi[best_k_local])
        best_sat = int(sat_ids[best_k_local])

        cur_sat = int(serving_u)
        t_s = snap.t_s
        # 情况 1: 首次接入或当前服务星不可见 → 强制接入/切换
        if cur_sat < 0 or cur_sat not in sat_ids:
            return True, dict(valid_k=valid_k, sat_ids=sat_ids,
                              H=H, tau=tau, L=L, psi=psi,
                              cur_sat=cur_sat, reason='force_attach')
        # 当前服务星索引
        cur_k_local = int(np.where(sat_ids == cur_sat)[0][0])
        cur_H = H[cur_k_local]; cur_tau = tau[cur_k_local]
        cur_L = L[cur_k_local]
        cur_psi = float(psi[cur_k_local])

        # 情况 2: 强制切换（当前服务星即将失可见）
        if cur_tau * self.tau_ref < self.tau_force:
            return True, dict(valid_k=valid_k, sat_ids=sat_ids,
                              H=H, tau=tau, L=L, psi=psi,
                              cur_sat=cur_sat, reason='force_loss')
        # 情况 3: 自愿切换（ψ 迟滞触发 + 驻留时间）
        dwell_ok = (t_s - self.last_ho_time[u]) > self.T_dwell
        trig_psi = (best_psi - cur_psi) > self.delta_psi
        if dwell_ok and trig_psi:
            return True, dict(valid_k=valid_k, sat_ids=sat_ids,
                              H=H, tau=tau, L=L, psi=psi,
                              cur_sat=cur_sat, reason='voluntary',
                              cur_psi=cur_psi)
        return False, None

    # ------------------------------------------------------------------
    def _build_state(self, snap, u, info):
        """构造状态向量（展平候选矩阵 + 当前服务星特征）。"""
        valid_k = info['valid_k']
        sat_ids = info['sat_ids']
        H = info['H']; tau = info['tau']; L = info['L']
        n_cand = len(valid_k)
        # 候选矩阵 (max_vis, 4)：[L̃, H̃, τ̃, mask]
        cand = np.zeros((self.max_vis, 4), dtype=np.float32)
        if n_cand > 0:
            n_fill = min(n_cand, self.max_vis)
            cand[:n_fill, 0] = L[:n_fill]
            cand[:n_fill, 1] = H[:n_fill]
            cand[:n_fill, 2] = tau[:n_fill]
            cand[:n_fill, 3] = 1.0  # visibility mask
        cand_flat = cand.flatten()
        # 当前服务星特征
        cur_sat = info.get('cur_sat', -1)
        if cur_sat >= 0 and cur_sat in sat_ids:
            cur_k_local = int(np.where(sat_ids == cur_sat)[0][0])
            cur_feat = np.array([L[cur_k_local], H[cur_k_local],
                                 tau[cur_k_local], 1.0], dtype=np.float32)
        else:
            cur_feat = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([cand_flat, cur_feat]).astype(np.float32)

    def _action_mask(self, snap, u, info):
        """构造动作掩码：可见候选=1，padding/不可见=0。"""
        n_cand = len(info['valid_k'])
        mask = np.zeros(self.max_vis, dtype=bool)
        if n_cand > 0:
            mask[:min(n_cand, self.max_vis)] = True
        return mask

    # ------------------------------------------------------------------
    def _advance_to_next_decision(self):
        """从当前 (t_idx, user_idx) 扫描，找到第一个需要决策的用户。
        返回 (state, user, mask)。若本 episode 结束，返回 (None, None, None)。
        """
        U = self.env.U
        T = self.env.T
        while self.t_idx < T:
            snap = self._snap
            # 从当前用户开始扫描到末尾
            start_u = self._pending_user if self._pending_user is not None else 0
            for u in range(start_u, U):
                if not snap.active_mask[u]:
                    continue
                need, info = self._check_user_trigger(
                    snap, u, self.serving[u])
                if need:
                    self._pending_user = u
                    self._pending_info = info
                    state = self._build_state(snap, u, info)
                    mask = self._action_mask(snap, u, info)
                    return state, u, mask
            # 本时隙所有用户处理完 → 推进到下一时隙
            self.t_idx += 1
            self._pending_user = None
            if self.t_idx < T:
                self._snap = self.env.snapshot(self.t_idx)
        return None, None, None

    # ------------------------------------------------------------------
    # 应用一次切换决策：执行前向物理后果（成败/速率），返回 reward
    # ------------------------------------------------------------------
    def _execute_and_reward(self, snap, u, action_k, info):
        """执行切换到 action_k 候选星，计算 reward。"""
        valid_k = info['valid_k']
        sat_ids = info['sat_ids']
        H = info['H']; tau = info['tau']; L = info['L']

        # 映射 action → 候选星 idx
        n_cand = len(valid_k)
        if action_k >= n_cand:
            # 越界（不应发生，mask 应阻止），给大惩罚
            return -1.0, False, 0.0
        cand_k_local = valid_k[action_k]  # 在 idx_row 中的位置
        target_sat = int(sat_ids[action_k])
        # 候选在 action_k 处的归一化特征
        L_t = float(L[action_k])
        H_t = float(H[action_k])
        tau_t = float(tau[action_k])
        psi_t = qos_utility(L_t, H_t, tau_t, self.wL, self.wH, self.wT)

        old_sat = int(self.serving[u])
        is_switch = (target_sat != old_sat)

        # ---- 前向物理后果：成败 + 有效速率 ----
        # 复用 handover.py 的瞬时 SINR + 负载边际 + 阴影波动逻辑
        sinr_dec_lin = snap.sinr_cand[u, cand_k_local]
        sinr_dec_db = 10 * np.log10(max(sinr_dec_lin, 1e-12))
        # 切换后该星用户数（含本次切换）
        n_u_target = int(round(self.load_view[target_sat] * self.N_beam)) + (1 if is_switch else 0)
        load_target = n_u_target / float(self.N_beam)
        L_ev_clip = np.clip(load_target, 0, 1)
        load_penalty_db = self.load_penalty_coef * L_ev_clip
        sinr_inst_db = (sinr_dec_db
                        + self.env.rng.normal(0, self.shadow_sigma)
                        - load_penalty_db)
        sinr_inst_lin = 10 ** (sinr_inst_db / 10.0)
        rate_inst = float(self.env.ch.rate_mbps(np.array([sinr_inst_lin]))[0])
        # 成败：瞬时 rate ≥ R_min 且容量未阻塞
        success = (rate_inst >= self.R_min) and (load_target < 1.0)
        rate_eff = rate_inst if load_target < 1.0 else 0.0

        # ---- Reward 组合 ----
        # (1) ψ 项（主奖励：选中星的 QoS 效用）
        r = psi_t
        # (2) 切换代价 ξ（每次切换的固定信令成本）
        if is_switch:
            r -= self.xi_ho
        # (3) 失败惩罚
        if not success:
            r -= self.p_fail
        # (4) 速率奖励（选中星可达 Shannon 速率归一，让 DQN 倾向好链路）
        r += self.rate_bonus * (rate_eff / self._rate_ref_mbps)

        # 应用决策：更新服务关联
        self.serving[u] = target_sat
        if is_switch:
            self.last_ho_time[u] = snap.t_s
        return float(r), success, rate_eff

    def _update_load_after_step(self):
        """每时隙结束时根据全网关联更新负载视图。"""
        # 用一个简化速率（每用户服务链路 Shannon 速率）回填负载
        snap = self._snap
        # 简化：每 active 用户的服务链路速率用其候选中服务星的 Shannon 速率
        rates = np.zeros(self.env.U)
        for u in range(self.env.U):
            s = self.serving[u]
            if s < 0:
                continue
            k = self.env.find_serving_k(snap, u, s)
            if k >= 0:
                rates[u] = snap.rate_cand_mbps[u, k]
        L = self.env.update_load(snap, self.serving, rates)
        self.load_view = L

    # ------------------------------------------------------------------
    def step(self, action):
        """执行一次切换决策（针对当前 pending 用户）。
        返回 (next_state, reward, done, info)。
        next_state: 若有下一个待决策用户则返回其 state，否则返回本时隙剩余处理后的
                    下一个 state；episode 结束返回 None。
        """
        assert self._pending_user is not None, "step() 调用前必须先 reset/advance"
        u = self._pending_user
        info = self._pending_info
        snap = self._snap

        reward, success, rate_eff = self._execute_and_reward(
            snap, u, int(action), info)
        self.episode_reward += reward
        self.episode_steps += 1

        # 推进到下一个待决策用户
        self._pending_user = u + 1
        self._pending_info = None
        # 若本时隙还有用户待扫描，先扫描本时隙
        next_state, next_u, next_mask = self._advance_to_next_decision()

        # 当跨时隙时（_advance_to_next_decision 内 t_idx 自增），
        # 在进入新时隙前更新上一时隙的负载视图
        if next_state is None:
            done = True
        else:
            done = False
        info_out = dict(
            user=next_u, action_mask=next_mask,
            success=success, rate_eff=rate_eff,
            step_reward=reward,
        )
        return next_state, reward, done, info_out

    # ------------------------------------------------------------------
    # 评估辅助：用策略 rollout 一条完整 episode，返回累积 reward + 统计
    # ------------------------------------------------------------------
    def rollout(self, policy_fn, max_steps=100000):
        """policy_fn(state, mask) -> action。返回 (total_reward, n_steps, n_switches, n_success, n_fail)。"""
        state, info = self.reset()
        total_r = 0.0
        n_steps = 0; n_sw = 0; n_suc = 0; n_fail = 0
        while state is not None and n_steps < max_steps:
            mask = info['action_mask']
            action = policy_fn(state, mask)
            state, r, done, info = self.step(action)
            total_r += r
            n_steps += 1
            # 统计：通过 step_reward 符号粗略判断（更精确可在 _execute 内计数）
        return total_r, n_steps


# ============================================================================
# 时间层 PPO+GRU 训练环境（论文 Eq.16-21）
#   把"何时切换"封装为 RL 决策：per active 用户 per 时隙，二值动作 {0=keep, 1=trigger}。
#   trigger=1 时"切到哪"由已训练的空间层 DQN（spatial_policy_fn）决定；trigger=0 留当前星。
#   state: o_seq (k,3) 历史序列（GRU 输入）+ o_ins 瞬时向量（5 维）。
#   reward 四 case（Eq.21），reward_case 参数控制激活集（Fig.4 消融）。
# ============================================================================
from collections import deque


class TemporalEnv:
    """时间层 PPO+GRU 训练环境（论文 Eq.16-21）。

    一个 episode = 一条 200s trajectory。每个时隙对每 active 用户做一次二值决策
    （是否触发切换），decision 走完整个 trajectory。

    spatial_policy_fn : 已训练空间层 DQN 的 greedy 策略回调（决定 trigger=1 时切到哪）。
        签名同 eval_rl.make_policy_fn：
            policy_fn(snap, u, valid_k, H, tau, L, psi, cur_sat, max_vis_cap)
                -> (chosen_local_idx, chosen_sat)
    """

    def __init__(self, weights, spatial_policy_fn, max_vis_cap=20,
                 n_users=20, episode_s=200.0, seed=SEED,
                 k_hist=10, reward_case='case234',
                 eta_force=1.0, eta_improve=1.0, xi_ho=0.05,
                 p_pingpong=0.5, env_reuse=None,
                 traffic_lambda_mult=None, extra_loss_dB=None,
                 rf_cfg=None):
        """
        weights           : (wL, wH, wT) — ψ 权重（reward 用，与空间层 DQN 一致）
        spatial_policy_fn : 决定 trigger=1 时切到哪颗候选星（DQN greedy）
        max_vis_cap       : 候选维上限（与空间层 DQN 训练一致）
        k_hist            : GRU 历史序列长度（论文 k=10）
        reward_case       : {'case2','case23','case234'} 控制激活的奖励 case
                            （Fig.4 三条消融曲线对应）
        eta_force         : η（Case2 必要切换奖励系数）
        eta_improve       : η'（Case3 改善切换奖励系数）
        xi_ho             : ξ（每次切换的固定信令代价）
        p_pingpong        : p（Case4 不必要切换惩罚）
        env_reuse         : 复用几何（reuse_geometry_from）
        """
        self.weights = weights
        self.wL, self.wH, self.wT = weights
        self.spatial_policy_fn = spatial_policy_fn
        self.max_vis = max_vis_cap
        self.n_users = n_users
        self.episode_s = episode_s
        self.seed = seed
        self.k_hist = k_hist
        self.reward_case = reward_case
        self.eta_force = eta_force
        self.eta_improve = eta_improve
        self.xi_ho = xi_ho
        self.p_pingpong = p_pingpong

        # 训练用 USER 配置（覆盖 num_users / 时长）
        user_cfg = dict(USER)
        user_cfg['num_users'] = n_users
        user_cfg['sim_duration_s'] = episode_s
        lats, lons = _gen_users(USER['region_lat'], USER['region_lon'],
                                n_users, seed)
        self.env = Environment(
            lats, lons, user_cfg=user_cfg,
            rf_cfg=rf_cfg or RF, seed=seed,
            traffic_lambda_mult=traffic_lambda_mult,
            extra_loss_dB=extra_loss_dB,
            reuse_geometry_from=env_reuse,
        )

        # 决策参数（同 SpatialEnv）
        self.tau_force = DECISION['tau_force_s']
        self.tau_ref = DECISION['tau_ref_s']
        self.sinr_floor_db = compute_sinr_floor_db()
        self.sinr_floor_lin = 10 ** (self.sinr_floor_db / 10.0)
        self.R_min = RF['min_rate_Mbps']
        self.N_beam = RF.get('sat_beam_max', 40)
        self.shadow_sigma = RF.get('shadow_sigma_dB', 5.0)
        self.load_penalty_coef = DECISION.get('load_penalty_db_per_L', 7.0)

        # 状态维度：
        #   o_seq: (k_hist, 3) — GRU 序列输入
        #   o_ins: (5,) — [θ(当前仰角归一), ψ(当前), Δψ(增益), 距上次切换/τ_ref, τ̃(当前)]
        self.seq_dim = (k_hist, 3)
        self.ins_dim = 5
        # 用于 critic 的扁平状态维度
        self.state_flat_dim = k_hist * 3 + self.ins_dim
        self.action_dim = 2

        self._reset_runstate()

    # ------------------------------------------------------------------
    def _reset_runstate(self):
        N = self.env.const.N
        self.serving = -np.ones(self.env.U, dtype=np.int64)
        self.last_ho_time = -np.full(self.env.U, 1e9)
        self.load_view = np.zeros(N)
        self.t_idx = 0
        self._snap = None
        self._pending_user = None
        self._pending_info = None
        # 每用户服务历史 deque(k_hist, [L̃, H̃, τ̃]) — 论文 Eq.17
        # 初值填零（不足 k 时 GRU 前面用零填充）
        self.hist = {u: deque(maxlen=self.k_hist) for u in range(self.env.U)}
        self.episode_reward = 0.0
        self.episode_steps = 0

    def reset(self):
        """重置 episode。返回首个状态 (o_seq, o_ins)，等待决策。
        若整个 episode 无任何 active 用户时隙，返回 (None, None)。"""
        self._reset_runstate()
        self._snap = self.env.snapshot(self.t_idx)
        seq, ins = self._advance_to_next_decision()
        return seq, ins

    # ------------------------------------------------------------------
    def _user_cand_info(self, snap, u, serving_u):
        """对单用户构造候选星信息（与 SpatialEnv._check_user_trigger 类似，
        但不应用触发逻辑——触发由 PPO 决定）。返回 dict 或 None（无可见星）。"""
        idx_row = snap.visible_idx[u]
        valid_k = np.where(idx_row >= 0)[0]
        if len(valid_k) == 0:
            return None
        sat_ids = idx_row[valid_k]
        H = snap.H_norm_cand[u, valid_k]
        tau_raw = snap.tau_cand_s[u, valid_k]
        tau = np.clip(tau_raw / self.tau_ref, 0, 1)
        L = np.clip(self.load_view[sat_ids], 0, 1)
        psi = qos_utility(L, H, tau, self.wL, self.wH, self.wT)
        best_k_local = int(np.argmax(psi))
        best_psi = float(psi[best_k_local])
        best_sat = int(sat_ids[best_k_local])
        return dict(valid_k=valid_k, sat_ids=sat_ids,
                    H=H, tau=tau, tau_raw=tau_raw, L=L, psi=psi,
                    best_k_local=best_k_local, best_psi=best_psi,
                    best_sat=best_sat, cur_sat=int(serving_u))

    # ------------------------------------------------------------------
    def _build_state(self, snap, u, info):
        """构造 (o_seq, o_ins)。"""
        hist = self.hist[u]
        # o_seq: (k_hist, 3) — 历史 [L̃, H̃, τ̃]，不足 k 用零前填
        seq = np.zeros((self.k_hist, 3), dtype=np.float32)
        h_arr = np.array(list(hist), dtype=np.float32) if len(hist) > 0 \
            else np.zeros((0, 3), dtype=np.float32)
        if h_arr.shape[0] > 0:
            seq[-h_arr.shape[0]:] = h_arr  # 末尾对齐（最近 k 步）
        # o_ins: 5 维瞬时向量
        #   [θ(当前仰角归一化), ψ(当前), Δψ(增益), 距上次切换时间/τ_ref, τ̃(当前)]
        sat_ids = info['sat_ids']
        cur_sat = info['cur_sat']
        if cur_sat >= 0 and cur_sat in sat_ids:
            cur_k_local = int(np.where(sat_ids == cur_sat)[0][0])
            cur_H = float(info['H'][cur_k_local])
            cur_tau = float(info['tau'][cur_k_local])
            cur_L = float(info['L'][cur_k_local])
            cur_psi = float(info['psi'][cur_k_local])
        else:
            # 首次接入或当前星不可见：用最优候选代理 + 标记紧急
            cur_k_local = info['best_k_local']
            cur_H = float(info['H'][cur_k_local])
            cur_tau = float(info['tau'][cur_k_local])
            cur_L = float(info['L'][cur_k_local])
            cur_psi = float(info['psi'][cur_k_local])
        # 当前仰角归一化：用 sin(elev) 近似（zenith=1, horizon=0）
        cur_elev = float(snap.visible_elev[u, info['valid_k'][cur_k_local]])
        theta_norm = np.clip(np.sin(np.deg2rad(cur_elev)), 0.0, 1.0)
        # Δψ 增益：最优候选 ψ − 当前 ψ（≥0）
        gain_psi = max(0.0, info['best_psi'] - cur_psi)
        # 距上次切换时间（归一化到 τ_ref；首次接入填大值表示很久）
        t_since_ho = (snap.t_s - self.last_ho_time[u])
        t_since_norm = np.clip(t_since_ho / self.tau_ref, 0.0, 5.0) / 5.0  # [0,1]
        ins = np.array([theta_norm, cur_psi, gain_psi, t_since_norm, cur_tau],
                       dtype=np.float32)
        return seq, ins

    # ------------------------------------------------------------------
    def _advance_to_next_decision(self):
        """从当前 (t_idx, user_idx) 扫描，找下一个需要决策的 active 用户时隙。
        返回 (o_seq, o_ins)。episode 结束返回 (None, None)。"""
        U = self.env.U
        T = self.env.T
        while self.t_idx < T:
            snap = self._snap
            start_u = self._pending_user if self._pending_user is not None else 0
            for u in range(start_u, U):
                if not snap.active_mask[u]:
                    continue
                info = self._user_cand_info(snap, u, self.serving[u])
                if info is None:
                    continue
                # 该用户在本时隙需要做时间层决策
                self._pending_user = u
                self._pending_info = info
                seq, ins = self._build_state(snap, u, info)
                return seq, ins
            # 本时隙所有用户处理完 → 更新负载 + 推进到下一时隙
            self._update_load_after_step()
            self.t_idx += 1
            self._pending_user = None
            if self.t_idx < T:
                self._snap = self.env.snapshot(self.t_idx)
        return None, None

    # ------------------------------------------------------------------
    def _update_load_after_step(self):
        """每时隙结束时根据全网关联更新负载视图（同 SpatialEnv）。"""
        snap = self._snap
        rates = np.zeros(self.env.U)
        for u in range(self.env.U):
            s = self.serving[u]
            if s < 0:
                continue
            k = self.env.find_serving_k(snap, u, s)
            if k >= 0:
                rates[u] = snap.rate_cand_mbps[u, k]
        L = self.env.update_load(snap, self.serving, rates)
        self.load_view = L

    # ------------------------------------------------------------------
    def _execute_and_reward(self, snap, u, action, info):
        """执行二值决策（0=keep, 1=trigger），计算 reward。
        trigger=1：调 spatial_policy_fn 选目标星（DQN greedy），执行切换。
        trigger=0：保持当前星（首次接入时若 a=0 强制接入最优候选——否则掉线）。

        reward 四 case（Eq.21，由 reward_case 控制激活）：
          Case1 (a=0, 非紧急留在当前星):  +ψ(当前)
          Case2 (a=1, 必要切换——紧急/失可见):  +η·ψ(target)   [case2/23/234]
          Case3 (a=1, 改善切换——Δψ 显著):  +η'·ψ(target) − ξ [case23/234]
          Case4 (a=1, 不必要切换——乒乓):  −p                [case234]
        返回 (reward, switched, target_sat, psi_target, reason)。
        """
        valid_k = info['valid_k']
        sat_ids = info['sat_ids']
        H = info['H']; tau = info['tau']; L = info['L']; psi = info['psi']
        tau_raw = info['tau_raw']
        cur_sat = info['cur_sat']
        best_psi = info['best_psi']
        best_sat = info['best_sat']

        # 判定当前是否"紧急"（即将失可见）
        if cur_sat >= 0 and cur_sat in sat_ids:
            cur_k_local = int(np.where(sat_ids == cur_sat)[0][0])
            cur_tau_raw = float(tau_raw[cur_k_local])
            cur_psi = float(psi[cur_k_local])
        else:
            cur_k_local = info['best_k_local']
            cur_tau_raw = float(tau_raw[cur_k_local])
            cur_psi = float(psi[cur_k_local])
        urgent = (cur_tau_raw < self.tau_force)

        # 首次接入/当前星不可见：无论 a 取何值都必须接入（强制）
        no_serving = (cur_sat < 0) or (cur_sat not in sat_ids)

        # 决定目标星 + 是否真正切换
        if action == 1 or no_serving:
            # trigger：由空间层 DQN（spatial_policy_fn）选目标
            chosen_local, chosen_sat = self.spatial_policy_fn(
                snap, u, valid_k, H, tau, L, psi, cur_sat, self.max_vis)
            if chosen_sat is None:
                chosen_local = info['best_k_local']
                chosen_sat = best_sat
            target_sat = int(chosen_sat)
            target_k_local = int(chosen_local)
            switched = (target_sat != cur_sat)
        else:
            # keep：留在当前星
            target_sat = cur_sat
            target_k_local = cur_k_local
            switched = False

        # 目标星 ψ（在 valid_k 内的下标）
        # 注意：DQN 可能选到非最优候选，用其真实 ψ 作为 reward 输入
        psi_target = float(psi[target_k_local])

        # ---- reward 四 case（Eq.21）----
        if action == 0 and not no_serving:
            # Case1：留在当前星（非紧急，由 PPO 选择 keep）
            reward = cur_psi
            reason = 'case1_keep'
        elif action == 1 and (urgent or no_serving):
            # Case2：必要切换（紧急即将失可见，或首次接入）
            reward = self.eta_force * psi_target
            reason = 'case2_force'
        elif action == 1 and (best_psi - cur_psi) > (self.xi_ho + 1e-6):
            # Case3：改善切换（Δψ 显著，超过 ξ 阈值）
            reward = self.eta_improve * psi_target - self.xi_ho
            reason = 'case3_improve'
        elif action == 1:
            # Case4：不必要切换（非紧急、无显著改善 = 乒乓）
            reward = -self.p_pingpong
            reason = 'case4_pingpong'
        else:
            # 兜底（不应到达）
            reward = cur_psi
            reason = 'case1_keep'

        # 根据 reward_case 过滤：未激活的 case 降级为 Case1（keep reward）
        #   'case2'   : 只 Case1 + Case2（紧急切换奖励）；Case3/4 不激活
        #               → 任何 a=1 非紧急视为"白切换"，奖励 = cur_psi（等于 keep）
        #                 但仍有切换代价（在下面扣 ξ），让 RL 学到"只在紧急才切"
        #   'case23'  : Case1 + Case2 + Case3；Case4 不激活（乒乓无额外惩罚）
        #   'case234' : 全部激活（完整奖励）
        if reason == 'case3_improve' and self.reward_case == 'case2':
            reward = cur_psi  # 降级为 keep 等价
            reason = 'case1_keep_degraded'
        if reason == 'case4_pingpong' and self.reward_case in ('case2', 'case23'):
            reward = cur_psi  # 降级为 keep 等价
            reason = 'case1_keep_degraded'
        # 切换代价 ξ：任何真实切换（switched=True）都扣，无论 reward_case
        # （物理理由：每次切换有信令开销，独立于奖励 case）
        if switched:
            reward -= self.xi_ho

        # 应用决策：更新服务关联 + 历史
        self.serving[u] = target_sat
        if switched:
            self.last_ho_time[u] = snap.t_s
        # 历史记录本次服务星特征（用于下一时隙 GRU 输入）
        target_L = float(L[target_k_local])
        target_H = float(H[target_k_local])
        target_tau = float(tau[target_k_local])
        self.hist[u].append([target_L, target_H, target_tau])

        return float(reward), switched, target_sat, psi_target, reason

    # ------------------------------------------------------------------
    def step(self, action):
        """执行一次时间层二值决策。返回 (next_seq, next_ins, reward, done, info)。"""
        assert self._pending_user is not None, "step() 前必须先 reset/advance"
        u = self._pending_user
        info = self._pending_info
        snap = self._snap
        a = int(action)
        assert a in (0, 1), f"动作必须二值 {{0,1}}，收到 {a}"

        reward, switched, target_sat, psi_target, reason = \
            self._execute_and_reward(snap, u, a, info)
        self.episode_reward += reward
        self.episode_steps += 1

        # 推进到下一个待决策用户/时隙
        self._pending_user = u + 1
        self._pending_info = None
        next_seq, next_ins = self._advance_to_next_decision()

        done = (next_seq is None)
        info_out = dict(
            user=u, action=a, switched=switched, target_sat=target_sat,
            psi_target=psi_target, reason=reason, step_reward=reward,
        )
        return next_seq, next_ins, reward, done, info_out

    # ------------------------------------------------------------------
    def rollout(self, policy_fn, max_steps=100000):
        """policy_fn(seq, ins) -> int {0,1}。返回 (total_reward, n_steps, n_switches)。"""
        seq, ins = self.reset()
        total_r = 0.0; n_steps = 0; n_sw = 0
        while seq is not None and n_steps < max_steps:
            a = int(policy_fn(seq, ins))
            seq, ins, r, done, info = self.step(a)
            total_r += r; n_steps += 1
            if info.get('switched'):
                n_sw += 1
        return total_r, n_steps, n_sw


if __name__ == '__main__':
    # 自检
    import time
    t0 = time.time()
    env = SpatialEnv(weights=(0.4, 0.3, 0.3), n_users=20, seed=SEED)
    print(f'[rl_env] 构造耗时 {time.time()-t0:.2f}s, T={env.env.T}, '
          f'max_vis(env)={env.env.max_vis}, state_dim={env.state_dim}')
    print(f'[rl_env] 中纬可见星 mean={env.env.n_vis_per.mean():.1f}')
    state, info = env.reset()
    print(f'[rl_env] reset state shape={state.shape}, '
          f'first pending user={info["user"]}, '
          f'n_visible={int(info["action_mask"].sum())}')
    # 随机策略 rollout 几步
    rng = np.random.default_rng(0)
    n = 0
    total_r = 0.0
    while state is not None and n < 50:
        mask = info['action_mask']
        valid = np.where(mask)[0]
        a = int(rng.choice(valid))
        state, r, done, info = env.step(a)
        total_r += r
        n += 1
    print(f'[rl_env] 50 步随机策略: total_r={total_r:.3f}, '
          f'mean_step_r={total_r/max(n,1):.3f}')

    # ---- TemporalEnv 自检（用贪心 spatial stub）----
    print()
    print('[rl_env/Temporal] 自检...')

    def greedy_spatial_stub(snap, u, valid_k, H, tau, L, psi, cur_sat, mv):
        # 用 argmax ψ 代理 DQN（验证 TemporalEnv 流程跑通）
        a = int(np.argmax(psi))
        sat_ids = snap.visible_idx[u][valid_k]
        return a, int(sat_ids[a])

    t0 = time.time()
    tenv = TemporalEnv(
        weights=(0.4, 0.3, 0.3), spatial_policy_fn=greedy_spatial_stub,
        n_users=20, reward_case='case234', seed=SEED)
    print(f'[Temporal] 构造耗时 {time.time()-t0:.2f}s, '
          f'seq_dim={tenv.seq_dim}, ins_dim={tenv.ins_dim}, '
          f'state_flat_dim={tenv.state_flat_dim}')
    seq, ins = tenv.reset()
    if seq is not None:
        print(f'[Temporal] reset seq.shape={seq.shape}, ins.shape={ins.shape}')
        # "永远 keep" rollout（看会否掉线 / reward）
        rng = np.random.default_rng(0)
        n = 0; tot_r = 0.0; n_sw = 0
        while seq is not None and n < 200:
            # 用一个简单启发：当前 τ̃ < 0.1 或 首次接入时强制 trigger，否则 keep
            a = 1 if (ins[4] < 0.1) else 0
            seq, ins, r, done, info = tenv.step(a)
            tot_r += r; n += 1
            if info.get('switched'):
                n_sw += 1
        print(f'[Temporal] 200 步 (启发): total_r={tot_r:.2f}, '
              f'mean_r={tot_r/max(n,1):.3f}, n_switches={n_sw}, '
              f'reasons sample: 见下')
    print('[rl_env] 自检完成')
