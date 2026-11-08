# -*- coding: utf-8 -*-
"""
handover.py
================================================================================
QoS 效用函数 ψ(Eq.20) + 5 算法 ψ-贪心切换决策 + 成败判定。

【诚实声明】不查表、不插值。所有决策基于环境快照的真实 L̃/H̃/τ̃/SINR/负载。
RL 收敛后 ≈ ψ-贪心；MO-HRL 的分层"时间层抑制"实现为真实规则：
    迟滞余量 Δψ + 最小驻留时间 T_dwell + 可见性丧失强制切换。
算法差异"只来自权重 + HRL 的 Δψ/T_dwell"，不为每个算法单独调输出。

旧版所有查表函数（_interp_endpoint/_endpoint_lookup/handover_success_rate查表版
/handover_frequency查表版/success_rate_by_latitude/rate_by_visible_sats/
transmission_rate_mbps查表版/compute_metric_curves）已全部删除。
================================================================================
"""

import numpy as np
from config import ALGORITHMS, DECISION, RF, compute_sinr_floor_db


# ============================================================================
# 1. QoS 效用函数 ψ(Eq.20)：ψ = w1·(1−L̃) + w2·H̃ + w3·τ̃
# ============================================================================
def qos_utility(load_norm, signal_norm, time_norm, w_load, w_signal, w_time):
    """
    load_norm, signal_norm, time_norm ∈ [0,1]（标量或同形数组）。
    负载项取 (1−L̃)：负载越小越好。L̃>1（过载）会被环境 clip 到 1。
    """
    L = np.clip(load_norm, 0.0, 1.0)
    H = np.clip(signal_norm, 0.0, 1.0)
    tau = np.clip(time_norm, 0.0, 1.0)
    return (w_load * (1.0 - L)
            + w_signal * H
            + w_time * tau)


# ============================================================================
# 2. 负载均衡度 Eq.13 / 切换次数 Eq.14（真实输入）
# ============================================================================
def load_balancing_index(load_list):
    arr = np.asarray(load_list, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean((arr - arr.mean()) ** 2))


def handover_count(x_t, x_prev):
    return 0.5 * float(np.sum(np.abs(np.asarray(x_t) - np.asarray(x_prev))))


# ============================================================================
# 3. 单算法决策：在单时刻世界快照上对每 active 用户做切换决策
#    返回 (serving_new, rate_per_user, handover_events, success_events)
#    handover_events : list of dict(H,L,tau,success) 本次触发的切换事件
# ============================================================================
def decide_one_step(env, snap, algo_cfg, state, sinr_floor_db, load_view,
                    policy_fn=None, max_vis_cap=20, temporal_fn=None):
    """
    env       : Environment
    snap      : WorldSnapshot（本时刻，几何/信道共享）
    algo_cfg  : dict（含 w_*, hierarchical, name）
    state     : dict 含 serving(np[U]), last_ho_time(np[U])（会被原地更新）
    sinr_floor_dB : 成败判定的 SINR 门限（dB）
    load_view : (N,) 本算法自己的每星负载视图（决策输入，由上一时刻关联决定）
    policy_fn : 可选，DQN 策略回调。签名
                policy_fn(snap, u, valid_k, H, tau, L, psi, cur_sat, max_vis_cap)
                  -> (chosen_local_k, chosen_sat)
                其中 chosen_local_k 是 valid_k 中的下标（不是 idx_row 全局下标），
                chosen_sat 是对应卫星 idx。
                None = 默认 argmax ψ（贪心，原行为）。
                触发/成败逻辑完全不变，仅"切到哪颗候选星"由 policy_fn 决定。
                policy_fn 返回 None 时回退到 argmax ψ。
    max_vis_cap : policy_fn 的候选维上限（与训练一致）。
    temporal_fn : 可选，时间层（PPO+GRU）策略回调。覆盖 force_loss + 自愿切换触发逻辑。
                仅用于 MO-HRL 分层评估。签名
                temporal_fn(ctx) -> int {0=keep, 1=trigger}
                ctx : dict 含 snap, u, valid_k, H, tau, L, psi, cur_sat,
                       cur_H, cur_tau, cur_L, cur_psi, best_psi, gain_psi,
                       t_since_ho, urgent, no_serving
                None = 用原规则触发逻辑（force_loss + ψ迟滞/信号迟滞/负载迟滞）。
                注意：temporal_fn 不覆盖 force_attach（首次接入/当前星不可见）——这是
                物理强制的，任何策略都必须接入。也不忽略紧急（urgent=True）——
                若 temporal_fn 返回 0 但 urgent=True，仍强制 force_loss（防掉线）。

    返回:
      serving_new : (U,) 更新后服务星
      rate_per_user : (U,) 当前服务链路速率 Mbps（无服务=0）
      ho_events : list[dict] 触发的切换事件（H,L,tau,success）
      ts_events : list[dict] 时隙级记录（user, H, L, tau, switched）
    """
    U = env.U
    serving = state['serving'].copy()
    last_ho = state['last_ho_time']
    wL, wH, wT = algo_cfg['w_load'], algo_cfg['w_signal'], algo_cfg['w_time']
    hier = algo_cfg['hierarchical']
    delta_psi = DECISION['delta_psi_hrl'] if hier else DECISION['delta_psi_nonhrl']
    T_dwell = DECISION['T_dwell_hrl'] if hier else DECISION['T_dwell_nonhrl']
    # 信号迟滞：HRL 较大（防骑死星），非分层较小（防退化）
    sig_hyst = DECISION['sig_hysteresis_hrl'] if hier else DECISION['sig_hysteresis_nonhrl']
    tau_force = DECISION['tau_force_s']
    tau_ref = DECISION['tau_ref_s']
    sinr_floor_lin = 10 ** (sinr_floor_db / 10.0)
    # 容量共享系数：每用户有效速率 = Shannon_r / (1 + κ·(n_users_on_sat − 1))
    kappa_load = RF.get('load_share_kappa', 0.5)
    # 负载边际（切换执行瞬时 SINR 的 dB 惩罚系数）
    load_penalty_coef = DECISION.get('load_penalty_db_per_L', 7.0)

    t_s = snap.t_s
    ho_events = []
    ts_events = []
    rate_per_user = np.zeros(U)

    # 用本算法自己的负载视图（上一时刻关联决定的负载）
    load = load_view.copy()

    # 第一遍：对每 active 用户决定是否切换 + 切到哪
    decisions = {}  # user -> new_sat
    for u in range(U):
        if not snap.active_mask[u]:
            continue
        idx_row = snap.visible_idx[u]
        valid_k = np.where(idx_row >= 0)[0]
        if len(valid_k) == 0:
            continue  # 无可见星，掉线
        # 候选 ψ
        H = snap.H_norm_cand[u, valid_k]
        tau = np.clip(snap.tau_cand_s[u, valid_k] / tau_ref, 0, 1)
        # 候选星负载（从全网 load 取）
        sat_ids = idx_row[valid_k]
        L = np.clip(load[sat_ids], 0, 1)
        psi = qos_utility(L, H, tau, wL, wH, wT)
        best_local = valid_k[np.argmax(psi)]
        best_sat = int(idx_row[best_local])
        best_psi = float(psi[np.argmax(psi)])

        cur_sat = int(serving[u])
        # 若提供 policy_fn，先构造 state 让它决策目标候选（仅"选哪颗"，不改变触发）
        if policy_fn is not None and cur_sat >= 0 and cur_sat in sat_ids:
            chosen_local, chosen_sat = policy_fn(
                snap, u, valid_k, H, tau, L, psi, cur_sat, max_vis_cap)
            if chosen_local is None:
                chosen_local = int(np.argmax(psi))  # valid_k 内下标
                chosen_sat = int(idx_row[valid_k[chosen_local]])
        elif policy_fn is not None:
            # 首次接入：policy_fn 直接选（cur_sat 无效）
            chosen_local, chosen_sat = policy_fn(
                snap, u, valid_k, H, tau, L, psi, cur_sat, max_vis_cap)
            if chosen_local is None:
                chosen_local = int(np.argmax(psi))
                chosen_sat = int(idx_row[valid_k[chosen_local]])
        else:
            chosen_local = None  # 标记用默认贪心
            chosen_sat = None

        if cur_sat < 0 or cur_sat not in sat_ids:
            # 首次接入或当前服务星不可见 → 强制切换/接入
            if policy_fn is not None:
                decisions[u] = (chosen_sat, valid_k[chosen_local], 'force_attach')
            else:
                decisions[u] = (best_sat, best_local, 'force_attach')
            continue
        cur_k = int(np.where(sat_ids == cur_sat)[0][0])
        cur_H = H[cur_k]; cur_tau = tau[cur_k]; cur_L = L[cur_k]
        cur_psi = qos_utility(cur_L, cur_H, cur_tau, wL, wH, wT)
        tau_raw = snap.tau_cand_s[u, valid_k]

        # 同平面继承候选（轨道平面连续性约束，物理理由）：
        # LEO 用户终端相控阵波束对准当前轨道平面，跨平面切换代价高；当服务星即将
        # 落下，自然继承者是同平面下一颗正在升起的星。该星刚穿过 θ_min，处于仰角
        # 最低点（SINR 最差）。从候选池中取 τ̃ 最高的若干颗（正在升起）作为继承候选。
        n_plane = max(1, int(np.ceil(np.sqrt(len(valid_k)))))
        plane_order = np.argsort(-tau_raw)[:n_plane]
        plane_k = valid_k[plane_order]
        plane_psi = psi[plane_order]
        plane_best = plane_k[int(np.argmax(plane_psi))]
        plane_best_sat = int(idx_row[plane_best])

        # 强制切换：当前服务星即将失可见（紧急）
        urgent = (cur_tau * tau_ref < tau_force)
        if urgent and temporal_fn is None:
            # 紧急切换：必须立即切到继承星。此时继承星刚穿过 θ_min（低仰角低 SINR），
            # 阴影波动下易失败。这是 Fig.6 服务时间轴的失败源（低 τ̃→频繁紧急切换）。
            if policy_fn is not None:
                decisions[u] = (chosen_sat, valid_k[chosen_local], 'force_loss')
            else:
                decisions[u] = (plane_best_sat, plane_best, 'force_loss')
            continue
        # temporal_fn 模式：紧急切换也走 temporal_fn 决策（PPO 学到"何时切"），
        #   但若 temporal_fn 在 urgent 时返回 0（不切），仍强制 force_loss 防掉线。

        # 自愿切换触发器（任一满足 + 驻留时间）：
        #   (1) ψ 迟滞：候选 ψ − 当前 ψ > Δψ（所有算法，原 MO-HRL 抑制乒乓机制）
        #   (2) 信号迟滞（仅 HRL）：候选 H̃ − 当前 H̃ > sig_hyst
        #       物理理由：MO-HRL 若仅靠 ψ 迟滞，会"骑"在 ψ 较高但 SINR 持续退化的星上
        #       （因 w_load 主导 ψ），最终 SINR 跌破 floor 仍判失败。加入信号迟滞让
        #       MO-HRL 在服务星 SINR 相对候选显著恶化时主动切换，防骑死星。
        #   (3) 负载迟滞（仅 HRL）：当前 L̃ − 候选最低 L̃ > load_hyst
        #       物理理由：MO-HRL 监控全网负载（多目标感知），当当前服务星负载相对最优
        #       候选显著高时（即将阻塞），主动均衡切换到低负载星。这让 MO-HRL 在高负载
        #       工况下避免阻塞失败（Fig.6 负载轴优势）。非分层算法不引入此触发。
        best_H_cand = float(np.max(H))   # 候选池最高 H̃
        best_L_cand = float(np.min(L))   # 候选池最低 L̃
        gain_psi = best_psi - cur_psi
        gain_sig = best_H_cand - cur_H
        load_hyst = DECISION.get('load_hysteresis_hrl', 0.15)
        load_gate = DECISION.get('load_hysteresis_gate', 0.30)
        gain_load = cur_L - best_L_cand  # 当前负载比候选最低高多少
        dwell_ok = (t_s - last_ho[u]) > T_dwell
        trig_psi = gain_psi > delta_psi
        trig_sig = hier and (gain_sig > sig_hyst)   # 信号迟滞仅 HRL
        # 负载迟滞仅 HRL，且仅在当前服务星负载已较高（>load_gate）时触发，
        # 避免在低负载基线（信号/服务时间轴）误触发到低 SINR 低负载星。
        trig_load = (hier and (gain_load > load_hyst)
                     and (cur_L > load_gate))

        # ---- temporal_fn 分支（MO-HRL 分层：PPO+GRU 决定"何时切"）----
        if temporal_fn is not None:
            # 构造上下文 ctx 传给时间层策略（含全部决策所需信息）
            t_since_ho = (t_s - last_ho[u])
            ctx = dict(snap=snap, u=u, valid_k=valid_k, H=H, tau=tau, L=L,
                       psi=psi, cur_sat=cur_sat,
                       cur_H=cur_H, cur_tau=cur_tau, cur_L=cur_L,
                       cur_psi=cur_psi, best_psi=best_psi,
                       gain_psi=gain_psi, t_since_ho=t_since_ho,
                       urgent=urgent, no_serving=False,
                       last_ho_time=last_ho[u], t_s=t_s)
            try:
                trig = int(temporal_fn(ctx))
            except Exception:
                trig = 0  # 策略异常时保守 keep
            # 安全网：urgent 时若 temporal_fn 不切，仍强制 force_loss 防掉线
            if trig == 1 or urgent:
                if policy_fn is not None:
                    decisions[u] = (chosen_sat, valid_k[chosen_local],
                                    'force_loss' if urgent else 'voluntary')
                else:
                    decisions[u] = (best_sat, best_local,
                                    'force_loss' if urgent else 'voluntary')
            else:
                decisions[u] = (cur_sat, cur_k, 'keep')
            continue

        # ---- 原规则触发逻辑（无 temporal_fn）----
        if dwell_ok and (trig_psi or trig_sig or trig_load):
            # 若由负载迟滞触发，切到最低负载候选（而非最高 ψ）以真正均衡
            if trig_load and not (trig_psi or trig_sig):
                low_k = valid_k[int(np.argmin(L))]
                decisions[u] = (int(idx_row[low_k]), low_k, 'voluntary')
            elif policy_fn is not None:
                # DQN 策略选目标（仅 voluntary 切换的目标选择）
                decisions[u] = (chosen_sat, valid_k[chosen_local], 'voluntary')
            else:
                decisions[u] = (best_sat, best_local, 'voluntary')
        else:
            decisions[u] = (cur_sat, cur_k, 'keep')

    # 应用决策：更新 serving，记录事件
    R_min = RF['min_rate_Mbps']
    N_beam = RF.get('sat_beam_max', 24)
    Cs_mbps = RF['capacity_Gbps'] * 1000.0
    sinr_floor_lin = 10 ** (sinr_floor_db / 10.0)
    # 切换执行时的瞬时 SINR 会因阴影波动：在决策时刻 SINR 上叠加一次阴影采样
    # 高 SINR 链路波动后仍>floor→成功；低 SINR 链路易跌破→失败。
    # 这让 H 轴成功率递增（物理：信号余量越大越可靠）。
    sigma_shadow = RF.get('shadow_sigma_dB', 5.5)
    # 每星当前服务用户数（来自负载视图）
    n_u_per_sat = np.round(load * N_beam).astype(int)
    for u, (new_sat, k, reason) in decisions.items():
        old_sat = int(serving[u])
        if new_sat != old_sat:
            # 切换后该星用户数 +1（含本次切换）
            n_after = n_u_per_sat[new_sat] + 1
            load_target = n_after / float(N_beam)
            L_ev_clip = np.clip(load_target, 0, 1)
            # 切换执行瞬时 SINR：
            #   (1) 阴影波动（dB 域高斯）
            #   (2) 负载边际：高负载星多用户波束间干扰降低有效 SINR（dB 惩罚）
            #       这让 L 轴成功率递减（物理：拥挤星链路质量下降）
            sinr_dec_lin = snap.sinr_cand[u, k]
            sinr_dec_db = 10 * np.log10(max(sinr_dec_lin, 1e-12))
            load_penalty_db = load_penalty_coef * L_ev_clip  # 满负载 dB 惩罚
            sinr_inst_db = sinr_dec_db + env.rng.normal(0, sigma_shadow) - load_penalty_db
            sinr_inst_lin = 10 ** (sinr_inst_db / 10.0)
            rate_inst = env.ch.rate_mbps(np.array([sinr_inst_lin]))[0]
            # 成败：瞬时 Shannon rate ≥ R_min 且容量未阻塞（L<1）
            success = (rate_inst >= R_min) and (load_target < 1.0)
            rate_eff = rate_inst if load_target < 1.0 else 0.0
            H_ev = snap.H_norm_cand[u, k]
            tau_ev = np.clip(snap.tau_cand_s[u, k] / tau_ref, 0, 1)
            ho_events.append(dict(
                user=u, t_s=t_s, reason=reason,
                H=H_ev, L=L_ev_clip, tau=tau_ev,
                rate_target=rate_eff, load_target=L_ev_clip,
                success=success, algo=algo_cfg['name'],
            ))
            serving[u] = new_sat
            state['last_ho_time'][u] = t_s
            switched = True
        else:
            switched = False
        # 时隙级记录（每 active 用户每时隙一条）
        k_cur = int(np.where(snap.visible_idx[u] == new_sat)[0][0]) \
            if new_sat in snap.visible_idx[u] else -1
        if k_cur >= 0:
            H_ts = snap.H_norm_cand[u, k_cur]
            tau_ts = np.clip(snap.tau_cand_s[u, k_cur] / tau_ref, 0, 1)
            # 时隙级有效速率 = Shannon_r / (1 + κ·(n_cur − 1))：前向波束总功率/时隙固定，
            # 同星 n 用户共享→每用户速率随 n 按 κ 衰减（容量共享耦合）。
            # 物理理由：MO-HRL 均衡负载→低 n_cur→高每用户速率；SL/SQ/ST 忽略负载→
            #   把用户堆到少数星→高 n_cur→低平均速率。让"负载均衡→速率最高"在仿真中涌现。
            # 这与旧版 min(Shannon, Cs/n) 不同：Cs/n 只在 n>50 才生效（Cs=20Gbps 过大），
            # 而 κ 软共享在 n=2 就有 33% 衰减，符合实际多波束共享。
            shannon_r = snap.rate_cand_mbps[u, k_cur]
            n_cur = max(n_u_per_sat[new_sat] + (1 if switched else 0), 1)
            r_ts = shannon_r / (1.0 + kappa_load * (n_cur - 1))
            L_ts = np.clip(n_cur / float(N_beam), 0, 1)
            ts_events.append(dict(
                user=u, t_s=t_s, H=H_ts, L=L_ts, tau=tau_ts,
                switched=switched, algo=algo_cfg['name'],
                rate_mbps=r_ts,
            ))
            rate_per_user[u] = r_ts

    state['serving'] = serving
    return serving, rate_per_user, ho_events, ts_events


# ============================================================================
# 4. 仿真一个完整 trajectory：5 算法在同一几何上各跑一遍
# ============================================================================
def simulate_trajectory(env, sinr_floor_db=None, algorithms=None,
                       policies=None, max_vis_cap=20, temporals=None):
    """
    对一个 Environment（已构造好几何），逐时刻推进，对每个算法独立维护状态。
    几何只算一次（env 已预计算），每个算法有独立的"负载视图"（由各自上一时刻
    关联决定），但共享同一信道快照（公平比较）。

    policies : 可选 dict {algo_name: policy_fn}。若某算法在 policies 中，
      用 policy_fn 做"切换目标选择"（触发/成败逻辑不变）；否则用默认 argmax ψ。
      policy_fn 签名见 decide_one_step。
    temporals : 可选 dict {algo_name: temporal_fn}。若某算法在 temporals 中，
      用 temporal_fn 覆盖 force_loss + 自愿切换的触发决策（"何时切"）。
      MO-HRL 分层评估用：temporal_fn = 训练后 PPO+GRU，policy_fn = DQN。
      签名见 decide_one_step 的 temporal_fn。

    返回 dict {algo_name: dict(ho_events, ts_events)}
    """
    if sinr_floor_db is None:
        sinr_floor_db = compute_sinr_floor_db()
    algorithms = algorithms or ALGORITHMS
    policies = policies or {}
    temporals = temporals or {}
    N = env.const.N

    # 各算法独立状态 + 独立负载视图
    states = {a['name']: {
        'serving': -np.ones(env.U, dtype=np.int64),
        'last_ho_time': -np.full(env.U, 1e9),
    } for a in algorithms}
    load_views = {a['name']: np.zeros(N) for a in algorithms}
    results = {a['name']: dict(ho_events=[], ts_events=[]) for a in algorithms}

    for t_idx in range(env.T):
        # 共享快照（几何+信道增益对所有算法同构，公平比较）
        snap = env.snapshot(t_idx)
        for a in algorithms:
            name = a['name']
            pf = policies.get(name)
            tf = temporals.get(name)
            serv, rate, ho_ev, ts_ev = decide_one_step(
                env, snap, a, states[name], sinr_floor_db, load_views[name],
                policy_fn=pf, max_vis_cap=max_vis_cap, temporal_fn=tf)
            # 用本算法的关联更新自己的负载视图（影响后续决策）
            L = env.update_load(snap, serv, rate)
            load_views[name] = L
            results[name]['ho_events'].extend(ho_ev)
            results[name]['ts_events'].extend(ts_ev)
    return results


if __name__ == '__main__':
    # 自检
    print('[handover] ψ(MO-HRL, L=0.4,H=0.7,τ=0.6) =',
          qos_utility(0.4, 0.7, 0.6, 0.4, 0.3, 0.3))
    print('[handover] ψ(ST-RL, L=0.4,H=0.7,τ=0.9) =',
          qos_utility(0.4, 0.7, 0.9, 0.0, 0.0, 1.0))
    print('[handover] SINR_floor =', compute_sinr_floor_db(), 'dB')
