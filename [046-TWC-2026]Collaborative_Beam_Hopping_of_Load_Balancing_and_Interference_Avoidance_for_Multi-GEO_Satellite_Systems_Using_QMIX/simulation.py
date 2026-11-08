"""
simulation.py — 主仿真逻辑
训练 QMIX 一次, 在所有流量点评估
"""
import numpy as np
import os
import pickle
import time
import config as cfg
from environment import GEOBeamHoppingEnv
from qmix_agent import QMIXAgent, SADRLAgent, IQLAgent
from baselines import (
    ALGORITHMS_FIG9, ALGORITHMS_FIG11,
    evaluate_algorithm, run_cmsbh, run_sadrl_bh, run_madrl_bh, run_cmarl_msbh
)


def train_qmix(env, n_epochs=cfg.QMIX_N_EPOCHS, n_steps=cfg.QMIX_N_STEPS_PER_EPOCH,
               verbose=True):
    """训练 QMIX 智能体 (一次训练, 多流量点评估)"""
    qmix = QMIXAgent(env.n_sat, env.n_cells, env.n_cells,
                     env.n_sat * env.n_cells,
                     hidden_dim=cfg.QMIX_GRU_HIDDEN_DIM)
    losses = []
    rewards_history = []

    for epoch in range(n_epochs):
        env.reset()
        # 缩放流量到评估范围 (使训练与评估一致)
        scale_to = np.random.uniform(cfg.TRAFFIC_RANGE[0], cfg.TRAFFIC_RANGE[1])
        total_t = np.sum(env.traffic)
        if total_t > 0:
            env.traffic = env.traffic * (scale_to / total_t)
        total_reward = 0
        epoch_loss = 0
        n_updates = 0

        for step in range(n_steps):
            obs_list = [env.get_obs(s) for s in range(env.n_sat)]
            state = env.get_state().flatten()

            # 选择动作
            actions = qmix.select_actions(obs_list)

            # 记录步前服务量
            prev_served = env.served_data.copy()

            # 执行动作
            for s in range(env.n_sat):
                env.set_action(actions[s], s)
            env.step()

            # 奖励 = 归一化吞吐量 (正激励为主)
            step_tp_mbps = np.sum(env.served_data - prev_served)  # 本步新增(Mbps)
            traffic_total = max(np.sum(env.traffic + prev_served), 1)
            # 吞吐量占需求比例作为主要奖励
            reward = step_tp_mbps / 1000.0

            next_obs_list = [env.get_obs(s) for s in range(env.n_sat)]
            next_state = env.get_state().flatten()
            done = (step == n_steps - 1)

            qmix.store_experience(obs_list, actions, reward, next_obs_list,
                                  done, state, next_state)
            loss = qmix.train_step()
            if loss is not None:
                epoch_loss += loss
                n_updates += 1
            qmix.update_epsilon()
            total_reward += reward

        avg_loss = epoch_loss / max(n_updates, 1)
        losses.append(avg_loss)
        rewards_history.append(total_reward)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, "
                  f"Reward: {total_reward:.2f}, Eps: {qmix.epsilon:.3f}")

    return qmix, losses, rewards_history


def train_sadrl(env, n_epochs=cfg.QMIX_N_EPOCHS, n_steps=cfg.QMIX_N_STEPS_PER_EPOCH,
                verbose=True):
    """训练 SADRL 智能体"""
    agents = [SADRLAgent(env.n_cells, env.n_cells) for _ in range(env.n_sat)]
    for epoch in range(n_epochs):
        env.reset()
        # 缩放流量到评估范围
        scale_to = np.random.uniform(cfg.TRAFFIC_RANGE[0], cfg.TRAFFIC_RANGE[1])
        total_t = np.sum(env.traffic)
        if total_t > 0:
            env.traffic = env.traffic * (scale_to / total_t)
        for step in range(n_steps):
            prev_served = env.served_data.copy()
            actions = []
            for s in range(env.n_sat):
                obs = env.get_obs(s)
                action = agents[s].select_action(obs)
                env.set_action(action, s)
                actions.append(action)
            env.step()
            for s in range(env.n_sat):
                obs = env.get_obs(s)
                # 本步新增吞吐量
                step_tp_s = np.sum(env.served_data[s] - prev_served[s]) / 1000.0
                reward = step_tp_s
                next_obs = env.get_obs(s)
                done = (step == n_steps - 1)
                agents[s].update(obs, actions[s], reward, next_obs, done)
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  SADRL Epoch {epoch+1}/{n_epochs}")
    return agents


def train_madrl(env, n_epochs=cfg.QMIX_N_EPOCHS, n_steps=cfg.QMIX_N_STEPS_PER_EPOCH,
                verbose=True):
    """训练 MADRL-BH (IQL)"""
    agents = [IQLAgent(env.n_cells, env.n_cells) for _ in range(env.n_sat)]
    for epoch in range(n_epochs):
        env.reset()
        scale_to = np.random.uniform(cfg.TRAFFIC_RANGE[0], cfg.TRAFFIC_RANGE[1])
        total_t = np.sum(env.traffic)
        if total_t > 0:
            env.traffic = env.traffic * (scale_to / total_t)
        for step in range(n_steps):
            prev_served = env.served_data.copy()
            actions = []
            for s in range(env.n_sat):
                obs = env.get_obs(s)
                action = agents[s].select_action(obs)
                env.set_action(action, s)
                actions.append(action)
            env.step()
            for s in range(env.n_sat):
                obs = env.get_obs(s)
                step_tp = np.sum(env.served_data[s] - prev_served[s]) / 1000.0
                reward = step_tp
                next_obs = env.get_obs(s)
                done = (step == n_steps - 1)
                agents[s].update(obs, actions[s], reward, next_obs, done)
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  MADRL Epoch {epoch+1}/{n_epochs}")
    return agents


def train_cmarl(env, n_epochs=cfg.QMIX_N_EPOCHS, n_steps=cfg.QMIX_N_STEPS_PER_EPOCH,
                verbose=True):
    """训练 CMARL-MSBH"""
    agents = [IQLAgent(env.n_cells, env.n_cells) for _ in range(env.n_sat)]
    for epoch in range(n_epochs):
        env.reset()
        scale_to = np.random.uniform(cfg.TRAFFIC_RANGE[0], cfg.TRAFFIC_RANGE[1])
        total_t = np.sum(env.traffic)
        if total_t > 0:
            env.traffic = env.traffic * (scale_to / total_t)
        for step in range(n_steps):
            prev_served = env.served_data.copy()
            actions = []
            for s in range(env.n_sat):
                obs = env.get_obs(s)
                action = agents[s].select_action(obs)
                env.set_action(action, s)
                actions.append(action)
            env.step()
            for s in range(env.n_sat):
                obs = env.get_obs(s)
                step_tp = np.sum(env.served_data - prev_served) / 1000.0
                reward = step_tp / env.n_sat
                next_obs = env.get_obs(s)
                done = (step == n_steps - 1)
                agents[s].update(obs, actions[s], reward, next_obs, done)
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  CMARL Epoch {epoch+1}/{n_epochs}")
    return agents


def evaluate_algo_at_traffic(env, algo_fn, traffic_demand, n_timeslots=cfg.N_TIMESLOTS,
                             extra_args=None):
    """在指定流量需求下评估算法"""
    per_cell = traffic_demand / (env.n_sat * env.n_cells)
    env.reset(mean_base=per_cell)
    # 缩放流量到目标值
    total = np.sum(env.traffic)
    if total > 0:
        env.traffic = env.traffic * (traffic_demand / total)

    extra_args = extra_args or {}
    # 重置环境状态
    env.served_data = np.zeros((env.n_sat, env.n_cells))
    env.delay = np.zeros((env.n_sat, env.n_cells))
    env.t = 0

    result = evaluate_algorithm(env, algo_fn, n_timeslots=n_timeslots, **extra_args)
    return result


def run_full_simulation(verbose=True, use_cached=False):
    """运行完整仿真 (训练一次, 评估所有流量点)"""
    traffic_demands = np.linspace(cfg.TRAFFIC_RANGE[0], cfg.TRAFFIC_RANGE[1],
                                  cfg.N_TRAFFIC_POINTS)

    print(f"\n{'#'*60}")
    print(f"流量点: {len(traffic_demands)}, 范围: {cfg.TRAFFIC_RANGE[0]}-{cfg.TRAFFIC_RANGE[1]} Mbps")
    print(f"每点运行: {cfg.N_RUNS} 次")
    print(f"{'#'*60}\n")

    if use_cached:
        try:
            fig9 = load_results('fig9_results.pkl')
            fig11 = load_results('fig11_results.pkl')
            print("使用缓存结果")
            return fig9, fig11, traffic_demands
        except FileNotFoundError:
            print("未找到缓存, 重新运行")

    # ========== 训练阶段 ==========
    print("\n" + "="*60)
    print("训练阶段: 训练 QMIX, SADRL, MADRL, CMARL")
    print("="*60)

    train_env = GEOBeamHoppingEnv(seed=42)
    train_env.reset(mean_base=250.0)

    # 训练 QMIX
    print("\n训练 QMIX (CMSBH)...")
    t0 = time.time()
    cmsbh_agent, qmix_losses, _ = train_qmix(train_env, verbose=verbose)
    print(f"  QMIX 训练完成, 耗时 {(time.time()-t0)/60:.1f} min")

    # 训练 SADRL
    print("\n训练 SADRL-BH...")
    t0 = time.time()
    sadrl_agents = train_sadrl(train_env, verbose=verbose)
    print(f"  SADRL 训练完成, 耗时 {(time.time()-t0)/60:.1f} min")

    # 训练 MADRL
    print("\n训练 MADRL-BH...")
    t0 = time.time()
    madrl_agents = train_madrl(train_env, verbose=verbose)
    print(f"  MADRL 训练完成, 耗时 {(time.time()-t0)/60:.1f} min")

    # 训练 CMARL
    print("\n训练 CMARL-MSBH...")
    t0 = time.time()
    cmarl_agents = train_cmarl(train_env, verbose=verbose)
    print(f"  CMARL 训练完成, 耗时 {(time.time()-t0)/60:.1f} min")

    # ========== Fig.9-10 评估 ==========
    print("\n" + "="*60)
    print("Fig.9-10 评估: 单星 BH 对比")
    print("="*60)

    fig9_results = {name: {'throughput': [], 'delay_offset': []}
                    for name in ALGORITHMS_FIG9}

    for name in ALGORITHMS_FIG9:
        print(f"\n  {name}:")
        for demand in traffic_demands:
            run_results = []
            for run in range(cfg.N_RUNS):
                env_i = GEOBeamHoppingEnv(seed=int(demand * 10 + run))

                if name == 'CMSBH':
                    extra = {'qmix_agent': cmsbh_agent}
                    res = evaluate_algo_at_traffic(
                        env_i, run_cmsbh, demand, extra_args=extra)
                elif name == 'SADRL-BH':
                    # 使用训练好的 SADRL 评估
                    extra = {'agent': sadrl_agents[0]}
                    res = evaluate_algo_at_traffic(
                        env_i, run_sadrl_bh, demand, extra_args=extra)
                elif name == 'MADRL-BH':
                    extra = {'agents': madrl_agents}
                    res = evaluate_algo_at_traffic(
                        env_i, run_madrl_bh, demand, extra_args=extra)
                else:
                    fn = ALGORITHMS_FIG9[name]
                    res = evaluate_algo_at_traffic(env_i, fn, demand)

                run_results.append(res)

            avg_tp = np.mean([r['throughput'] for r in run_results])
            avg_delay = np.mean([r['delay_offset'] for r in run_results])
            fig9_results[name]['throughput'].append(avg_tp)
            fig9_results[name]['delay_offset'].append(avg_delay)

            if verbose:
                print(f"    {demand:.0f} Mbps: TP={avg_tp:.0f}, D={avg_delay:.1f}")

    # ========== Fig.11 评估 ==========
    print("\n" + "="*60)
    print("Fig.11 评估: 多星 MSBH 对比")
    print("="*60)

    fig11_results = {name: {'satisfaction': [], 'norm_load_offset': []}
                     for name in ALGORITHMS_FIG11}

    for name in ALGORITHMS_FIG11:
        print(f"\n  {name}:")
        for demand in traffic_demands:
            run_results = []
            for run in range(cfg.N_RUNS):
                env_i = GEOBeamHoppingEnv(seed=int(demand * 10 + run + 1000))

                if name == 'CMSBH':
                    extra = {'qmix_agent': cmsbh_agent}
                    res = evaluate_algo_at_traffic(
                        env_i, run_cmsbh, demand, extra_args=extra)
                elif name == 'CMARL-MSBH':
                    extra = {'agents': cmarl_agents}
                    res = evaluate_algo_at_traffic(
                        env_i, run_cmarl_msbh, demand, extra_args=extra)
                else:
                    fn = ALGORITHMS_FIG11[name]
                    res = evaluate_algo_at_traffic(env_i, fn, demand)

                run_results.append(res)

            avg_sat = np.mean([r['satisfaction'] for r in run_results])
            avg_load = np.mean([r['norm_load_offset'] for r in run_results])
            fig11_results[name]['satisfaction'].append(avg_sat)
            fig11_results[name]['norm_load_offset'].append(avg_load)

            if verbose:
                print(f"    {demand:.0f} Mbps: Sat={avg_sat:.3f}, Load={avg_load:.3f}")

    # 保存结果
    save_results(fig9_results, 'fig9_results.pkl')
    save_results(fig11_results, 'fig11_results.pkl')

    # 保存损失
    save_results({'qmix_losses': qmix_losses}, 'training_losses.pkl')

    return fig9_results, fig11_results, traffic_demands


def save_results(results, filename):
    out_dir = 'C:/Users/windows/Desktop/文章复现/[046-TWC-2026]Collaborative_Beam_Hopping_of_Load_Balancing_and_Interference_Avoidance_for_Multi-GEO_Satellite_Systems_Using_QMIX/output'
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"结果已保存: {filepath}")


def load_results(filename):
    out_dir = 'C:/Users/windows/Desktop/文章复现/[046-TWC-2026]Collaborative_Beam_Hopping_of_Load_Balancing_and_Interference_Avoidance_for_Multi-GEO_Satellite_Systems_Using_QMIX/output'
    filepath = os.path.join(out_dir, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)