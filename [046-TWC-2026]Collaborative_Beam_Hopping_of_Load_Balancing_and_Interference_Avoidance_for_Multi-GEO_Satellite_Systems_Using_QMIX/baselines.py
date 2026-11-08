"""
baselines.py — 对比算法实现
包括: RBH, PBH, DFBH, GABH, SADRL-BH, MADRL-BH, LBGT-MSBH, PH-MSBH, I-MSBH, CMARL-MSBH
"""
import numpy as np
import config as cfg
from environment import GEOBeamHoppingEnv


def run_rbh(env, n_timeslots=cfg.N_TIMESLOTS):
    """RBH: 随机跳波束
    每个时隙随机选择 N_k 个波束激活
    """
    for _ in range(n_timeslots):
        for s in range(env.n_sat):
            action = np.zeros(env.n_cells)
            chosen = env.rng.choice(env.n_cells, size=env.n_beams, replace=False)
            action[chosen] = 1
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_pbh(env, period=10, n_timeslots=cfg.N_TIMESLOTS):
    """PBH: 周期跳波束
    按固定周期轮询激活波束
    """
    # 预分配轮询顺序
    for s in range(env.n_sat):
        perm = env.rng.permutation(env.n_cells)
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            action = np.zeros(env.n_cells)
            start = (t % (env.n_cells // env.n_beams + 1)) * env.n_beams
            chosen = np.arange(start, min(start + env.n_beams, env.n_cells))
            action[chosen] = 1
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_dfbh(env, n_timeslots=cfg.N_TIMESLOTS):
    """DFBH: 延迟公平跳波束
    优先服务延迟最大的 N_k 个小区
    """
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            action = np.zeros(env.n_cells)
            # 选择延迟最大的 N_k 个小区
            delay_vals = env.delay[s]
            chosen = np.argsort(-delay_vals)[:env.n_beams]
            action[chosen] = 1
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_gabh(env, n_timeslots=cfg.N_TIMESLOTS):
    """GABH: 遗传算法优化跳波束
    每时隙使用 GA 选择最优波束分配
    """
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            # 遗传算法优化
            action = _ga_optimize(env, s)
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def _ga_optimize(env, s):
    """遗传算法: 寻找最优波束分配"""
    pop_size = cfg.GA_POP_SIZE
    n_gens = cfg.GA_N_GENERATIONS
    mut_rate = cfg.GA_MUTATION_RATE
    n_cells = env.n_cells
    n_beams = env.n_beams

    # 初始化种群
    pop = []
    for _ in range(pop_size):
        ind = np.zeros(n_cells)
        chosen = env.rng.choice(n_cells, n_beams, replace=False)
        ind[chosen] = 1
        pop.append(ind)

    for gen in range(n_gens):
        # 评估适应度 (使用吞吐量 + 延迟公平)
        fitness = []
        for ind in pop:
            env.set_action(ind, s)
            # 模拟一步
            old_traffic = env.traffic[s].copy()
            # 简化: 直接用小区流量作为适应度
            fit = np.sum(env.traffic[s] * ind)
            fitness.append(fit)
        fitness = np.array(fitness)

        # 选择 (锦标赛)
        new_pop = []
        for _ in range(pop_size):
            i1, i2 = env.rng.choice(pop_size, 2, replace=False)
            if fitness[i1] > fitness[i2]:
                new_pop.append(pop[i1].copy())
            else:
                new_pop.append(pop[i2].copy())
        pop = new_pop

        # 交叉 (单点)
        for i in range(0, pop_size, 2):
            if env.rng.rand() < 0.8 and i + 1 < pop_size:
                point = env.rng.randint(1, n_cells - 1)
                temp = pop[i][point:].copy()
                pop[i][point:] = pop[i+1][point:]
                pop[i+1][point:] = temp
                # 修复: 确保每个体恰好有 n_beams 个1
                for j in range(2):
                    idx = i + j
                    if np.sum(pop[idx]) != n_beams:
                        ones = np.where(pop[idx] == 1)[0]
                        zeros = np.where(pop[idx] == 0)[0]
                        if np.sum(pop[idx]) > n_beams:
                            remove = env.rng.choice(ones, int(np.sum(pop[idx]) - n_beams), replace=False)
                            pop[idx][remove] = 0
                        else:
                            add = env.rng.choice(zeros, int(n_beams - np.sum(pop[idx])), replace=False)
                            pop[idx][add] = 1

        # 变异
        for i in range(pop_size):
            if env.rng.rand() < mut_rate:
                ones = np.where(pop[i] == 1)[0]
                zeros = np.where(pop[i] == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    add = env.rng.choice(zeros)
                    remove = env.rng.choice(ones)
                    pop[i][add] = 1
                    pop[i][remove] = 0

    # 返回最优个体
    best_idx = np.argmax(fitness)
    return pop[best_idx]


def run_sadrl_bh(env, n_timeslots=cfg.N_TIMESLOTS, agent=None):
    """SADRL-BH: 单智能体 DRL 跳波束
    每个卫星独立 DQN
    """
    from qmix_agent import SADRLAgent
    if agent is None:
        agent = SADRLAgent(env.n_cells, env.n_cells)
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            obs = env.get_obs(s)
            action = agent.select_action(obs)
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_madrl_bh(env, n_timeslots=cfg.N_TIMESLOTS, agents=None):
    """MADRL-BH: 多智能体 DRL (独立Q学习 / IQL)
    每个卫星独立 IQL
    """
    from qmix_agent import IQLAgent
    if agents is None:
        agents = [IQLAgent(env.n_cells, env.n_cells) for _ in range(env.n_sat)]
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            obs = env.get_obs(s)
            action = agents[s].select_action(obs)
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_cmsbh(env, n_timeslots=cfg.N_TIMESLOTS, qmix_agent=None):
    """CMSBH: 协作多卫星跳波束 (QMIX)
    使用 QMIX 的 CTDE 框架
    """
    from qmix_agent import QMIXAgent
    if qmix_agent is None:
        qmix_agent = QMIXAgent(env.n_sat, env.n_cells, env.n_cells,
                               env.n_sat * env.n_cells)
    for t in range(n_timeslots):
        obs_list = [env.get_obs(s) for s in range(env.n_sat)]
        actions = qmix_agent.select_actions_exploit(obs_list)
        for s in range(env.n_sat):
            env.set_action(actions[s], s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


# ========== 多星 MSBH 对比算法 (Fig.11) ==========

def run_lbgt_msbh(env, n_timeslots=cfg.N_TIMESLOTS):
    """LBGT-MSBH: 负载均衡贪心多星跳波束
    按负载排序, 优先服务高负载小区
    """
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            action = np.zeros(env.n_cells)
            # 按负载 (已服务数据量) 排序
            load_vals = env.served_data[s]
            chosen = np.argsort(-load_vals)[:env.n_beams]
            action[chosen] = 1
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_ph_msbh(env, n_timeslots=cfg.N_TIMESLOTS):
    """PH-MSBH: 比例启发式多星跳波束
    按比例分配波束给各小区
    """
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            action = np.zeros(env.n_cells)
            # 按流量需求比例分配波束
            traffic = env.traffic[s]
            traffic_sum = np.sum(traffic)
            if traffic_sum > 0:
                prob = traffic / traffic_sum
                # 确保概率和为1 (浮点误差)
                prob = prob / np.sum(prob)
                # 检查是否有足够非零概率的条目
                non_zero = np.sum(prob > 0)
                if non_zero < env.n_beams:
                    # 不足时, 随机选择
                    chosen = env.rng.choice(env.n_cells, size=env.n_beams, replace=False)
                else:
                    chosen = env.rng.choice(env.n_cells, size=env.n_beams, replace=False, p=prob)
            else:
                # 无流量需求时, 随机选择
                chosen = env.rng.choice(env.n_cells, size=env.n_beams, replace=False)
            action[chosen] = 1
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_i_msbh(env, n_timeslots=cfg.N_TIMESLOTS):
    """I-MSBH: 独立多星跳波束
    各卫星独立决策, 不考虑其他卫星
    """
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            action = np.zeros(env.n_cells)
            # 独立随机选择
            chosen = env.rng.choice(env.n_cells, size=env.n_beams, replace=False)
            action[chosen] = 1
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


def run_cmarl_msbh(env, n_timeslots=cfg.N_TIMESLOTS, agents=None):
    """CMARL-MSBH: 协作MARL多星跳波束 (VDN式)
    各智能体独立决策但共享联合奖励
    """
    from qmix_agent import IQLAgent
    if agents is None:
        agents = [IQLAgent(env.n_cells, env.n_cells) for _ in range(env.n_sat)]
    for t in range(n_timeslots):
        for s in range(env.n_sat):
            obs = env.get_obs(s)
            action = agents[s].select_action(obs)
            env.set_action(action, s)
        env.step()
    tp, delay, load = env.get_metrics()
    return tp, delay, load


# ========== 统一评估接口 ==========

ALGORITHMS_FIG9 = {
    'RBH': run_rbh,
    'PBH': run_pbh,
    'DFBH': run_dfbh,
    'GABH': run_gabh,
    'SADRL-BH': run_sadrl_bh,
    'MADRL-BH': run_madrl_bh,
    'CMSBH': run_cmsbh,
}

ALGORITHMS_FIG11 = {
    'RBH': run_rbh,
    'PBH': run_pbh,
    'LBGT-MSBH': run_lbgt_msbh,
    'PH-MSBH': run_ph_msbh,
    'I-MSBH': run_i_msbh,
    'CMARL-MSBH': run_cmarl_msbh,
    'CMSBH': run_cmsbh,
}


def evaluate_algorithm(env, algo_func, n_timeslots=cfg.N_TIMESLOTS, **kwargs):
    """评估单个算法, 返回指标"""
    tp, delay, load = algo_func(env, n_timeslots=n_timeslots, **kwargs)
    # 计算汇总指标
    total_throughput = np.sum(tp)  # Mbps
    # 延迟偏移 (Eq.19-20)
    delay_offsets = []
    for s in range(env.n_sat):
        d = delay[s]
        delay_offsets.append(np.max(d) - np.min(d))
    mean_delay_offset = np.mean(delay_offsets)
    # 负载偏移 (Eq.24-26)
    loads = np.sum(load, axis=1)
    load_offset = np.max(loads) - np.min(loads)
    # 满意度率
    total_served = np.sum(load)
    total_traffic = total_served + np.sum(env.traffic)
    satisfaction = total_served / max(total_traffic, 1e-10)
    # 归一化负载偏移
    norm_load_offset = load_offset / max(np.sum(loads), 1e-10)
    return {
        'throughput': total_throughput,
        'delay_offset': mean_delay_offset,
        'load_offset': load_offset,
        'satisfaction': satisfaction,
        'norm_load_offset': norm_load_offset,
    }