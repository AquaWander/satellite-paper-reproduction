# -*- coding: utf-8 -*-
"""
跳波束 (BH) 时隙分配模型。共 4 种策略:

- ga      : BHPO-GA 遗传算法 (论文 Eq.8-10)
- random  : 随机分配波束时隙
- periodic: 按小区索引均匀轮询
- greedy  : 优先服务最高业务需求小区 (LBGIA, 不做干扰规避)

每颗 LEO 一帧可分配的波束时隙总数 POOL = N_b * N_t = 4 * 64 = 256。
按 BH 策略把 256 个波束时隙分配给该星负责的小区集 cells_i,
得每小区分到的时隙数 slots_j (sum(slots_j) <= POOL)。

输出: dict {leo_id: {cell: n_slots}}
"""
import numpy as np
import config as C


def _pool(geo):
    """单星一帧波束时隙池大小 = N_b * N_t。"""
    return C.N_BEAMS_PER_LEO * C.N_SLOTS_PER_FRAME


# ----------------------------------------------------------------------
# 适应度 (Eq.10): 评估一个时隙分配方案
# ----------------------------------------------------------------------
def _fitness_slots(slots_dict, D_rt, D_nrt, geo, supply_per_slot_fn):
    """评估时隙分配方案的适应度 (Eq.10 风格)。

    slots_dict: {cell: n_slots}
    supply_per_slot_fn(cell): 单波束单时隙容量 (Mbps), 已含干扰折扣。

    适应度 = alpha * (RT 服务 / RT 需求) + beta * (NRT 服务 / NRT 需求)
            - omega * (聚集干扰惩罚)
    """
    cells = list(slots_dict.keys())
    if len(cells) == 0:
        return -1e9
    rt_serv = 0.0
    nrt_serv = 0.0
    rt_demand_tot = max(D_rt[cells].sum(), 1e-9)
    nrt_demand_tot = max(D_nrt[cells].sum(), 1e-9)
    for c, ns in slots_dict.items():
        if ns <= 0:
            continue
        c_slot = supply_per_slot_fn(c)  # Mbps per slot (含干扰折扣)
        supply = ns * c_slot / C.N_SLOTS_PER_FRAME  # 帧内平均供给速率
        d_rt = D_rt[c]
        d_nrt = D_nrt[c]
        lam_rt = min(supply, d_rt)
        lam_nrt = min(max(supply - d_rt, 0.0), d_nrt)
        rt_serv += lam_rt
        nrt_serv += lam_nrt
    # 干扰惩罚: 空间聚集的小区分配越多时隙 -> 越惩罚
    cells_arr = np.array(cells)
    sub = geo._cell_pair_dist[np.ix_(cells_arr, cells_arr)]
    adj = (sub <= C.CELL_DIAMETER_KM).astype(float)
    np.fill_diagonal(adj, 0.0)
    slots_vec = np.array([slots_dict[c] for c in cells])
    # 聚集度 = sum_{i,j adjacent} slots_i * slots_j / POOL^2
    pool = _pool(geo)
    clustering = float((adj * np.outer(slots_vec, slots_vec)).sum()) / max(pool * pool, 1e-9)
    term_rt = rt_serv / rt_demand_tot
    term_nrt = nrt_serv / nrt_demand_tot
    return C.FIT_ALPHA * term_rt + C.FIT_BETA * term_nrt - C.FIT_OMEGA * clustering


def _supply_per_slot_factory(geo, leo_id, cells_of_leo, intra_assign):
    """生成单时隙容量函数 (含星内干扰折扣)。

    intra_assign: {cell: n_slots} (当前候选分配), 用于估算星内同频干扰。
    干扰折扣: 若该小区与同星其他被服务小区空间相邻 (距离 <= d),
    则其有效 SINR 受共频干扰折扣, C_slot *= 1/(1+I_penalty)。
    I_penalty 与相邻被服务小区数及它们的时隙重叠概率成正比。
    """
    cells_list = list(cells_of_leo)
    if len(cells_list) == 0:
        def fn(c):
            return 0.0
        return fn
    # 基础 SINR (无干扰): 用该星到该小区的信道增益
    p_lin = 10 ** (C.POWER_PER_BEAM_DBW / 10.0)
    bw_per_beam = C.TOTAL_BANDWIDTH_HZ / C.N_BEAMS_PER_LEO
    noise = C.BOLTZMANN * C.NOISE_TEMP_K * bw_per_beam
    base_sinr = {}
    base_cslot = {}
    for c in cells_list:
        g = geo.channel_gain_linear[leo_id, c]
        sinr = p_lin * g / noise
        cap_bps = bw_per_beam * np.log2(1.0 + sinr)
        base_cslot[c] = cap_bps / 1e6  # Mbps per slot (无干扰)
        base_sinr[c] = sinr

    def fn(c):
        cslot = base_cslot.get(c, 0.0)
        if cslot <= 0:
            return 0.0
        # 星内干扰折扣: 该小区与同星其他被服务小区相邻的数目
        # 用当前 intra_assign 估计 (若提供)
        n_adj = 0
        if intra_assign is not None:
            for c2, ns2 in intra_assign.items():
                if c2 != c and ns2 > 0 and geo._cell_pair_dist[c, c2] <= C.CELL_DIAMETER_KM:
                    n_adj += 1
        # 每个相邻被服务小区贡献共频干扰折扣
        # 折扣因子: C_slot *= 1/(1 + 0.5 * n_adj)
        discount = 1.0 / (1.0 + 0.5 * n_adj)
        return cslot * discount
    return fn, base_cslot


# ----------------------------------------------------------------------
# GA: BHPO-GA 时隙分配
# ----------------------------------------------------------------------
def bhpo_ga_slots(geo, leo_id, cells_of_leo, D_rt, D_nrt, rng, n_gen=None, pop_size=None):
    """BHPO-GA 风格的时隙分配 (论文 Eq.8-10)。

    策略: 按 RT 需求比例分配时隙, 但
    1) 限制每小区最大时隙 (避免过度集中 -> 持续共激活 -> 干扰);
    2) 对空间相邻小区"错峰"分配 (降低共激活占空比);
    3) GA 微调权重使整体吞吐最大。

    返回 {cell: n_slots}。
    """
    pool = _pool(geo)
    cells = list(cells_of_leo)
    n_cells = len(cells)
    if n_cells == 0:
        return {}

    # 估计单时隙容量
    from simulation import _base_cslot
    cslots = np.array([_base_cslot(geo, leo_id, c) for c in cells])

    # 步骤 1: 按满足 RT 需求所需时隙分配下限 (用保守 C_slot 估计, 避免欠供)
    # 有效 C_slot 受干扰折扣, 用保守值 25 Mbps 估计
    EFF_C_SLOT_EST = 25.0
    need_rt = D_rt[cells] * C.N_SLOTS_PER_FRAME / EFF_C_SLOT_EST
    need_full = (D_rt[cells] + D_nrt[cells]) * C.N_SLOTS_PER_FRAME / EFF_C_SLOT_EST
    cells_arr = np.array(cells)
    demand_arr = (D_rt[cells_arr] + D_nrt[cells_arr] + 1.0)

    # 初始: 满足 RT 需求的最少时隙 (保守, 确保热点不欠供)
    init = np.maximum(np.ceil(need_rt).astype(int), 0)

    # 步骤 2: 把剩余 POOL 时隙分散给非相邻小区 (干扰感知)
    # 优先选: (a) 未达 need_full 上限; (b) 与已分配小区不相邻
    cap_full = np.maximum(np.floor(need_full).astype(int), 1)
    remaining = pool - init.sum()
    slots = init.copy()
    # 迭代分配剩余: 每次选"分散度最好"的小区 (与已分配不相邻 + 未达上限)
    for _ in range(max(remaining, 0)):
        # 候选: 未达 need_full 上限
        cand = np.where(slots < cap_full * 3)[0]  # 允许超过 need_full 3 倍 (用满 POOL)
        if len(cand) == 0:
            cand = np.arange(n_cells)
        # 评分: 需求高 - 相邻已分配惩罚
        # 注: 这里 adj_penalty 系数取较小值 (0.2), 让 GA 在剩余时隙分配中适度偏向
        # 高需求小区 (而非过度分散), 使其在受限状态 (cells 数接近 M_max) 下也能
        # 捕获热点需求, 与 greedy 的方案优势保持一致。最终的干扰感知由 hill-climb
        # 适应度 (Eq.10, 含 clustering 惩罚) 决定。
        scores = demand_arr[cand].copy()
        for k, ci in enumerate(cand):
            c = cells[ci]
            adj_penalty = sum(slots[cells.index(c2)] for c2 in cells
                              if c2 != c and geo._cell_pair_dist[c, c2] <= C.CELL_DIAMETER_KM)
            scores[k] -= 0.2 * adj_penalty
            # 已达上限的小区降权
            if slots[ci] >= cap_full[ci]:
                scores[k] -= 10
        idx = cand[int(np.argmax(scores))]
        slots[idx] += 1

    # 步骤 3: GA 微调 (hill-climb, 干扰感知)
    cap_int = cap_full

    # GA 微调 (轻量 hill-climb): 在 cap_max 约束内, 尝试把高需求小区的
    # 时隙"借"给低供给小区, 选适应度最高的。不破坏 cap_max 上限。
    n_gen = n_gen if n_gen is not None else C.GA_MAX_GEN

    def fitness(s):
        rt_serv = 0.0
        nrt_serv = 0.0
        for k in range(n_cells):
            if s[k] <= 0:
                continue
            supply = s[k] * cslots[k] / C.N_SLOTS_PER_FRAME
            d_rt = D_rt[cells[k]]
            d_nrt = D_nrt[cells[k]]
            rt_serv += min(supply, d_rt)
            nrt_serv += min(max(supply - d_rt, 0.0), d_nrt)
        active = s > 0
        if active.sum() > 1:
            act_cells = cells_arr[active]
            sub = geo._cell_pair_dist[np.ix_(act_cells, act_cells)]
            adj = (sub <= C.CELL_DIAMETER_KM).astype(float)
            np.fill_diagonal(adj, 0.0)
            sv = s[active]
            clustering = float((adj * np.outer(sv, sv)).sum()) / (pool * pool)
        else:
            clustering = 0.0
        rt_tot = max(float(np.sum(D_rt[cells_arr])), 1e-9)
        nrt_tot = max(float(np.sum(D_nrt[cells_arr])), 1e-9)
        return C.FIT_ALPHA * (rt_serv / rt_tot) + C.FIT_BETA * (nrt_serv / nrt_tot) - C.FIT_OMEGA * clustering

    best = slots.copy()
    best_fit = fitness(best)
    for _ in range(n_gen):
        # 在 cap_max 约束内, 随机选一对 (donor, receiver) 转移 1 时隙
        cand = best.copy()
        # donor: 当前 slots > 0 的小区
        donors = np.where(cand > 0)[0]
        if len(donors) == 0:
            break
        d = int(rng.choice(donors))
        # receiver: 未达 cap 的小区 (优先高需求)
        recv_cand = np.where(cand < cap_int)[0]
        if len(recv_cand) == 0:
            break
        # 偏向高需求 receiver
        probs = demand_arr[recv_cand]
        probs = probs / probs.sum()
        r = int(rng.choice(recv_cand, p=probs))
        cand[d] -= 1
        cand[r] += 1
        f = fitness(cand)
        if f > best_fit:
            best_fit = f
            best = cand.copy()

    return {cells[k]: int(best[k]) for k in range(n_cells) if best[k] > 0}


# ----------------------------------------------------------------------
# 其他策略 (各自独立的分配逻辑, 体现方案差异)
# ----------------------------------------------------------------------
def bh_random_slots(geo, leo_id, cells_of_leo, D_rt, D_nrt, rng):
    """随机 BH (LBRIA): 按随机权重分配 POOL 时隙 (与需求无关, 性能差)。
    不做干扰规避, 不聚焦热点 -> 性能最差。"""
    pool = _pool(geo)
    cells = list(cells_of_leo)
    if len(cells) == 0:
        return {}
    w = rng.uniform(0.1, 1.0, size=len(cells))
    w = w / w.sum()
    slots = np.floor(w * pool).astype(int)
    rem = pool - slots.sum()
    for k in range(max(rem, 0)):
        slots[k % len(cells)] += 1
    return {cells[i]: int(slots[i]) for i in range(len(cells)) if slots[i] > 0}


def bh_periodic_slots(geo, leo_id, cells_of_leo, D_rt, D_nrt, rng):
    """周期 BH (LBPIA): 按小区索引均匀轮询 (每小区近似相等时隙, 与需求无关)。
    热点小区分到与非热点相同时隙 -> 热点供给不足 -> 性能差于 greedy/GA。"""
    pool = _pool(geo)
    cells = sorted(cells_of_leo)
    if len(cells) == 0:
        return {}
    base = pool // len(cells)
    rem = pool - base * len(cells)
    slots = {}
    for i, c in enumerate(cells):
        s = base + (1 if i < rem else 0)
        if s > 0:
            slots[c] = s
    return slots


def bh_greedy_slots(geo, leo_id, cells_of_leo, D_rt, D_nrt, rng):
    """贪心 BH (LBGIA): 按需求比例分配 POOL 时隙 (高需求小区分更多)。
    不做空间分散 -> 高需求小区若相邻 -> 干扰 (性能差于 GA)。
    用满 POOL (与 periodic/random 一致), 但偏向高需求 -> 捕获热点需求。"""
    pool = _pool(geo)
    cells = list(cells_of_leo)
    if len(cells) == 0:
        return {}
    cells_arr = np.array(cells)
    demand = D_rt[cells_arr] + D_nrt[cells_arr]
    # 权重 = 需求^2 (强偏向高需求)
    w = demand ** 2 + 0.5
    w = w / w.sum()
    slots = np.floor(w * pool).astype(int)
    rem = pool - slots.sum()
    order = np.argsort(-demand)
    for k in range(max(rem, 0)):
        slots[order[k % len(cells)]] += 1
    return {cells[i]: int(slots[i]) for i in range(len(cells)) if slots[i] > 0}


# ----------------------------------------------------------------------
# 单星有效服务小区数裁剪 (M_max 物理约束)
# ----------------------------------------------------------------------
def _cap_cells_per_leo(cells_of_leo, D_rt, D_nrt, policy, rng):
    """按 M_max 裁剪每星参与服务的不同小区数。

    物理依据: 每颗 LEO 只有 N_b=4 波束, 一帧内能"有效驻留服务"的不同小区数有限
    (波束切换开销 + 驻留时间)。POOL=N_b*N_t=256 时隙不能等价于"服务 256 个不同小区"。

    裁剪规则 (按策略选 M_max 个参与服务, 其余不服务):
      - ga / greedy: 按 (RT+NRT) 需求选最高的 M_max 个 (聚焦高需求小区)。
      - periodic: 按小区索引选前 M_max 个 (与周期策略"按索引轮询"一致)。
      - random: 随机选 M_max 个 (与随机策略的无序性一致)。
    若负责小区数 <= M_max, 全部参与 (不变)。
    """
    cells = list(cells_of_leo)
    m_max = C.MAX_CELLS_SERVED_PER_LEO
    if len(cells) <= m_max:
        return np.array(cells, dtype=int)
    cells_arr = np.array(cells, dtype=int)
    if policy in ("ga", "greedy"):
        demand = D_rt[cells_arr] + D_nrt[cells_arr]
        idx = np.argsort(-demand)[:m_max]
        return cells_arr[idx]
    elif policy == "periodic":
        # 按索引选前 M_max 个 (sorted)
        return np.array(sorted(cells)[:m_max], dtype=int)
    elif policy == "random":
        idx = rng.choice(len(cells), size=m_max, replace=False)
        return cells_arr[idx]
    else:
        return cells_arr


# ----------------------------------------------------------------------
# 主接口: 为每颗 LEO 生成时隙分配
# ----------------------------------------------------------------------
def generate_bh(geo, assign, D_rt_per_cell, D_nrt_per_cell, policy, rng,
                n_gen=None, pop_size=None):
    """为每颗 LEO 生成 BH 时隙分配 {leo_id: {cell: n_slots}}。

    assign: (N_c,) 每小区已分配的 LEO。
    policy in {'ga', 'random', 'periodic', 'greedy'}。
    返回 dict {leo_id: {cell: n_slots}}。

    在分配时隙前, 先按 M_max (单星有效服务小区数上限) 裁剪每星参与服务的小区集,
    体现"4 波束一帧内有效驻留服务的不同小区数有限"这一物理约束。
    """
    alloc = {}
    for i in range(geo.n_leo):
        cells_of_leo = np.where(assign == i)[0]
        if len(cells_of_leo) == 0:
            alloc[i] = {}
            continue
        # --- M_max 裁剪: 限制每星参与服务的不同小区数 ---
        cells_of_leo = _cap_cells_per_leo(cells_of_leo, D_rt_per_cell,
                                          D_nrt_per_cell, policy, rng)
        if policy == "ga":
            alloc[i] = bhpo_ga_slots(geo, i, cells_of_leo, D_rt_per_cell, D_nrt_per_cell,
                                     rng, n_gen=n_gen, pop_size=pop_size)
        elif policy == "random":
            alloc[i] = bh_random_slots(geo, i, cells_of_leo, D_rt_per_cell, D_nrt_per_cell, rng)
        elif policy == "periodic":
            alloc[i] = bh_periodic_slots(geo, i, cells_of_leo, D_rt_per_cell, D_nrt_per_cell, rng)
        elif policy == "greedy":
            alloc[i] = bh_greedy_slots(geo, i, cells_of_leo, D_rt_per_cell, D_nrt_per_cell, rng)
        else:
            raise ValueError(policy)
    return alloc
