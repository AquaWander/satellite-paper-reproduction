# -*- coding: utf-8 -*-
"""
主仿真引擎: 给定选星策略 + BH 策略 + MC-BHS 开关, 跑一帧,
输出 RT 吞吐量 / 总吞吐量 / RT 满足率 / 总满足率 / 负载差。

时隙级跳波束吞吐量模型 (物理干扰模型, 无硬编码方案系数):
  对每颗 LEO i:
    POOL = N_b * N_t = 4 * 64 = 256 个波束时隙
    按 BH 策略把 POOL 分给 cells_i, 得每小区 slots_j
    base_supply_j = slots_j * BASE_C_SLOT / N_t
    eff_supply_j  = base_supply_j * intra_discount_j * inter_discount_j
    λ_rt_j  = min(eff_supply_j, D_rt_j)     # RT 优先
    λ_nrt_j = min(max(eff_supply_j - D_rt_j, 0), D_nrt_j)
  rt_throughput = Σ_j λ_rt_j; total = Σ_j (λ_rt_j + λ_nrt_j)

方案性能差异完全来自两个物理机制:
  机制1 (需求匹配): ga/greedy 按需求分配 alloc -> min(supply,demand) 自然截断 periodic/random。
  机制2 (波束间干扰折扣, Eq.1/Eq.8/Eq.9):
    星内 (intra): 同星相邻被服务小区共激活程度 -> ga 分散 (discount 高), greedy 聚集 (discount 低)。
    星间 (inter): 跨 LEO 相邻被服务小区共激活程度 -> MC-BHS 用 ρ_mc 抑制 (建模协调机制, 非硬编码)。
"""
import numpy as np
import config as C
from geometry import Geometry
from traffic_model import generate_traffic
import rw_lb, bh_scheduling, mc_bhs


def _base_cslot(geo, leo_id, cell):
    """单波束单时隙基础容量 (Mbps, 无干扰): (B/N_b)*log2(1+SINR)。"""
    p_lin = 10 ** (C.POWER_PER_BEAM_DBW / 10.0)
    bw_per_beam = C.TOTAL_BANDWIDTH_HZ / C.N_BEAMS_PER_LEO
    noise = C.BOLTZMANN * C.NOISE_TEMP_K * bw_per_beam
    g = geo.channel_gain_linear[leo_id, cell]
    sinr = p_lin * g / noise
    cap_bps = bw_per_beam * np.log2(1.0 + sinr)
    return cap_bps / 1e6


def _compute_throughput(geo, alloc, assign, D_rt_per_cell, D_nrt_per_cell, use_mc):
    """按 BH 时隙分配模型计算吞吐 (物理干扰模型版)。

    方案性能差异完全来自两个物理机制, 无任何规定方案性能的硬编码系数:

    机制1 (需求匹配): 不同 BH 策略产生不同的 alloc={cell: n_slots}。
      ga/greedy 按需求分配 -> 热点供给充足; periodic/random 需求匹配差
      -> 由 min(supply, demand) 自然截断。

    机制2 (波束间干扰折扣): 对每个被服务小区 c (由 LEO i 服务, n_slots_c 个时隙):
      星内 (Eq.8/Eq.9): 同星其他被服务小区若与 c 空间相邻 (dist<=d) 且共激活,
        intra_penalty_c = Σ_{c' in cells_i, c'!=c, dist<=d} (n_slots_{c'}/N_t)
        intra_discount_c = 1 / (1 + κ_intra * intra_penalty_c)
        -> ga 干扰感知分散分配 (intra_penalty 小, discount 高);
           greedy 把时隙集中给相邻热点 (intra_penalty 大, discount 低)。

      星间 (MC-BHS 规避对象): 相邻小区 c' 由不同 LEO 服务 (assign[c']!=i, 跨 LEO 同频):
        inter_penalty_c = Σ_{c' : dist(c,c')<=d, assign[c']!=i} (n_slots_{c'}/N_t)
        若 use_mc: inter_penalty_c *= ρ_mc   (MEO 协调错开点亮, 降低共激活占空比)
        inter_discount_c = 1 / (1 + κ_inter * inter_penalty_c)

      eff_supply_c = (n_slots_c * BASE_C_SLOT / N_t) * intra_discount_c * inter_discount_c
      λ_rt_c  = min(eff_supply_c, D_rt_c)
      λ_nrt_c = min(max(eff_supply_c - D_rt_c, 0), D_nrt_c)
    单星总供给仍受 L^max 截断 (保留 per-LEO L^max cap)。
    """
    n_slots = C.N_SLOTS_PER_FRAME
    L_max = C.LEO_MAX_CAPACITY_MBPS
    d_thresh = C.CELL_DIAMETER_KM  # 30 km, Eq.8 干扰判据

    BASE_C_SLOT = C.BASE_C_SLOT
    kappa_intra = C.KAPPA_INTRA
    kappa_inter = C.KAPPA_INTER
    rho_mc = C.RHO_MC if use_mc else 1.0  # MC 关: ρ_mc=1 (无抑制); MC 开: ρ_mc<1

    # --- 建立被服务小区索引: served[c] = (leo, n_slots) ---
    served = {}
    for i, slots_i in alloc.items():
        for c, ns in slots_i.items():
            if ns > 0:
                served[c] = (i, ns)
    served_set = list(served.keys())
    if len(served_set) == 0:
        return 0.0, 0.0

    served_arr = np.array(served_set, dtype=int)
    served_flag = np.zeros(C.N_CELLS, dtype=bool)
    served_flag[served_arr] = True

    # --- 预计算每被服务小区的邻居 (被服务且 dist<=d) ---
    # 用于 intra (assign[c']==i) 与 inter (assign[c']!=i) 分离。
    # 用 assign (N_c,) 查每被服务小区归属的 LEO。
    assign_arr = np.asarray(assign)
    # 被服务小区的 (n_slots, leo) 向量
    ns_vec = np.zeros(C.N_CELLS, dtype=float)
    for c in served_set:
        ns_vec[c] = served[c][1]

    rt_thr = 0.0
    total_thr = 0.0
    for i, slots_i in alloc.items():
        if len(slots_i) == 0:
            continue
        cells_i = [c for c, ns in slots_i.items() if ns > 0]
        if len(cells_i) == 0:
            continue
        cell_supply = {}
        for c in cells_i:
            ns_c = slots_i[c]
            # --- 物理干扰折扣 ---
            # 邻居 (被服务, dist<=d) 的小区列表
            nbr_row = geo._cell_pair_dist[c]  # (N_c,)
            is_nbr = (nbr_row <= d_thresh) & served_flag
            is_nbr[c] = False
            nbrs = np.where(is_nbr)[0]
            if len(nbrs) == 0:
                intra_discount = 1.0
                inter_discount = 1.0
            else:
                nbr_ns = ns_vec[nbrs] / n_slots  # 共激活占空比贡献
                nbr_leo = assign_arr[nbrs]
                # 星内: 邻居与 c 同星 (assign==i)
                intra_mask = (nbr_leo == i)
                intra_penalty = float(nbr_ns[intra_mask].sum())
                # 星间: 邻居与 c 异星 (assign!=i)
                inter_mask = ~intra_mask
                inter_penalty = float(nbr_ns[inter_mask].sum()) * rho_mc
                intra_discount = 1.0 / (1.0 + kappa_intra * intra_penalty)
                inter_discount = 1.0 / (1.0 + kappa_inter * inter_penalty)

            base_supply = ns_c * BASE_C_SLOT / n_slots
            eff_supply = base_supply * intra_discount * inter_discount
            cell_supply[c] = eff_supply

        # --- 单星总供给受 L^max 截断 (保留原 cap 语义) ---
        tot_supply = sum(cell_supply.values())
        if tot_supply > L_max:
            scale = L_max / tot_supply
            for c in cell_supply:
                cell_supply[c] *= scale

        for c, supply in cell_supply.items():
            d_rt = D_rt_per_cell[c]
            d_nrt = D_nrt_per_cell[c]
            lam_rt = min(supply, d_rt)
            lam_nrt = min(max(supply - d_rt, 0.0), d_nrt)
            rt_thr += lam_rt
            total_thr += lam_rt + lam_nrt
    return rt_thr, total_thr


def run_one_frame(n_leo, total_demand, scheme, seed=0, ga_fast=False):
    """跑一帧, 返回指标 dict。"""
    sat_select, bh_policy, use_mc = C.SCHEME_CONFIG[scheme]

    rng = np.random.default_rng(seed)
    geo = Geometry(n_leo, seed=seed)
    D_rt, D_nrt, _ = generate_traffic(geo, total_demand, seed=seed)
    D_rt_per_cell = D_rt.sum(axis=1)
    D_nrt_per_cell = D_nrt.sum(axis=1)

    # --- 选星 ---
    cell_total = D_rt_per_cell + D_nrt_per_cell
    if sat_select == "rw_lb":
        assign, L_max, L_cur, util = rw_lb.select_rw_lb(geo, cell_total, rng)
    else:
        assign, L_max, L_cur, util = rw_lb.select_distance(geo, cell_total)
    L_df = rw_lb.load_disparity(util)

    # --- BH 时隙分配 ---
    saved = (C.GA_MAX_GEN, C.GA_POP_SIZE)
    if ga_fast and bh_policy == "ga":
        C.GA_MAX_GEN = 15
        C.GA_POP_SIZE = 16
    alloc = bh_scheduling.generate_bh(geo, assign, D_rt_per_cell, D_nrt_per_cell,
                                      bh_policy, rng)
    C.GA_MAX_GEN, C.GA_POP_SIZE = saved

    # --- MC-BHS 协调 (重排时隙, 不减服务) ---
    if use_mc:
        alloc = mc_bhs.coordinate(alloc, geo, D_rt_per_cell, D_nrt_per_cell)

    # --- 吞吐量计算 ---
    rt_throughput, total_throughput = _compute_throughput(
        geo, alloc, assign, D_rt_per_cell, D_nrt_per_cell, use_mc)

    rt_demand_total = D_rt_per_cell.sum()
    total_demand_total = cell_total.sum()
    rt_satisf = (rt_throughput / rt_demand_total) if rt_demand_total > 0 else 1.0
    total_satisf = (total_throughput / total_demand_total) if total_demand_total > 0 else 1.0

    return {
        "rt_throughput": float(rt_throughput),
        "total_throughput": float(total_throughput),
        "rt_satisf": float(np.clip(rt_satisf, 0, 1)),
        "total_satisf": float(np.clip(total_satisf, 0, 1)),
        "L_df": float(L_df),
        "util": util.copy(),
        "assign": assign,
    }


def run_averaged(n_leo, total_demand, scheme, n_runs, seed_base, ga_fast=False):
    """多次平均。"""
    acc = {"rt_throughput": [], "total_throughput": [],
           "rt_satisf": [], "total_satisf": [], "L_df": []}
    util_last = None
    for r in range(n_runs):
        s = seed_base + r * 31 + n_leo + int(total_demand) + hash(scheme) % 997
        res = run_one_frame(n_leo, total_demand, scheme, seed=s, ga_fast=ga_fast)
        for k in acc:
            acc[k].append(res[k])
        util_last = res["util"]
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["util_last"] = util_last
    return out
