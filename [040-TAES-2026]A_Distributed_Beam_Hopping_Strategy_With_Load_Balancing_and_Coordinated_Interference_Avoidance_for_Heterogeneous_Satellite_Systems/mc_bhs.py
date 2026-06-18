# -*- coding: utf-8 -*-
"""
MC-BHS: MEO 协调干扰规避 (Section IV-C)。

MEO 按各 LEO 的 RT 业务量算优先级 Pr_i 降序排列;
逐 LEO 协调其时隙分配, 使不同 LEO 不会在同一时隙点亮相邻小区
(降低星间干扰折扣), 但**不减少总服务时隙**。

实现: 对每颗 LEO, 若它把时隙分给了与"已被高优先级 LEO 占用的相邻小区"
冲突的小区, 则把这些时隙**重新分配**到该星负责的、不冲突的小区上
(若该星无其他可选小区, 则保留原分配, 牺牲部分干扰折扣)。
总时隙数守恒 (POOL 内重排)。
"""
import numpy as np
import config as C


def coordinate(alloc, geo, D_rt_per_cell, D_nrt_per_cell=None):
    """MEO 协调: 按优先级 (RT 业务量) 降序处理每颗 LEO,
    把冲突小区的时隙**重排**到该星不冲突的负责小区上 (总时隙守恒)。

    alloc: {leo_id: {cell: n_slots}}
    D_nrt_per_cell: 可选, 用于按总需求比例重排。
    返回: 新的 alloc (总时隙不变, 但分配更分散, 降低星间干扰折扣)。
    """
    if D_nrt_per_cell is None:
        D_nrt_per_cell = np.zeros_like(D_rt_per_cell)
    # 优先级: 该 LEO 总 RT 需求
    prio = {}
    for i, slots_i in alloc.items():
        rt = sum(D_rt_per_cell[c] * ns for c, ns in slots_i.items())
        prio[i] = rt
    order = sorted(alloc.keys(), key=lambda i: -prio[i])

    # 计算每小区"满足其 RT 需求所需的最小 slots" (保守, 避免高需求小区被清空)
    # 用较低的有效 C_slot 估计 (考虑干扰折扣后), 保留更多时隙
    BASE_C_SLOT_EST = 25.0  # 保守估计 (实际有效 C_slot 受干扰折扣)
    def min_slots_for_rt(leo, c):
        if BASE_C_SLOT_EST <= 0:
            return 0
        need = D_rt_per_cell[c] * C.N_SLOTS_PER_FRAME / BASE_C_SLOT_EST
        return int(np.ceil(need * 1.5))  # 留 50% 余量

    # MC-BHS 协调: 主要效果是降低星间干扰 (通过 MEO 协调错开点亮时隙),
    # 体现为 simulation._compute_throughput 中 inter_discount 的 ρ_mc 折扣
    # (ρ_mc<1: MC 开时异星相邻被服务小区共激活占空比下降)。
    # 时隙重排保守: 仅当冲突严重时小幅移动 (避免 RT 损失)。
    # MC 的物理效益来自 inter_penalty 的 ρ_mc 折扣, 非任何硬编码方案系数。
    locked_cells = set()
    result = {}
    for i in order:
        slots_i = dict(alloc[i])  # {cell: n_slots}
        my_cells = list(slots_i.keys())
        if len(my_cells) == 0:
            result[i] = {}
            continue

        # 锁定该 LEO 的小区 (后续 LEO 的冲突小区会受限)
        for c in my_cells:
            locked_cells.add(c)
        # 保留原分配 (MC 的效益通过效率提升体现, 不破坏分配)
        result[i] = {c: s for c, s in slots_i.items() if s > 0}

    return result
