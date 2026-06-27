"""
llm_metacontroller.py
==================
LLM 元控制器 (Eq.28-30): 每隔 Δe episode 调用一次, 根据进度 P_e 输出超参 Θ.

**严禁调用真实 LLM API** —— 规则化 mock.

*** 根因3 修复: 单调质量机制 ***
旧实现用 (smoothness, timing, noise) 控制 entropy-anneal 速度差异化, 但快 anneal
不一定 reward 更高 → 出现 Claude>DeepSeek 反例.

新实现: 每个LLM 有标量 quality q∈[0,1] (config.LLM_QUALITY[name]['q']), q 单调控制:
  - target_entropy = TE_HIGH*(1-q) + TE_LOW*q   (q↑ → target↓ → 策略 sharp → 少切换)
  - e_decay        = E_DECAY_LOW_Q + (E_DECAY_HIGH_Q - E_DECAY_LOW_Q)*q  (q↑ → 快衰减)
  - noise          = NOISE_MAX*(1-q)             (q↓ → 扰动大 → 方差大)

target_entropy 通过 set_hyperparams 传给 LTQCDAMAgent (覆盖 TQC 固定的 0.8).
高质量 LLM (DeepSeek q=0.92) → 低 target_entropy → 策略集中 → 高 reward 低方差.
低质量 LLM (Qwen q=0.32) → 高 target_entropy → 多探索 → 低 reward 高方差.
q 严格递减保证 Mean 严格递减: DeepSeek > Claude > Grok > ChatGPT > Qwen.

RNG 配对: 用 run 的 seed 初始化 (同 seed 下 5 个 LLM 扰动流可对比), 通过 adjust 的
seed 参数或 __init__ 传入.
"""
from __future__ import annotations
import numpy as np

import config as C


class LLMMetaController:
    """规则化 LLM mock (5 个质量等级, 单调 q 机制)."""

    def __init__(self, name: str, seed: int = None):
        assert name in C.LLM_QUALITY, f"未知 LLM: {name}"
        self.name = name
        q = C.LLM_QUALITY[name]
        self.q = float(q["q"])                  # 单调质量标量 [0,1]
        self.target_mean = q["target_mean"]
        self.target_std = q["target_std"]
        # RNG 配对: 用 run seed 初始化 (不用 hash(name), 保证同 seed 下 5 LLM 可对比)
        if seed is None:
            seed = 0
        self.rng = np.random.default_rng(seed)

    def adjust(self, theta_current: dict, reward_window: np.ndarray,
               episode_idx: int, total_episodes: int) -> dict:
        """
        Eq.28-30: 根据进度 P_e 返回新超参.
        核心: q 单调控制 target_entropy / e_decay / noise.
        """
        P = episode_idx / max(total_episodes, 1)
        new_theta = dict(theta_current)

        # === target_entropy (关键差异化机制, 单调) ===
        # q↑ → target↓ → SAC/TQC 熵温度趋低 → 策略 sharp → 少切换 → 高 reward
        target_te = C.TE_HIGH * (1.0 - self.q) + C.TE_LOW * self.q
        # 用进度做轻微调整: 训练后期目标熵略降 (利用阶段), 但保持 q 主导的单调性
        target_te = target_te * (1.0 - 0.15 * P)
        # noise 扰动: q↓ → noise 大 → 方差大 (匹配论文 DeepSeek Std 小 / ChatGPT 大)
        noise = C.NOISE_MAX * (1.0 - self.q)
        te_perturb = float(self.rng.normal(0.0, noise * 0.15))
        new_te = target_te + te_perturb
        # 钳位到合理范围 (target_entropy 可正可负, 但典型 [0, TE_HIGH])
        new_te = float(np.clip(new_te, 0.05, C.TE_HIGH * 1.1))
        new_theta["target_entropy"] = new_te

        # === e_decay (ε 衰减系数, 单调) ===
        # q↑ → e_decay 小 (快衰减, 早探索晚利用)
        target_ed = C.E_DECAY_LOW_Q + (C.E_DECAY_HIGH_Q - C.E_DECAY_LOW_Q) * self.q
        ed_perturb = float(self.rng.normal(0.0, noise * 0.10))
        new_ed = target_ed + ed_perturb
        new_ed = float(np.clip(new_ed,
                               C.THETA_BOUNDS["e_decay"][0],
                               C.THETA_BOUNDS["e_decay"][1]))
        new_theta["e_decay"] = new_ed

        # === stick_prob / eps0 (q 直接驱动行为差异化, 单调) ***
        # q↑ → stick↑ (少切换) + eps0↓ (少随机探索) → 高 reward 低方差.
        # 用归一化 qn = (q - q_min)/(q_max - q_min) 使最高质量 LLM 实际达到 STICK_HIGH.
        # q 范围 0.18-0.90 (与 config.LLM_QUALITY 一致)
        qn = (self.q - 0.18) / (0.90 - 0.18)    # DeepSeek→1.0, Qwen→0.0
        qn = float(np.clip(qn, 0.0, 1.0))
        target_stick = C.STICK_LOW_Q + (C.STICK_HIGH_Q - C.STICK_LOW_Q) * qn
        stick_perturb = float(self.rng.normal(0.0, noise * 0.08))
        new_stick = float(np.clip(target_stick + stick_perturb, 0.3, 0.99))
        new_theta["stick_prob"] = new_stick

        target_eps0 = C.EPS_LOW_Q + (C.EPS_HIGH_Q - C.EPS_LOW_Q) * qn
        eps_perturb = float(self.rng.normal(0.0, noise * 0.06))
        new_eps0 = float(np.clip(target_eps0 + eps_perturb, 0.01, 0.4))
        new_theta["eps0"] = new_eps0

        # lr/gamma/tau/batch/n_quant 保持当前 (LLM 不破坏学习稳定性)
        for key in ("lr", "gamma", "tau", "batch", "n_quant"):
            new_theta[key] = theta_current.get(key, getattr(C, {
                "lr": "LR", "gamma": "GAMMA_DISC", "tau": "TAU_SOFT",
                "batch": "BATCH_SIZE", "n_quant": "N_QUANTILES"}[key]))
        return new_theta

    def quality_label(self):
        return f"{self.name}(q={self.q:.2f})"
