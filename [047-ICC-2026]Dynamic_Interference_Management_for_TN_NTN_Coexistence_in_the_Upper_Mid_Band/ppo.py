"""中央 PPO：共享逐站 actor、混合连续/二元动作及精确联合策略概率。

论文公开了状态、动作和奖励，但未公开网络/训练超参。本模块的网络、GAE、
rollout 和优化设置均为可复验的实现选择，不是原作者的训练配置。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal


@dataclass(frozen=True)
class PPOSettings:
    hidden_size: int = 64
    learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float = 0.08
    rollout_steps: int = 1024
    minibatch_size: int = 256
    update_epochs: int = 4
    initial_log_std: float = -0.8
    initial_active_logit: float = 3.0
    initial_power_normalized: float = 0.95
    initial_tilt_normalized: float = -0.4
    value_scale: float = 100.0
    critic_time_features: bool = True
    critic_target: str = "discounted_mc"


def _linear(nin, nout, gain=math.sqrt(2), bias=0.0):
    layer = nn.Linear(nin, nout)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """集中观测，共享逐站参数；训练使用所有站的联合 log probability。"""

    def __init__(self, n_gnb=106, local_features=4, settings=None,
                 pass_steps=160, sweep_start_deg=10.0, sweep_step_deg=1.0):
        super().__init__()
        self.n_gnb = int(n_gnb)
        self.local_features = int(local_features)
        self.settings = settings or PPOSettings()
        self.pass_steps = int(pass_steps)
        self.sweep_start_deg = float(sweep_start_deg)
        self.sweep_step_deg = float(sweep_step_deg)
        h = self.settings.hidden_size
        self.actor_body = nn.Sequential(
            _linear(5 + self.local_features, h), nn.Tanh(),
            _linear(h, h), nn.Tanh(),
        )
        self.continuous_head = _linear(h, 3, gain=0.01)
        with torch.no_grad():
            self.continuous_head.bias.copy_(torch.tensor([
                math.atanh(self.settings.initial_power_normalized),
                math.atanh(self.settings.initial_power_normalized),
                math.atanh(self.settings.initial_tilt_normalized),
            ]))
        self.active_head = _linear(h, 1, gain=0.01,
                                   bias=self.settings.initial_active_logit)
        self.log_std = nn.Parameter(torch.full((3,), self.settings.initial_log_std))
        # 对站索引置换不敏感的中央价值估计，输入全局状态及局部均值/最大值。
        self.critic = nn.Sequential(
            _linear(5 + 2 * self.local_features + (4 if self.settings.critic_time_features else 0), h), nn.Tanh(),
            _linear(h, h), nn.Tanh(),
            _linear(h, 1, gain=0.01 if self.settings.value_scale > 1 else 1.0),
        )

    def features(self, obs):
        obs = obs.reshape(-1, 5 + self.n_gnb * self.local_features)
        global_state = obs[:, :5]
        local_state = obs[:, 5:].reshape(-1, self.n_gnb, self.local_features)
        site_state = torch.cat((global_state[:, None, :].expand(-1, self.n_gnb, -1),
                                local_state), dim=-1)
        pooled = torch.cat((global_state, local_state.mean(1),
                            local_state.max(1).values), dim=-1)
        if self.settings.critic_time_features:
            # scan角本来已包含时间；显式基函数改善有限时域价值的归纳偏置，
            # 没有给策略添加未来链路或未观测的信息。
            elapsed = ((global_state[:, 0] * 180 - self.sweep_start_deg)
                       / max(self.sweep_step_deg, 1e-6))
            remaining = (self.pass_steps - elapsed).clamp(0, self.pass_steps)
            fraction = remaining / self.pass_steps
            if self.settings.gamma < 1:
                horizon = ((1 - torch.pow(self.settings.gamma, remaining))
                           / (1 - self.settings.gamma) / self.settings.value_scale)
            else:
                horizon = remaining / self.settings.value_scale
            pooled = torch.cat((pooled, torch.stack(
                (fraction, fraction.square(), fraction.pow(3), horizon), dim=-1)), dim=-1)
        return site_state, pooled

    def distributions(self, obs):
        sites, pooled = self.features(obs)
        features = self.actor_body(sites)
        means = self.continuous_head(features)
        # 限制方差仅为数值稳定性，不对动作均值或输出结果作人为拟合。
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(means)
        continuous = Normal(means, std)
        active = Bernoulli(logits=self.active_head(features).squeeze(-1))
        values = self.critic(pooled).squeeze(-1) * self.settings.value_scale
        return continuous, active, values

    @staticmethod
    def joint_log_prob(continuous, active_dist, latent, active):
        # tanh 变量变换的精确、数值稳定的 Jacobian。
        log_jacobian = 2.0 * (math.log(2.0) - latent - nn.functional.softplus(-2.0 * latent))
        logp_cont = (continuous.log_prob(latent) - log_jacobian).sum((-1, -2))
        logp_active = active_dist.log_prob(active).sum(-1)
        return logp_cont + logp_active

    def sample(self, obs, deterministic=False):
        continuous, active_dist, values = self.distributions(obs)
        latent = continuous.mean if deterministic else continuous.sample()
        active = (active_dist.probs >= 0.5).float() if deterministic else active_dist.sample()
        actions = torch.cat((latent.tanh(), active[..., None]), dim=-1)
        logp = self.joint_log_prob(continuous, active_dist, latent, active)
        return actions, latent, active, logp, values

    def evaluate(self, obs, latent, active):
        continuous, active_dist, values = self.distributions(obs)
        logp = self.joint_log_prob(continuous, active_dist, latent, active)
        # 连续部分的熵是未变换正态的探索正则；不冒称 tanh 后的精确熵。
        entropy_per_site = (continuous.entropy().sum(-1) + active_dist.entropy()).mean(-1)
        return logp, values, entropy_per_site


class PPOPolicy:
    def __init__(self, model, metadata=None):
        self.model = model.cpu().eval()
        self.metadata = metadata or {}

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        actions = self.model.sample(torch.from_numpy(obs)[None], deterministic)[0]
        return actions[0].numpy()

    @torch.no_grad()
    def action_statistics(self, obs):
        continuous, active, values = self.model.distributions(
            torch.as_tensor(np.asarray(obs, dtype=np.float32))[None])
        return {"continuous_mean": continuous.mean.tanh()[0].numpy(),
                "active_probability": active.probs[0].numpy(),
                "value": float(values[0])}


def load_policy(path):
    """仅装载本项目 checkpoint 的张量及基本元数据，返回可直接评估的策略。"""
    torch.set_num_threads(1)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    saved_settings = dict(checkpoint["ppo_settings"])
    if checkpoint.get("format_version", 1) < 2:
        # 旧版直接预测未归一化value且没有显式时域基函数，保留读取诊断能力。
        saved_settings.update(value_scale=1.0, critic_time_features=False,
                              critic_target="gae", initial_power_normalized=0.0,
                              initial_tilt_normalized=0.0)
    settings = PPOSettings(**saved_settings)
    sim = checkpoint.get("simulation_config", {})
    model = ActorCritic(checkpoint["n_gnb"], checkpoint["local_features"], settings,
                        sim.get("n_snapshots", 160), sim.get("elevation_start_deg", 10.0),
                        sim.get("elevation_step_deg", 1.0))
    model.load_state_dict(checkpoint["model_state"])
    return PPOPolicy(model, {k: v for k, v in checkpoint.items() if k != "model_state"})


def _number(info, keys, default=float("nan")):
    for key in keys:
        if key in info:
            arr = np.asarray(info[key])
            if arr.size:
                return float(np.mean(arr))
    return default


def train(cfg, seed, episodes, out, settings=None):
    """真实环境采样 + GAE + clipped PPO；保存每回合日志、更新日志和权重。

    Args:
        cfg: SimConfig。训练任务按Fig.4/5的独立目标均匀选择。
        seed: 训练随机种子；评估应另用独立的场景种子。
        episodes: 完整 160-step pass 的数量。
        out: 输出根目录，模型位于 out/models，日志位于 out/training。
    """
    from tn_ntn_model import CoexistenceEnv

    source_root = Path(__file__).resolve().parent
    source_hashes = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("config.py", "tn_ntn_model.py", "ppo.py")
    }

    if settings is None:
        initial_tilt = (2 * (cfg.default_tilt_deg - cfg.tilt_min_deg)
                        / (cfg.tilt_max_deg - cfg.tilt_min_deg) - 1)
        settings = replace(PPOSettings(), initial_tilt_normalized=float(np.clip(initial_tilt, -0.999, 0.999)))
    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    model_dir, log_dir = out / "models", out / "training"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = CoexistenceEnv(cfg, int(seed))
    # 5个独立论文目标：3密度全T1，加中密度的混合/全T2；去除重复的中密度全T1。
    # 任务选择使用独立RNG，不为选择任务而消耗几何生成器的随机数。
    middle_density = float(cfg.density_values[len(cfg.density_values) // 2])
    training_tasks = list(dict.fromkeys(
        [(float(density), 1.0) for density in cfg.density_values]
        + [(middle_density, float(mix)) for mix in cfg.mix_values]))
    task_seed_entropy = [int(seed), 0x54534B]
    task_rng = np.random.default_rng(np.random.SeedSequence(task_seed_entropy))
    task_counts = np.zeros(len(training_tasks), dtype=int)
    current_task = 0

    def reset_training_task():
        nonlocal current_task
        current_task = int(task_rng.integers(len(training_tasks)))
        task_counts[current_task] += 1
        density, mix = training_tasks[current_task]
        return np.asarray(env.reset(density=density, t1_fraction=mix), dtype=np.float32)

    obs = reset_training_task()
    local_features = (obs.size - 5) // cfg.n_gnb
    if obs.size != 5 + cfg.n_gnb * local_features:
        raise ValueError("invalid observation shape")
    model = ActorCritic(cfg.n_gnb, local_features, settings, cfg.n_snapshots,
                        cfg.elevation_start_deg, cfg.elevation_step_deg)
    config_dict = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    checkpoint_metadata = {
        "format_version": 2, "n_gnb": cfg.n_gnb, "local_features": local_features,
        "seed": int(seed), "ppo_settings": asdict(settings),
        "simulation_config": config_dict, "source_sha256": source_hashes,
        "optimizer_design": "independent actor/critic Adam and separate gradient-norm clipping",
        "training_task_sampler": {
            "scheme": "uniform over distinct Fig.4/5 target tasks; independent of geometry RNG",
            "task_seed_entropy": task_seed_entropy,
            "tasks": [{"density": density, "t1_fraction": mix}
                      for density, mix in training_tasks],
        },
    }
    # 保存实际训练起点，以区分动作初始化带来的改善和梯度学习的额外增益。
    torch.save({**checkpoint_metadata, "episodes": 0, "env_steps": 0,
                "model_state": model.cpu().state_dict()},
               model_dir / f"ppo_untrained_seed_{seed}.pt")
    # 两个网络没有共享可训练参数，分别优化/裁剪，避免较大的价值梯度
    # 通过全模型范数裁剪把策略梯度一同压小。奖励及联合PPO目标不变。
    actor_parameters = [parameter for name, parameter in model.named_parameters()
                        if not name.startswith("critic.")]
    critic_parameters = list(model.critic.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=settings.learning_rate, eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=settings.critic_learning_rate, eps=1e-5)
    initial_weights = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    initial_parameters = {name: parameter.detach().clone()
                          for name, parameter in model.named_parameters()}
    steps_per_episode = getattr(cfg, "n_snapshots", getattr(cfg, "pass_steps", 160))
    target_steps = int(episodes) * int(steps_per_episode)
    started = time.perf_counter()
    episode_rows, update_rows = [], []
    total_steps, episode_number = 0, 0
    ep_rewards, ep_metrics = [], []

    while total_steps < target_steps:
        n = min(settings.rollout_steps, target_steps - total_steps)
        observations = np.empty((n, obs.size), dtype=np.float32)
        latent_actions = np.empty((n, cfg.n_gnb, 3), dtype=np.float32)
        active_actions = np.empty((n, cfg.n_gnb), dtype=np.float32)
        rewards, values, log_probs, dones = [np.empty(n, dtype=np.float32) for _ in range(4)]
        model.eval()
        for t in range(n):
            observations[t] = obs
            with torch.no_grad():
                action, latent, active, logp, value = model.sample(torch.from_numpy(obs)[None])
            action_np = action[0].numpy()
            next_obs, reward, done, info = env.step(action_np)
            latent_actions[t], active_actions[t] = latent[0].numpy(), active[0].numpy()
            rewards[t], values[t], log_probs[t], dones[t] = reward, value.item(), logp.item(), done
            total_steps += 1
            ep_rewards.append(float(reward))
            ep_metrics.append((
                _number(info, ("active_ratio", "active_fraction")),
                _number(info, ("harmful_fraction", "harmful_ratio", "violation_ratio", "eta")),
                _number(info, ("throughput_bps", "ntn_throughput_bps", "throughput")),
            ))
            obs = np.asarray(next_obs, dtype=np.float32)
            if done:
                episode_number += 1
                metrics = np.asarray(ep_metrics)
                row = {
                    "seed": int(seed), "episode": episode_number, "env_steps": total_steps,
                    "return": float(np.sum(ep_rewards)), "mean_reward": float(np.mean(ep_rewards)),
                    "active_ratio": float(np.mean(metrics[:, 0])),
                    "harmful_fraction": float(np.mean(metrics[:, 1])),
                    "throughput_bps": float(np.mean(metrics[:, 2])),
                    "density": float(env.density), "t1_fraction": float(env.t1_fraction),
                    "task_index": current_task,
                }
                episode_rows.append(row)
                if episode_number % 32 == 0 or episode_number == episodes:
                    print(f"PPO seed={seed} episode={episode_number}/{episodes} "
                          f"reward={row['mean_reward']:.4f} "
                          f"active={row['active_ratio']:.3f}", flush=True)
                ep_rewards, ep_metrics = [], []
                if total_steps < target_steps:
                    obs = reset_training_task()

        with torch.no_grad():
            next_value = model.distributions(torch.from_numpy(obs)[None])[2].item()
        advantages = np.empty(n, dtype=np.float32)
        advantage = 0.0
        for t in reversed(range(n)):
            bootstrap = next_value if t == n - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + settings.gamma * bootstrap * nonterminal - values[t]
            advantage = delta + settings.gamma * settings.gae_lambda * nonterminal * advantage
            advantages[t] = advantage
        if settings.critic_target == "discounted_mc":
            # rollout内到终止的样本采用实际折扣回报；仅最后一个截断片段
            # 用下一状态value bootstrap，不把GAE的慢漂移目标当作MC回报。
            returns = np.empty(n, dtype=np.float32)
            tail = next_value
            for t in reversed(range(n)):
                tail = rewards[t] + settings.gamma * tail * (1.0 - dones[t])
                returns[t] = tail
        elif settings.critic_target == "gae":
            returns = advantages + values
        else:
            raise ValueError(f"unknown critic target: {settings.critic_target}")
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        batch_obs = torch.from_numpy(observations)
        batch_latent = torch.from_numpy(latent_actions)
        batch_active = torch.from_numpy(active_actions)
        batch_old_logp = torch.from_numpy(log_probs)
        batch_adv = torch.from_numpy(advantages)
        batch_return = torch.from_numpy(returns)
        # 保留末期少量学习率，便于在未知论文超参下持续改进而不冻结。
        fraction = 1.0 - 0.8 * total_steps / target_steps
        optimizer.param_groups[0]["lr"] = settings.learning_rate * fraction
        critic_optimizer.param_groups[0]["lr"] = settings.critic_learning_rate * fraction
        model.train()
        losses = []
        stop_actor = False
        actor_updates = 0
        for epoch in range(settings.update_epochs):
            indices = torch.randperm(n)
            for begin in range(0, n, settings.minibatch_size):
                idx = indices[begin:begin + settings.minibatch_size]
                new_logp, new_values, entropy = model.evaluate(
                    batch_obs[idx], batch_latent[idx], batch_active[idx])
                log_ratio = new_logp - batch_old_logp[idx]
                # 联合 ratio 计算不按站数归一化，也不替换成局部独立 PPO。
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > settings.clip_ratio).float().mean()
                if approx_kl.item() > 1.5 * settings.target_kl:
                    stop_actor = True
                pg1 = -batch_adv[idx] * ratio
                pg2 = -batch_adv[idx] * ratio.clamp(1 - settings.clip_ratio, 1 + settings.clip_ratio)
                policy_loss = torch.maximum(pg1, pg2).mean()
                raw_value_loss = (new_values - batch_return[idx]).square().mean()
                value_loss = raw_value_loss / settings.value_scale ** 2
                entropy_loss = entropy.mean()
                actor_loss = policy_loss - settings.entropy_coefficient * entropy_loss
                critic_loss = settings.value_coefficient * value_loss
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError("non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                if not stop_actor:
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor_parameters, settings.max_grad_norm)
                    optimizer.step()
                    actor_updates += 1
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic_parameters, settings.max_grad_norm)
                critic_optimizer.step()
                losses.append((policy_loss.item(), value_loss.item(), entropy_loss.item(),
                               approx_kl.item(), clip_fraction.item(), raw_value_loss.item()))
        metric = np.mean(losses, axis=0)
        with torch.no_grad():
            fitted_values = model.distributions(batch_obs)[2].numpy()
        return_variance = float(np.var(returns))
        explained_variance = (1.0 - float(np.var(returns - fitted_values)) / return_variance
                              if return_variance > 1e-10 else 0.0)
        update_rows.append({
            "seed": int(seed), "update": len(update_rows) + 1, "env_steps": total_steps,
            "policy_loss": float(metric[0]), "value_loss": float(metric[1]),
            "value_loss_raw": float(metric[5]),
            "base_entropy_per_site": float(metric[2]), "approx_joint_kl": float(metric[3]),
            "clip_fraction": float(metric[4]), "learning_rate": optimizer.param_groups[0]["lr"],
            "critic_learning_rate": critic_optimizer.param_groups[0]["lr"],
            "value_prediction_mean": float(np.mean(fitted_values)),
            "return_mean": float(np.mean(returns)), "explained_variance": explained_variance,
            "actor_updates": actor_updates, "critic_updates": len(losses),
        })

    if episode_number != int(episodes):
        raise RuntimeError(f"episode length mismatch: expected {episodes}, got {episode_number}")
    final_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
    parameter_change = float(torch.linalg.vector_norm(final_weights - initial_weights))
    group_changes = {}
    for group in ("actor", "critic"):
        squares = sum((parameter.detach() - initial_parameters[name]).square().sum()
                      for name, parameter in model.named_parameters()
                      if name.startswith("critic.") == (group == "critic"))
        group_changes[group] = float(torch.sqrt(squares))
    if not parameter_change > 0:
        raise RuntimeError("PPO parameters did not change")
    checkpoint_path = model_dir / f"ppo_seed_{seed}.pt"
    current_hashes = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in source_hashes
    }
    if current_hashes != source_hashes:
        raise RuntimeError("simulation/training source changed during training; rerun from one version")
    # 仅保存张量和基本类型，兼容 torch.load(weights_only=True)。
    torch.save({
        **checkpoint_metadata, "episodes": int(episodes), "env_steps": total_steps,
        "model_state": model.cpu().state_dict(),
    }, checkpoint_path)
    for label, rows in (("episodes", episode_rows), ("updates", update_rows)):
        with (log_dir / f"{label}_seed_{seed}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "seed": int(seed), "episodes": episode_number, "env_steps": total_steps,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "parameter_change_l2": parameter_change,
        "actor_parameter_change_l2": group_changes["actor"],
        "critic_parameter_change_l2": group_changes["critic"],
        "first_value_loss": update_rows[0]["value_loss"],
        "last_value_loss": update_rows[-1]["value_loss"],
        "first_value_loss_raw": update_rows[0]["value_loss_raw"],
        "last_value_loss_raw": update_rows[-1]["value_loss_raw"],
        "last_explained_variance": update_rows[-1]["explained_variance"],
        "last_value_prediction_mean": update_rows[-1]["value_prediction_mean"],
        "last_return_mean": update_rows[-1]["return_mean"],
        "first_32_mean_reward": float(np.mean([r["mean_reward"] for r in episode_rows[:32]])),
        "last_32_mean_reward": float(np.mean([r["mean_reward"] for r in episode_rows[-32:]])),
        "checkpoint": str(checkpoint_path.name), "ppo_settings": asdict(settings),
        "source_sha256": source_hashes,
        "training_task_counts": [{"density": density, "t1_fraction": mix,
                                   "episodes": int(count)}
                                  for (density, mix), count in zip(training_tasks, task_counts)],
    }
    (log_dir / f"summary_seed_{seed}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return PPOPolicy(model), summary


if __name__ == "__main__":
    import argparse
    from config import CFG

    parser = argparse.ArgumentParser(description="Train actual PPO on the TN-NTN pass environment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--profile", choices=("main", "ascent_envelope"), default="main")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    training_cfg = (CFG if args.profile == "main" else replace(
        CFG, ut_antenna_model="wbes_envelope", gnb_antenna_model="ascent_array",
        noise_temperature_k=200.0))
    for training_seed in args.seeds:
        _, training_summary = train(training_cfg, training_seed, args.episodes, args.output)
        print(json.dumps(training_summary, ensure_ascii=False), flush=True)
