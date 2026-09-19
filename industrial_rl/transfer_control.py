from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
from torch import nn

from .neural_common import set_global_seed
from .sac_energy import EnergyProductionConfig, EnergyProductionEnv, evaluate_energy_policy


@dataclass(frozen=True)
class TransferDQNConfig:
    episodes: int = 1400
    action_levels: int = 9
    batch_size: int = 96
    replay_size: int = 30_000
    warmup: int = 500
    learning_rate: float = 8e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    target_update: int = 250
    seed: int = 811


class ProductionQNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProductionDQN:
    def __init__(self, base_config: EnergyProductionConfig, config: TransferDQNConfig = TransferDQNConfig()):
        self.base_config = base_config
        self.config = config
        set_global_seed(config.seed)
        self.actions = np.linspace(-1.0, 1.0, config.action_levels, dtype=np.float32)
        self.online = ProductionQNetwork(5, config.action_levels)
        self.target = ProductionQNetwork(5, config.action_levels)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque = deque(maxlen=config.replay_size)
        self.rng = np.random.default_rng(config.seed)
        self.updates = 0

    def clone(self) -> "ProductionDQN":
        other = ProductionDQN(self.base_config, self.config)
        other.online.load_state_dict(self.online.state_dict())
        other.target.load_state_dict(self.target.state_dict())
        return other

    def _epsilon(self, episode: int, total: int) -> float:
        frac = episode / max(total - 1, 1)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def _update(self) -> None:
        if len(self.replay) < max(self.config.warmup, self.config.batch_size):
            return
        idx = self.rng.integers(0, len(self.replay), size=self.config.batch_size)
        batch = [self.replay[int(i)] for i in idx]
        s = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        a = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        ns = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        d = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)

        q = self.online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            target = r + (1.0 - d) * self.config.gamma * self.target(ns).max(1).values
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def fit(
        self,
        *,
        episodes: int | None = None,
        config_sampler: Callable[[np.random.Generator], EnergyProductionConfig] | None = None,
        seed_offset: int = 0,
    ) -> "ProductionDQN":
        total = int(self.config.episodes if episodes is None else episodes)
        for ep in range(total):
            env_config = self.base_config if config_sampler is None else config_sampler(self.rng)
            env = EnergyProductionEnv(env_config)
            obs = env.reset(seed=self.config.seed + seed_offset + ep)
            done = False
            eps = self._epsilon(ep, total)
            while not done:
                if self.rng.random() < eps:
                    action_index = int(self.rng.integers(0, len(self.actions)))
                else:
                    with torch.no_grad():
                        q = self.online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
                    action_index = int(torch.argmax(q).item())
                next_obs, reward, done, _ = env.step(float(self.actions[action_index]))
                self.replay.append((obs.copy(), action_index, float(reward), next_obs.copy(), bool(done)))
                obs = next_obs
                self._update()
        self.target.load_state_dict(self.online.state_dict())
        return self

    def policy(self) -> Callable[[np.ndarray], float]:
        def choose(obs: np.ndarray) -> float:
            with torch.no_grad():
                q = self.online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
            return float(self.actions[int(torch.argmax(q).item())])
        return choose


def source_factory_config() -> EnergyProductionConfig:
    return EnergyProductionConfig(base_demand=5.5, demand_amplitude=1.4, demand_noise_std=0.8, base_price=0.95)


def target_factory_config() -> EnergyProductionConfig:
    return EnergyProductionConfig(base_demand=7.1, demand_amplitude=2.2, demand_noise_std=1.35, base_price=1.12, price_amplitude=0.68)


def domain_randomization_sampler(base: EnergyProductionConfig) -> Callable[[np.random.Generator], EnergyProductionConfig]:
    def sample(rng: np.random.Generator) -> EnergyProductionConfig:
        return replace(
            base,
            base_demand=float(rng.uniform(5.0, 7.5)),
            demand_amplitude=float(rng.uniform(1.0, 2.5)),
            demand_noise_std=float(rng.uniform(0.6, 1.5)),
            base_price=float(rng.uniform(0.85, 1.2)),
            price_amplitude=float(rng.uniform(0.35, 0.75)),
        )
    return sample


def evaluate_transfer(agent: ProductionDQN, config: EnergyProductionConfig, *, episodes: int = 200, seed: int = 120_000) -> dict[str, float]:
    return evaluate_energy_policy(agent.policy(), config, episodes=episodes, seed=seed)


def transfer_benchmark(
    train_config: TransferDQNConfig = TransferDQNConfig(),
    *,
    fine_tune_episodes: int = 300,
    evaluation_episodes: int = 200,
) -> dict[str, dict[str, float]]:
    source = source_factory_config()
    target = target_factory_config()

    source_agent = ProductionDQN(source, train_config).fit()
    zero_shot = evaluate_transfer(source_agent, target, episodes=evaluation_episodes)

    fine_tuned = source_agent.clone()
    fine_tuned.base_config = target
    fine_tuned.fit(episodes=fine_tune_episodes, seed_offset=50_000)
    fine_tuned_metrics = evaluate_transfer(fine_tuned, target, episodes=evaluation_episodes)

    scratch = ProductionDQN(target, train_config).fit(episodes=fine_tune_episodes, seed_offset=70_000)
    scratch_metrics = evaluate_transfer(scratch, target, episodes=evaluation_episodes)

    randomized = ProductionDQN(source, train_config).fit(config_sampler=domain_randomization_sampler(source), seed_offset=90_000)
    randomized_metrics = evaluate_transfer(randomized, target, episodes=evaluation_episodes)

    return {
        "zero_shot_source_to_target": zero_shot,
        "fine_tuned_on_target": fine_tuned_metrics,
        "target_from_scratch_same_budget": scratch_metrics,
        "domain_randomized_zero_shot": randomized_metrics,
    }
