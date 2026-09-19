from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .neural_common import set_global_seed


@dataclass(frozen=True)
class EnergyProductionConfig:
    horizon: int = 18
    max_rate: float = 10.0
    max_inventory: float = 24.0
    max_backlog: float = 18.0
    initial_inventory: float = 3.0
    base_demand: float = 6.0
    demand_amplitude: float = 1.8
    demand_noise_std: float = 1.0
    base_price: float = 1.0
    price_amplitude: float = 0.55
    price_noise_std: float = 0.08
    linear_energy_coeff: float = 0.55
    quadratic_energy_coeff: float = 0.045
    holding_cost: float = 0.35
    backlog_cost: float = 4.5
    ramp_cost: float = 0.18
    overflow_cost: float = 1.5

    def validate(self) -> None:
        if self.horizon < 1 or self.max_rate <= 0:
            raise ValueError("horizon and max_rate must be positive")
        if self.max_inventory <= 0 or self.max_backlog <= 0:
            raise ValueError("inventory/backlog bounds must be positive")
        if not -self.max_backlog <= self.initial_inventory <= self.max_inventory:
            raise ValueError("initial inventory outside bounds")
        if self.demand_noise_std < 0 or self.price_noise_std < 0:
            raise ValueError("noise standard deviations must be non-negative")


class EnergyProductionEnv:
    """Continuous production-rate control under stochastic demand and energy prices."""

    def __init__(self, config: EnergyProductionConfig = EnergyProductionConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.inventory = config.initial_inventory
        self.previous_rate = 0.0
        self.price = self.expected_price(0)

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def action_dim(self) -> int:
        return 1

    def expected_demand(self, t: int) -> float:
        phase = 2.0 * np.pi * int(t) / self.config.horizon
        return float(
            max(
                0.2,
                self.config.base_demand
                + self.config.demand_amplitude * np.sin(phase - 0.3),
            )
        )

    def expected_price(self, t: int) -> float:
        phase = 2.0 * np.pi * int(t) / self.config.horizon
        return float(
            max(
                0.05,
                self.config.base_price
                + self.config.price_amplitude * np.cos(phase + 0.8),
            )
        )

    def observation(self) -> np.ndarray:
        demand_mean = self.expected_demand(self.t)
        return np.asarray(
            [
                self.t / self.config.horizon,
                self.inventory / max(self.config.max_inventory, self.config.max_backlog),
                demand_mean / (self.config.base_demand + self.config.demand_amplitude + 1e-6),
                self.price / (self.config.base_price + self.config.price_amplitude + 1e-6),
                self.previous_rate / self.config.max_rate,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.inventory = self.config.initial_inventory
        self.previous_rate = 0.0
        self.price = max(
            0.05,
            self.expected_price(0) + self.rng.normal(0.0, self.config.price_noise_std),
        )
        return self.observation()

    def _map_action(self, action: float | np.ndarray) -> float:
        raw = float(np.asarray(action).reshape(-1)[0])
        clipped = float(np.clip(raw, -1.0, 1.0))
        return (clipped + 1.0) * 0.5 * self.config.max_rate

    def transition(
        self,
        inventory: float,
        previous_rate: float,
        production_rate: float,
        demand: float,
        price: float,
    ):
        production_rate = float(np.clip(production_rate, 0.0, self.config.max_rate))
        raw_inventory = float(inventory) + production_rate - float(demand)
        next_inventory = float(
            np.clip(raw_inventory, -self.config.max_backlog, self.config.max_inventory)
        )
        overflow = max(raw_inventory - self.config.max_inventory, 0.0)
        unmet_overflow = max(-self.config.max_backlog - raw_inventory, 0.0)

        energy_use = (
            self.config.linear_energy_coeff * production_rate
            + self.config.quadratic_energy_coeff * production_rate**2
        )
        energy_cost = float(price) * energy_use
        holding = self.config.holding_cost * max(next_inventory, 0.0)
        backlog = self.config.backlog_cost * max(-next_inventory, 0.0)
        ramp = self.config.ramp_cost * abs(production_rate - float(previous_rate))
        overflow_penalty = self.config.overflow_cost * overflow + 8.0 * unmet_overflow
        cost = energy_cost + holding + backlog + ramp + overflow_penalty

        return next_inventory, -float(cost), {
            "cost": float(cost),
            "energy_cost": float(energy_cost),
            "energy_use": float(energy_use),
            "holding_cost": float(holding),
            "backlog_cost": float(backlog),
            "ramp_cost": float(ramp),
            "production_rate": float(production_rate),
            "demand": float(demand),
            "price": float(price),
            "inventory": float(next_inventory),
        }

    def step(self, action: float | np.ndarray):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode finished")

        production_rate = self._map_action(action)
        demand = max(
            0.0,
            self.expected_demand(self.t)
            + self.rng.normal(0.0, self.config.demand_noise_std),
        )
        next_inventory, reward, info = self.transition(
            self.inventory,
            self.previous_rate,
            production_rate,
            demand,
            self.price,
        )

        self.inventory = next_inventory
        self.previous_rate = production_rate
        self.t += 1
        done = self.t >= self.config.horizon

        if not done:
            self.price = max(
                0.05,
                self.expected_price(self.t)
                + self.rng.normal(0.0, self.config.price_noise_std),
            )

        return self.observation(), reward, done, info


def evaluate_energy_policy(
    policy: Callable[[np.ndarray], float],
    config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    episodes: int = 500,
    seed: int = 90_000,
) -> dict[str, float]:
    env = EnergyProductionEnv(config)
    costs, energy, final_backlog, ramping = [], [], [], []

    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        total_cost = 0.0
        total_energy = 0.0
        total_ramp = 0.0
        while not done:
            action = float(policy(obs))
            previous_rate = env.previous_rate
            obs, reward, done, info = env.step(action)
            total_cost += -reward
            total_energy += info["energy_use"]
            total_ramp += abs(info["production_rate"] - previous_rate)
        costs.append(total_cost)
        energy.append(total_energy)
        final_backlog.append(max(-env.inventory, 0.0))
        ramping.append(total_ramp)

    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_energy_use": float(np.mean(energy)),
        "mean_final_backlog": float(np.mean(final_backlog)),
        "mean_total_ramp": float(np.mean(ramping)),
    }


def mpc_grid_policy(
    config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    lookahead: int = 3,
    grid_points: int = 9,
):
    """Short deterministic expected-value lookahead over a production-rate grid."""

    env = EnergyProductionEnv(config)
    grid = np.linspace(0.0, config.max_rate, int(grid_points))

    def choose(obs: np.ndarray) -> float:
        t = int(round(float(obs[0]) * config.horizon))
        inventory = float(obs[1]) * max(config.max_inventory, config.max_backlog)
        price = float(obs[3]) * (config.base_price + config.price_amplitude + 1e-6)
        previous_rate = float(obs[4]) * config.max_rate

        best_cost = np.inf
        best_rate = 0.0

        def recurse(step, inv, prev_rate, accumulated, first_rate):
            nonlocal best_cost, best_rate
            if step >= lookahead or t + step >= config.horizon:
                if accumulated < best_cost:
                    best_cost = accumulated
                    best_rate = first_rate
                return

            period = t + step
            expected_demand = env.expected_demand(period)
            expected_price = price if step == 0 else env.expected_price(period)

            for rate in grid:
                next_inv, reward, _ = env.transition(
                    inv,
                    prev_rate,
                    float(rate),
                    expected_demand,
                    expected_price,
                )
                new_cost = accumulated - reward
                if new_cost >= best_cost:
                    continue
                recurse(
                    step + 1,
                    next_inv,
                    float(rate),
                    new_cost,
                    float(rate) if step == 0 else first_rate,
                )

        recurse(0, inventory, previous_rate, 0.0, 0.0)
        normalized = 2.0 * best_rate / config.max_rate - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    return choose


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 1.5)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mean)


class QCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.net(torch.cat([state, action], dim=-1))


@dataclass(frozen=True)
class SACConfig:
    total_steps: int = 30_000
    warmup_steps: int = 1_000
    batch_size: int = 128
    replay_size: int = 50_000
    learning_rate: float = 3e-4
    gamma: float = 1.0
    tau: float = 0.01
    alpha: float = 0.12
    updates_per_step: int = 1
    seed: int = 303


class SACProductionAgent:
    def __init__(
        self,
        env: EnergyProductionEnv,
        config: SACConfig = SACConfig(),
    ):
        self.env = env
        self.config = config
        set_global_seed(config.seed)

        self.actor = GaussianActor(env.state_dim)
        self.q1 = QCritic(env.state_dim)
        self.q2 = QCritic(env.state_dim)
        self.target_q1 = QCritic(env.state_dim)
        self.target_q2 = QCritic(env.state_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.learning_rate)

        self.replay = deque(maxlen=config.replay_size)
        self.rng = np.random.default_rng(config.seed)

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module):
        for s, t in zip(source.parameters(), target.parameters()):
            t.data.mul_(1.0 - self.config.tau).add_(self.config.tau * s.data)

    def _update(self):
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return

        idx = self.rng.integers(0, len(self.replay), size=self.config.batch_size)
        batch = [self.replay[int(i)] for i in idx]

        states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q = torch.min(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions),
            )
            target = rewards + (1.0 - dones) * self.config.gamma * (
                target_q - self.config.alpha * next_log_prob
            )

        q1_loss = (self.q1(states, actions) - target).pow(2).mean()
        q2_loss = (self.q2(states, actions) - target).pow(2).mean()

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
        self.q2_opt.step()

        new_actions, log_prob, _ = self.actor.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
        actor_loss = (self.config.alpha * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

    def fit(self):
        obs = self.env.reset(seed=self.config.seed)
        for step in range(self.config.total_steps):
            if step < self.config.warmup_steps:
                action = np.asarray([self.rng.uniform(-1.0, 1.0)], dtype=np.float32)
            else:
                with torch.no_grad():
                    action, _, _ = self.actor.sample(
                        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    )
                action = action[0].cpu().numpy()

            next_obs, reward, done, _ = self.env.step(action)
            self.replay.append(
                (
                    obs.copy(),
                    np.asarray(action, dtype=np.float32).copy(),
                    float(reward),
                    next_obs.copy(),
                    bool(done),
                )
            )
            obs = next_obs

            for _ in range(self.config.updates_per_step):
                self._update()

            if done:
                obs = self.env.reset(seed=self.config.seed + step + 1)

        return self

    def policy(self):
        def choose(obs: np.ndarray) -> float:
            with torch.no_grad():
                _, _, action = self.actor.sample(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                )
            return float(action.item())

        return choose
