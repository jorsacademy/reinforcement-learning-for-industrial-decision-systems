from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .neural_common import set_global_seed


@dataclass(frozen=True)
class EnergyProductionConfig:
    """Continuous-rate production planning under stochastic demand and energy prices."""

    horizon: int = 18
    max_inventory: float = 20.0
    max_backlog: float = 20.0
    max_production: float = 12.0
    initial_inventory: float = 2.0
    demand_base: float = 6.0
    demand_amplitude: float = 1.4
    production_cost: float = 1.10
    energy_linear: float = 0.55
    energy_quadratic: float = 0.025
    holding_cost: float = 0.45
    backlog_cost: float = 5.50
    price_mean: float = 1.0
    price_rho: float = 0.78
    price_sigma: float = 0.16
    price_min: float = 0.25
    price_max: float = 2.0
    initial_price: float = 1.0

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if min(self.max_inventory, self.max_backlog, self.max_production) <= 0:
            raise ValueError("inventory/backlog/production limits must be positive")
        if not -self.max_backlog <= self.initial_inventory <= self.max_inventory:
            raise ValueError("initial inventory outside state bounds")
        if not 0 < self.price_min < self.price_max:
            raise ValueError("invalid energy-price bounds")
        if not self.price_min <= self.initial_price <= self.price_max:
            raise ValueError("initial price outside bounds")
        if not 0 <= self.price_rho < 1:
            raise ValueError("price_rho must be in [0,1)")


class EnergyProductionEnv:
    """Finite-horizon energy-aware production environment with continuous action."""

    def __init__(self, config: EnergyProductionConfig = EnergyProductionConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.net_inventory = float(config.initial_inventory)
        self.energy_price = float(config.initial_price)

    @property
    def state_dim(self) -> int:
        return 4

    def demand_mean(self, t: int | None = None) -> float:
        period = self.t if t is None else int(t)
        phase = 2.0 * np.pi * period / max(self.config.horizon, 1)
        return float(
            max(
                0.25,
                self.config.demand_base
                + self.config.demand_amplitude * np.sin(phase)
                + 0.5 * self.config.demand_amplitude * np.cos(2.0 * phase + 0.3),
            )
        )

    def observation(self) -> np.ndarray:
        scale_inv = max(self.config.max_inventory, self.config.max_backlog)
        price_scaled = (
            (self.energy_price - self.config.price_min)
            / (self.config.price_max - self.config.price_min)
        )
        return np.asarray(
            [
                self.t / self.config.horizon,
                self.net_inventory / scale_inv,
                price_scaled,
                self.demand_mean() / max(self.config.max_production, 1e-9),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.net_inventory = float(self.config.initial_inventory)
        self.energy_price = float(self.config.initial_price)
        return self.observation()

    def energy_consumption(self, production: float) -> float:
        q = float(production)
        return self.config.energy_linear * q + self.config.energy_quadratic * q * q

    def transition(
        self,
        net_inventory: float,
        energy_price: float,
        production: float,
        demand: float,
    ) -> tuple[float, float, dict]:
        q = float(np.clip(production, 0.0, self.config.max_production))
        raw_next = float(net_inventory) + q - float(demand)
        next_inventory = float(
            np.clip(raw_next, -self.config.max_backlog, self.config.max_inventory)
        )
        overflow_backlog = max(-self.config.max_backlog - raw_next, 0.0)
        overflow_inventory = max(raw_next - self.config.max_inventory, 0.0)

        energy = self.energy_consumption(q)
        production_cost = self.config.production_cost * q
        energy_cost = float(energy_price) * energy
        inventory_cost = self.config.holding_cost * max(next_inventory, 0.0)
        backlog_cost = self.config.backlog_cost * max(-next_inventory, 0.0)
        overflow_cost = 8.0 * overflow_backlog + 1.0 * overflow_inventory
        cost = (
            production_cost
            + energy_cost
            + inventory_cost
            + backlog_cost
            + overflow_cost
        )
        info = {
            "cost": float(cost),
            "production": q,
            "demand": float(demand),
            "energy_price": float(energy_price),
            "energy": float(energy),
            "energy_cost": float(energy_cost),
            "net_inventory": next_inventory,
            "backlog": max(-next_inventory, 0.0),
        }
        return next_inventory, -float(cost), info

    def _next_price(self) -> float:
        shock = self.config.price_sigma * self.rng.normal()
        next_price = (
            self.config.price_mean
            + self.config.price_rho * (self.energy_price - self.config.price_mean)
            + shock
        )
        return float(np.clip(next_price, self.config.price_min, self.config.price_max))

    def step(self, production: float):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode already finished")
        demand = float(self.rng.poisson(self.demand_mean()))
        next_inventory, reward, info = self.transition(
            self.net_inventory,
            self.energy_price,
            production,
            demand,
        )
        self.net_inventory = next_inventory
        self.energy_price = self._next_price()
        self.t += 1
        done = self.t >= self.config.horizon
        return self.observation(), reward, done, info


def myopic_energy_policy(
    config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    grid_step: float = 0.25,
) -> Callable[[int, float, float], float]:
    env = EnergyProductionEnv(config)
    grid = np.arange(0.0, config.max_production + grid_step / 2.0, grid_step)

    def choose(t: int, inventory: float, energy_price: float) -> float:
        mean_demand = env.demand_mean(t)
        best_cost = np.inf
        best_q = 0.0
        for q in grid:
            _, reward, _ = env.transition(inventory, energy_price, float(q), mean_demand)
            if -reward < best_cost:
                best_cost = -reward
                best_q = float(q)
        return best_q

    return choose


def rolling_horizon_mpc_policy(
    config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    lookahead: int = 3,
    grid_points: int = 7,
) -> Callable[[int, float, float], float]:
    """Deterministic rolling-horizon grid MPC using mean demand and price forecasts."""

    if lookahead < 1 or grid_points < 2:
        raise ValueError("lookahead >=1 and grid_points >=2 required")
    env = EnergyProductionEnv(config)
    grid = np.linspace(0.0, config.max_production, grid_points)

    def choose(t: int, inventory: float, energy_price: float) -> float:
        remaining = config.horizon - int(t)
        horizon = min(lookahead, remaining)
        best_cost = np.inf
        best_first = 0.0

        for sequence in product(grid, repeat=horizon):
            inv = float(inventory)
            price = float(energy_price)
            total = 0.0
            for k, q in enumerate(sequence):
                mean_demand = env.demand_mean(int(t) + k)
                inv, reward, _ = env.transition(inv, price, float(q), mean_demand)
                total += -reward
                price = (
                    config.price_mean
                    + config.price_rho * (price - config.price_mean)
                )
            total += 1.5 * config.backlog_cost * max(-inv, 0.0)
            if total < best_cost - 1e-12:
                best_cost = total
                best_first = float(sequence[0])
        return best_first

    return choose


def evaluate_energy_policy(
    policy: Callable[[int, float, float], float],
    config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    episodes: int = 1000,
    seed: int = 90_000,
) -> dict[str, float]:
    env = EnergyProductionEnv(config)
    costs, energy_costs, backlog, production = [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_cost = 0.0
        total_energy_cost = 0.0
        total_production = 0.0
        done = False
        while not done:
            action = float(policy(env.t, env.net_inventory, env.energy_price))
            _, reward, done, info = env.step(action)
            total_cost += -reward
            total_energy_cost += info["energy_cost"]
            total_production += info["production"]
        costs.append(total_cost)
        energy_costs.append(total_energy_cost)
        backlog.append(max(-env.net_inventory, 0.0))
        production.append(total_production)
    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_energy_cost": float(np.mean(energy_costs)),
        "mean_final_backlog": float(np.mean(backlog)),
        "mean_total_production": float(np.mean(production)),
    }


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int):
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i] = state
        self.actions[i, 0] = float(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.states[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_states[idx]),
            torch.as_tensor(self.dones[idx]),
        )


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        h = self.body(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1), torch.tanh(mu)


class Critic(nn.Module):
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
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class SACConfig:
    episodes: int = 1800
    replay_size: int = 50_000
    batch_size: int = 128
    warmup_steps: int = 600
    learning_rate: float = 3e-4
    gamma: float = 1.0
    tau: float = 0.01
    entropy_alpha: float = 0.12
    updates_per_step: int = 1
    seed: int = 303


class SACEnergyAgent:
    """Compact Soft Actor-Critic implementation for one continuous production action."""

    def __init__(self, env: EnergyProductionEnv, config: SACConfig = SACConfig()):
        self.env = env
        self.config = config
        set_global_seed(config.seed)
        self.actor = SquashedGaussianActor(env.state_dim)
        self.q1 = Critic(env.state_dim)
        self.q2 = Critic(env.state_dim)
        self.target_q1 = Critic(env.state_dim)
        self.target_q2 = Critic(env.state_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_size, env.state_dim, config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.total_steps = 0

    def _scaled_env_action(self, normalized_action: float) -> float:
        return float((normalized_action + 1.0) * 0.5 * self.env.config.max_production)

    def _normalized_action(self, production: float) -> float:
        return float(2.0 * production / self.env.config.max_production - 1.0)

    @torch.no_grad()
    def _soft_update(self):
        for target, source in (
            (self.target_q1, self.q1),
            (self.target_q2, self.q2),
        ):
            for t_param, s_param in zip(target.parameters(), source.parameters()):
                t_param.data.mul_(1.0 - self.config.tau)
                t_param.data.add_(self.config.tau * s_param.data)

    def _update(self):
        if self.replay.size < max(self.config.warmup_steps, self.config.batch_size):
            return
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size
        )

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q = torch.min(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions),
            ) - self.config.entropy_alpha * next_log_prob
            target = rewards + (1.0 - dones) * self.config.gamma * target_q

        q1_loss = F.mse_loss(self.q1(states, actions), target)
        q2_loss = F.mse_loss(self.q2(states, actions), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
        self.q2_opt.step()

        sampled_actions, log_prob, _ = self.actor.sample(states)
        actor_loss = (
            self.config.entropy_alpha * log_prob
            - torch.min(
                self.q1(states, sampled_actions),
                self.q2(states, sampled_actions),
            )
        ).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        self._soft_update()

    def fit(self):
        for ep in range(self.config.episodes):
            obs = self.env.reset(seed=self.config.seed + ep)
            done = False
            while not done:
                if self.total_steps < self.config.warmup_steps:
                    production = float(
                        self.rng.uniform(0.0, self.env.config.max_production)
                    )
                    normalized = self._normalized_action(production)
                else:
                    state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action_t, _, _ = self.actor.sample(state_t)
                    normalized = float(action_t.item())
                    production = self._scaled_env_action(normalized)

                next_obs, reward, done, _ = self.env.step(production)
                self.replay.add(obs, normalized, reward, next_obs, done)
                obs = next_obs
                self.total_steps += 1
                for _ in range(self.config.updates_per_step):
                    self._update()
        return self

    def policy(self):
        config = self.env.config

        def choose(t: int, inventory: float, energy_price: float) -> float:
            scale_inv = max(config.max_inventory, config.max_backlog)
            price_scaled = (
                (energy_price - config.price_min)
                / (config.price_max - config.price_min)
            )
            obs = np.asarray(
                [
                    t / config.horizon,
                    inventory / scale_inv,
                    price_scaled,
                    self.env.demand_mean(t) / config.max_production,
                ],
                dtype=np.float32,
            )
            with torch.no_grad():
                _, _, mean_action = self.actor.sample(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                )
            return self._scaled_env_action(float(mean_action.item()))

        return choose
