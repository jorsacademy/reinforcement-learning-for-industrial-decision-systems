from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .neural_common import set_global_seed


@dataclass(frozen=True)
class WorkforceConfig:
    horizon: int = 12
    n_centers: int = 3
    max_backlog: int = 24
    base_capacity: tuple[int, int, int] = (4, 4, 4)
    flex_pool: int = 3
    flex_productivity: int = 2
    mean_workload: tuple[float, float, float] = (5.0, 4.5, 5.5)
    backlog_cost: tuple[float, float, float] = (4.0, 5.0, 6.0)
    flex_worker_cost: float = 2.0
    surge_probability: float = 0.12
    surge_size: int = 3

    def validate(self) -> None:
        if self.horizon < 1 or self.n_centers != 3:
            raise ValueError("this benchmark requires three work centers and positive horizon")
        if self.max_backlog < 1 or self.flex_pool < 0 or self.flex_productivity < 1:
            raise ValueError("invalid backlog/flex parameters")
        if len(self.base_capacity) != 3 or len(self.mean_workload) != 3 or len(self.backlog_cost) != 3:
            raise ValueError("center parameter lengths must equal three")
        if any(x < 0 for x in self.base_capacity) or any(x < 0 for x in self.mean_workload):
            raise ValueError("capacity/workload parameters must be non-negative")


def workforce_allocations(flex_pool: int) -> tuple[tuple[int, int, int], ...]:
    allocations = []
    for a, b, c in product(range(flex_pool + 1), repeat=3):
        if a + b + c <= flex_pool:
            allocations.append((a, b, c))
    return tuple(allocations)


class WorkforceEnv:
    """Dynamic flex-workforce allocation across three stochastic work centers."""

    def __init__(self, config: WorkforceConfig = WorkforceConfig()):
        config.validate()
        self.config = config
        self.actions = workforce_allocations(config.flex_pool)
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.backlog = np.zeros(3, dtype=int)

    @property
    def state_dim(self) -> int:
        return 7

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def expected_workload(self, t: int | None = None) -> np.ndarray:
        period = self.t if t is None else int(t)
        phase = 2.0 * np.pi * period / max(self.config.horizon, 1)
        seasonal = np.asarray(
            [
                0.8 * np.sin(phase),
                0.6 * np.cos(phase + 0.4),
                0.9 * np.sin(phase + 1.0),
            ],
            dtype=float,
        )
        return np.maximum(np.asarray(self.config.mean_workload) + seasonal, 0.2)

    def observation(self) -> np.ndarray:
        means = self.expected_workload()
        max_mean = max(max(self.config.mean_workload) + 1.0, 1.0)
        return np.concatenate(
            [
                np.asarray([self.t / self.config.horizon], dtype=np.float32),
                (self.backlog / self.config.max_backlog).astype(np.float32),
                (means / max_mean).astype(np.float32),
            ]
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.backlog = np.zeros(3, dtype=int)
        return self.observation()

    def sample_workload(self) -> np.ndarray:
        means = self.expected_workload()
        workload = self.rng.poisson(means).astype(int)
        if self.rng.random() < self.config.surge_probability:
            center = int(self.rng.integers(0, 3))
            workload[center] += self.config.surge_size
        return workload

    def transition(self, backlog: np.ndarray, action_index: int, workload: np.ndarray):
        if not 0 <= int(action_index) < self.n_actions:
            raise ValueError("invalid workforce allocation action")
        allocation = np.asarray(self.actions[int(action_index)], dtype=int)
        capacity = np.asarray(self.config.base_capacity, dtype=int) + self.config.flex_productivity * allocation
        available_work = np.asarray(backlog, dtype=int) + np.asarray(workload, dtype=int)
        processed = np.minimum(available_work, capacity)
        next_backlog = np.clip(available_work - processed, 0, self.config.max_backlog).astype(int)
        overflow = np.maximum(available_work - processed - self.config.max_backlog, 0)

        backlog_cost = float(np.dot(np.asarray(self.config.backlog_cost), next_backlog))
        overflow_cost = float(10.0 * overflow.sum())
        labor_cost = float(self.config.flex_worker_cost * allocation.sum())
        cost = backlog_cost + overflow_cost + labor_cost
        info = {
            "cost": cost,
            "allocation": tuple(int(x) for x in allocation),
            "processed": tuple(int(x) for x in processed),
            "workload": tuple(int(x) for x in workload),
            "backlog": tuple(int(x) for x in next_backlog),
            "flex_used": int(allocation.sum()),
        }
        return next_backlog, -cost, info

    def step(self, action_index: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode finished")
        workload = self.sample_workload()
        next_backlog, reward, info = self.transition(self.backlog, action_index, workload)
        self.backlog = next_backlog
        self.t += 1
        done = self.t >= self.config.horizon
        return self.observation(), reward, done, info


def myopic_expected_cost_policy(config: WorkforceConfig = WorkforceConfig()):
    """One-step expected-workload allocation baseline."""
    env = WorkforceEnv(config)

    def choose(t: int, backlog: np.ndarray) -> int:
        means = env.expected_workload(t)
        best_cost = np.inf
        best_action = 0
        for action_index in range(env.n_actions):
            allocation = np.asarray(env.actions[action_index], dtype=int)
            capacity = np.asarray(config.base_capacity) + config.flex_productivity * allocation
            expected_end = np.maximum(np.asarray(backlog, dtype=float) + means - capacity, 0.0)
            proxy = (
                np.dot(np.asarray(config.backlog_cost), expected_end)
                + config.flex_worker_cost * allocation.sum()
            )
            if proxy < best_cost - 1e-12:
                best_cost = float(proxy)
                best_action = action_index
        return int(best_action)

    return choose


def evaluate_workforce_policy(
    policy: Callable[[int, np.ndarray], int],
    config: WorkforceConfig = WorkforceConfig(),
    *,
    episodes: int = 1000,
    seed: int = 60_000,
):
    env = WorkforceEnv(config)
    costs, flex, final_backlog = [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_cost = 0.0
        total_flex = 0
        done = False
        while not done:
            action = int(policy(env.t, env.backlog.copy()))
            _, reward, done, info = env.step(action)
            total_cost += -reward
            total_flex += info["flex_used"]
        costs.append(total_cost)
        flex.append(total_flex / config.horizon)
        final_backlog.append(env.backlog.sum())
    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_flex_workers": float(np.mean(flex)),
        "mean_final_backlog": float(np.mean(final_backlog)),
    }


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor):
        h = self.body(states)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 140
    rollout_episodes: int = 24
    ppo_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 1.0
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    seed: int = 202


class PPOWorkforceAgent:
    def __init__(self, env: WorkforceEnv, config: PPOConfig = PPOConfig()):
        self.env = env
        self.config = config
        set_global_seed(config.seed)
        self.model = ActorCritic(env.state_dim, env.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(config.seed)

    def _collect(self, update: int):
        states, actions, old_log_probs, returns = [], [], [], []
        for episode in range(self.config.rollout_episodes):
            obs = self.env.reset(seed=self.config.seed + update * 10_000 + episode)
            episode_states, episode_actions, episode_logs, episode_rewards = [], [], [], []
            done = False
            while not done:
                state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = self.model(state_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                next_obs, reward, done, _ = self.env.step(int(action.item()))
                episode_states.append(obs.copy())
                episode_actions.append(int(action.item()))
                episode_logs.append(float(log_prob.item()))
                episode_rewards.append(float(reward))
                obs = next_obs

            running = 0.0
            episode_returns = [0.0] * len(episode_rewards)
            for i in range(len(episode_rewards) - 1, -1, -1):
                running = episode_rewards[i] + self.config.gamma * running
                episode_returns[i] = running

            states.extend(episode_states)
            actions.extend(episode_actions)
            old_log_probs.extend(episode_logs)
            returns.extend(episode_returns)

        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(old_log_probs, dtype=torch.float32),
            torch.as_tensor(returns, dtype=torch.float32),
        )

    def fit(self):
        for update in range(self.config.updates):
            states, actions, old_log_probs, returns = self._collect(update)
            with torch.no_grad():
                _, old_values = self.model(states)
                advantages = returns - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            n = len(states)
            for _ in range(self.config.ppo_epochs):
                order = self.rng.permutation(n)
                for start in range(0, n, self.config.minibatch_size):
                    idx = torch.as_tensor(order[start:start + self.config.minibatch_size], dtype=torch.long)
                    logits, values = self.model(states[idx])
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[idx])
                    ratio = torch.exp(new_log_probs - old_log_probs[idx])
                    adv = advantages[idx]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                    value_loss = (values - returns[idx]).square().mean()
                    entropy = dist.entropy().mean()
                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
        return self

    def policy(self):
        config = self.env.config

        def choose(t: int, backlog: np.ndarray) -> int:
            means = self.env.expected_workload(t)
            max_mean = max(max(config.mean_workload) + 1.0, 1.0)
            obs = np.concatenate(
                [
                    np.asarray([t / config.horizon], dtype=np.float32),
                    (np.asarray(backlog, dtype=float) / config.max_backlog).astype(np.float32),
                    (means / max_mean).astype(np.float32),
                ]
            )
            with torch.no_grad():
                logits, _ = self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())

        return choose
