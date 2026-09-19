from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .neural_common import set_global_seed


@dataclass(frozen=True)
class RiskInventoryConfig:
    horizon: int = 12
    max_inventory: int = 30
    max_order: int = 12
    base_demand_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    base_demand_probabilities: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.25,
        0.20,
        0.12,
        0.08,
    )
    surge_probability: float = 0.08
    surge_extra_values: tuple[int, ...] = (5, 8, 11)
    surge_extra_probabilities: tuple[float, ...] = (0.50, 0.30, 0.20)
    order_cost: float = 1.6
    fixed_order_cost: float = 0.4
    holding_cost: float = 0.45
    lost_sales_cost: float = 7.5
    initial_inventory: int = 8

    def validate(self) -> None:
        if self.horizon < 1 or self.max_inventory < 1 or self.max_order < 1:
            raise ValueError("horizon and inventory/action bounds must be positive")
        if not 0 <= self.initial_inventory <= self.max_inventory:
            raise ValueError("initial_inventory outside state range")
        base = np.asarray(self.base_demand_probabilities, dtype=float)
        surge = np.asarray(self.surge_extra_probabilities, dtype=float)
        if len(base) != len(self.base_demand_values) or not np.isclose(base.sum(), 1.0):
            raise ValueError("invalid base demand distribution")
        if len(surge) != len(self.surge_extra_values) or not np.isclose(surge.sum(), 1.0):
            raise ValueError("invalid surge distribution")
        if np.any(base < 0) or np.any(surge < 0):
            raise ValueError("demand probabilities must be non-negative")
        if not 0.0 <= self.surge_probability <= 1.0:
            raise ValueError("surge_probability must be in [0,1]")

    @property
    def max_demand(self) -> int:
        return max(self.base_demand_values) + max(self.surge_extra_values)


class RiskInventoryEnv:
    """Lost-sales inventory environment with rare demand surges."""

    def __init__(self, config: RiskInventoryConfig = RiskInventoryConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.inventory = config.initial_inventory
        self.previous_demand = 0

    @property
    def n_actions(self) -> int:
        return self.config.max_order + 1

    @property
    def state_dim(self) -> int:
        return 3

    def feasible_actions(self, inventory: int | None = None) -> np.ndarray:
        inv = self.inventory if inventory is None else int(inventory)
        upper = min(self.config.max_order, self.config.max_inventory - inv)
        return np.arange(upper + 1, dtype=int)

    def observation(self) -> np.ndarray:
        return np.asarray(
            [
                self.t / self.config.horizon,
                self.inventory / self.config.max_inventory,
                self.previous_demand / self.config.max_demand,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.inventory = self.config.initial_inventory
        self.previous_demand = 0
        return self.observation()

    def demand_pmf(self) -> tuple[np.ndarray, np.ndarray]:
        masses: dict[int, float] = {}
        no_surge = 1.0 - self.config.surge_probability
        for base, p_base in zip(
            self.config.base_demand_values,
            self.config.base_demand_probabilities,
        ):
            masses[int(base)] = masses.get(int(base), 0.0) + no_surge * p_base
            for extra, p_extra in zip(
                self.config.surge_extra_values,
                self.config.surge_extra_probabilities,
            ):
                value = int(base + extra)
                masses[value] = masses.get(value, 0.0) + (
                    self.config.surge_probability * p_base * p_extra
                )
        values = np.asarray(sorted(masses), dtype=int)
        probs = np.asarray([masses[int(v)] for v in values], dtype=float)
        probs = probs / probs.sum()
        return values, probs

    def sample_demand(self) -> int:
        base = int(
            self.rng.choice(
                self.config.base_demand_values,
                p=self.config.base_demand_probabilities,
            )
        )
        if self.rng.random() < self.config.surge_probability:
            base += int(
                self.rng.choice(
                    self.config.surge_extra_values,
                    p=self.config.surge_extra_probabilities,
                )
            )
        return base

    def transition(self, inventory: int, action: int, demand: int):
        action = int(action)
        if action not in self.feasible_actions(inventory):
            raise ValueError("infeasible order quantity")

        available = int(inventory) + action
        sales = min(available, int(demand))
        lost = max(int(demand) - available, 0)
        ending = available - sales

        cost = (
            self.config.order_cost * action
            + self.config.fixed_order_cost * float(action > 0)
            + self.config.holding_cost * ending
            + self.config.lost_sales_cost * lost
        )
        return ending, -float(cost), {
            "cost": float(cost),
            "demand": int(demand),
            "sales": int(sales),
            "lost_sales": int(lost),
            "ending_inventory": int(ending),
            "order": action,
        }

    def step(self, action: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode finished")
        demand = self.sample_demand()
        next_inventory, reward, info = self.transition(
            self.inventory,
            action,
            demand,
        )
        self.inventory = next_inventory
        self.previous_demand = demand
        self.t += 1
        done = self.t >= self.config.horizon
        return self.observation(), reward, done, info


def exact_risk_neutral_dp(
    config: RiskInventoryConfig = RiskInventoryConfig(),
):
    """Exact expected-cost DP for the declared finite surge-demand model."""
    config.validate()
    env = RiskInventoryEnv(config)
    demand_values, demand_probs = env.demand_pmf()

    values = np.zeros(
        (config.horizon + 1, config.max_inventory + 1),
        dtype=float,
    )
    policy = np.zeros(
        (config.horizon, config.max_inventory + 1),
        dtype=int,
    )

    for t in range(config.horizon - 1, -1, -1):
        for inv in range(config.max_inventory + 1):
            best_cost = np.inf
            best_action = 0
            for action in env.feasible_actions(inv):
                expected = 0.0
                for demand, prob in zip(demand_values, demand_probs):
                    next_inv, reward, _ = env.transition(
                        inv,
                        int(action),
                        int(demand),
                    )
                    expected += prob * (
                        -reward + values[t + 1, next_inv]
                    )
                if expected < best_cost - 1e-12:
                    best_cost = expected
                    best_action = int(action)
            values[t, inv] = best_cost
            policy[t, inv] = best_action

    return values, policy


def table_policy(table: np.ndarray):
    arr = np.asarray(table, dtype=int)

    def choose(t: int, inventory: int) -> int:
        return int(arr[t, inventory])

    return choose


def base_stock_policy(
    config: RiskInventoryConfig,
    target: int,
):
    if not 0 <= int(target) <= config.max_inventory:
        raise ValueError("target outside inventory range")

    def choose(t: int, inventory: int) -> int:
        del t
        return int(
            min(
                config.max_order,
                max(int(target) - int(inventory), 0),
            )
        )

    return choose


def empirical_cvar(
    costs: np.ndarray | list[float],
    alpha: float = 0.90,
) -> float:
    """Empirical upper-tail CVaR for a loss/cost sample."""
    a = np.asarray(costs, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("costs must be a non-empty 1D sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")

    var = float(np.quantile(a, alpha))
    tail = a[a >= var - 1e-12]
    return float(tail.mean())


def evaluate_risk_inventory_policy(
    policy: Callable[[int, int], int],
    config: RiskInventoryConfig = RiskInventoryConfig(),
    *,
    episodes: int = 1000,
    seed: int = 130_000,
    cvar_alpha: float = 0.90,
) -> dict[str, float]:
    env = RiskInventoryEnv(config)
    costs = []
    fill_rates = []
    stockout_rates = []

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_cost = 0.0
        total_sales = 0
        total_demand = 0
        stockout_periods = 0
        done = False

        while not done:
            action = int(policy(env.t, env.inventory))
            _, reward, done, info = env.step(action)
            total_cost += -reward
            total_sales += info["sales"]
            total_demand += info["demand"]
            stockout_periods += int(info["lost_sales"] > 0)

        costs.append(total_cost)
        fill_rates.append(total_sales / max(total_demand, 1))
        stockout_rates.append(stockout_periods / config.horizon)

    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "p95_cost": float(np.quantile(a, 0.95)),
        "cvar90_cost": empirical_cvar(a, cvar_alpha),
        "mean_fill_rate": float(np.mean(fill_rates)),
        "mean_stockout_period_rate": float(np.mean(stockout_rates)),
    }


def optimize_base_stock_target(
    config: RiskInventoryConfig = RiskInventoryConfig(),
    *,
    objective: str = "mean",
    episodes: int = 1000,
    seed: int = 125_000,
    cvar_alpha: float = 0.90,
) -> tuple[int, dict[str, float]]:
    """Common-random-number search over base-stock targets."""
    if objective not in {"mean", "cvar"}:
        raise ValueError("objective must be 'mean' or 'cvar'")

    best_target = 0
    best_score = np.inf
    best_metrics = None

    for target in range(config.max_inventory + 1):
        metrics = evaluate_risk_inventory_policy(
            base_stock_policy(config, target),
            config,
            episodes=episodes,
            seed=seed,
            cvar_alpha=cvar_alpha,
        )
        score = (
            metrics["mean_cost"]
            if objective == "mean"
            else metrics["cvar90_cost"]
        )
        if score < best_score - 1e-12:
            best_score = float(score)
            best_target = target
            best_metrics = metrics

    assert best_metrics is not None
    return best_target, best_metrics


class RiskActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
    ):
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
class RiskPPOConfig:
    updates: int = 140
    rollout_episodes: int = 32
    ppo_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 1.0
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    cvar_alpha: float = 0.90
    tail_weight: float = 4.0
    seed: int = 505


class TailWeightedPPOAgent:
    """PPO with empirical worst-tail episode weighting.

    This is a CVaR-focused training heuristic: transitions from episodes in the
    empirical upper-cost tail receive larger policy-gradient weight. It is not
    presented as an exact optimizer of the dynamic CVaR control problem.
    """

    def __init__(
        self,
        env: RiskInventoryEnv,
        config: RiskPPOConfig = RiskPPOConfig(),
    ):
        self.env = env
        self.config = config
        set_global_seed(config.seed)
        self.model = RiskActorCritic(
            env.state_dim,
            env.n_actions,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)

    def _masked_distribution(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
    ):
        logits, values = self.model(states)
        masked_logits = logits.masked_fill(~masks, -1e9)
        return Categorical(logits=masked_logits), values

    def _collect(self, update: int):
        states = []
        masks = []
        actions = []
        old_log_probs = []
        returns = []
        episode_ids = []
        episode_costs = []

        for episode in range(self.config.rollout_episodes):
            obs = self.env.reset(
                seed=self.config.seed + update * 10_000 + episode
            )
            ep_states = []
            ep_masks = []
            ep_actions = []
            ep_logs = []
            ep_rewards = []
            done = False

            while not done:
                feasible = self.env.feasible_actions()
                mask = np.zeros(self.env.n_actions, dtype=bool)
                mask[feasible] = True

                state_t = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                ).unsqueeze(0)
                mask_t = torch.as_tensor(
                    mask,
                    dtype=torch.bool,
                ).unsqueeze(0)

                with torch.no_grad():
                    dist, _ = self._masked_distribution(
                        state_t,
                        mask_t,
                    )
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_obs, reward, done, _ = self.env.step(
                    int(action.item())
                )

                ep_states.append(obs.copy())
                ep_masks.append(mask.copy())
                ep_actions.append(int(action.item()))
                ep_logs.append(float(log_prob.item()))
                ep_rewards.append(float(reward))
                obs = next_obs

            running = 0.0
            ep_returns = [0.0] * len(ep_rewards)
            for i in range(len(ep_rewards) - 1, -1, -1):
                running = (
                    ep_rewards[i]
                    + self.config.gamma * running
                )
                ep_returns[i] = running

            states.extend(ep_states)
            masks.extend(ep_masks)
            actions.extend(ep_actions)
            old_log_probs.extend(ep_logs)
            returns.extend(ep_returns)
            episode_ids.extend([episode] * len(ep_rewards))
            episode_costs.append(-sum(ep_rewards))

        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(np.asarray(masks), dtype=torch.bool),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(old_log_probs, dtype=torch.float32),
            torch.as_tensor(returns, dtype=torch.float32),
            np.asarray(episode_ids, dtype=int),
            np.asarray(episode_costs, dtype=float),
        )

    def fit(self):
        for update in range(self.config.updates):
            (
                states,
                masks,
                actions,
                old_log_probs,
                returns,
                episode_ids,
                episode_costs,
            ) = self._collect(update)

            with torch.no_grad():
                _, values = self._masked_distribution(states, masks)
                advantages = returns - values
                advantages = (
                    advantages - advantages.mean()
                ) / (advantages.std(unbiased=False) + 1e-8)

            var = float(
                np.quantile(
                    episode_costs,
                    self.config.cvar_alpha,
                )
            )
            episode_weights = np.ones(
                len(episode_costs),
                dtype=np.float32,
            )
            episode_weights[
                episode_costs >= var - 1e-12
            ] += float(self.config.tail_weight)

            transition_weights = torch.as_tensor(
                episode_weights[episode_ids],
                dtype=torch.float32,
            )
            transition_weights = (
                transition_weights / transition_weights.mean()
            )

            n = len(states)
            for _ in range(self.config.ppo_epochs):
                order = self.rng.permutation(n)
                for start in range(
                    0,
                    n,
                    self.config.minibatch_size,
                ):
                    idx = torch.as_tensor(
                        order[
                            start:
                            start + self.config.minibatch_size
                        ],
                        dtype=torch.long,
                    )

                    dist, values = self._masked_distribution(
                        states[idx],
                        masks[idx],
                    )
                    new_log_probs = dist.log_prob(actions[idx])
                    ratio = torch.exp(
                        new_log_probs - old_log_probs[idx]
                    )
                    adv = advantages[idx]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )

                    surrogate = torch.min(
                        ratio * adv,
                        clipped * adv,
                    )
                    policy_loss = -(
                        transition_weights[idx] * surrogate
                    ).mean()
                    value_loss = (
                        values - returns[idx]
                    ).square().mean()
                    entropy = dist.entropy().mean()

                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        5.0,
                    )
                    self.optimizer.step()

        return self

    def policy(self):
        config = self.env.config

        def choose(t: int, inventory: int) -> int:
            obs = np.asarray(
                [
                    t / config.horizon,
                    inventory / config.max_inventory,
                    0.0,
                ],
                dtype=np.float32,
            )
            feasible = self.env.feasible_actions(inventory)
            mask = np.zeros(self.env.n_actions, dtype=bool)
            mask[feasible] = True

            state_t = torch.as_tensor(
                obs,
                dtype=torch.float32,
            ).unsqueeze(0)
            mask_t = torch.as_tensor(
                mask,
                dtype=torch.bool,
            ).unsqueeze(0)

            with torch.no_grad():
                dist, _ = self._masked_distribution(
                    state_t,
                    mask_t,
                )
            return int(torch.argmax(dist.logits, dim=-1).item())

        return choose
