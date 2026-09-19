from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .neural_common import set_global_seed
from .ppo_workforce import WorkforceConfig, WorkforceEnv, workforce_allocations


@dataclass(frozen=True)
class SafeWorkforceConfig:
    workforce: WorkforceConfig = WorkforceConfig()
    service_backlog_limit: int = 4
    expected_violation_budget: float = 4.0
    soft_backlog_cost: float = 0.20

    def validate(self) -> None:
        self.workforce.validate()
        if self.service_backlog_limit < 0:
            raise ValueError("service_backlog_limit must be non-negative")
        if self.expected_violation_budget < 0:
            raise ValueError("expected_violation_budget must be non-negative")
        if self.soft_backlog_cost < 0:
            raise ValueError("soft_backlog_cost must be non-negative")


class SafeWorkforceEnv:
    """CMDP for flex-workforce allocation with an explicit service constraint.

    Reward represents the economic objective only. Constraint cost is reported
    separately and equals one when any work center ends the period above the
    declared backlog service limit.
    """

    def __init__(self, config: SafeWorkforceConfig = SafeWorkforceConfig()):
        config.validate()
        self.config = config
        self.base_env = WorkforceEnv(config.workforce)
        self.actions = workforce_allocations(config.workforce.flex_pool)

    @property
    def state_dim(self) -> int:
        return self.base_env.state_dim

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def t(self) -> int:
        return self.base_env.t

    @property
    def backlog(self) -> np.ndarray:
        return self.base_env.backlog

    def expected_workload(self, t: int | None = None) -> np.ndarray:
        return self.base_env.expected_workload(t)

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        return self.base_env.reset(seed=seed)

    def transition(
        self,
        backlog: np.ndarray,
        action_index: int,
        workload: np.ndarray,
    ):
        if not 0 <= int(action_index) < self.n_actions:
            raise ValueError("invalid workforce allocation action")

        allocation = np.asarray(self.actions[int(action_index)], dtype=int)
        capacity = (
            np.asarray(self.config.workforce.base_capacity, dtype=int)
            + self.config.workforce.flex_productivity * allocation
        )
        available_work = np.asarray(backlog, dtype=int) + np.asarray(workload, dtype=int)
        processed = np.minimum(available_work, capacity)
        next_backlog = np.clip(
            available_work - processed,
            0,
            self.config.workforce.max_backlog,
        ).astype(int)
        overflow = np.maximum(
            available_work - processed - self.config.workforce.max_backlog,
            0,
        )

        labor_cost = float(
            self.config.workforce.flex_worker_cost * allocation.sum()
        )
        soft_backlog = float(
            self.config.soft_backlog_cost * next_backlog.sum()
        )
        overflow_cost = float(10.0 * overflow.sum())
        economic_cost = labor_cost + soft_backlog + overflow_cost

        constraint_cost = float(
            np.any(next_backlog > self.config.service_backlog_limit)
        )

        info = {
            "economic_cost": economic_cost,
            "constraint_cost": constraint_cost,
            "allocation": tuple(int(x) for x in allocation),
            "workload": tuple(int(x) for x in workload),
            "backlog": tuple(int(x) for x in next_backlog),
            "flex_used": int(allocation.sum()),
            "service_violation": bool(constraint_cost > 0.0),
        }
        return next_backlog, -economic_cost, constraint_cost, info

    def step(self, action_index: int):
        if self.t >= self.config.workforce.horizon:
            raise RuntimeError("episode finished")

        workload = self.base_env.sample_workload()
        next_backlog, reward, constraint_cost, info = self.transition(
            self.backlog,
            action_index,
            workload,
        )
        self.base_env.backlog = next_backlog
        self.base_env.t += 1
        done = self.base_env.t >= self.config.workforce.horizon
        return self.base_env.observation(), reward, constraint_cost, done, info


def constrained_expected_workload_policy(
    config: SafeWorkforceConfig = SafeWorkforceConfig(),
):
    """Minimum-flex one-step policy satisfying the expected backlog limit if possible."""

    env = SafeWorkforceEnv(config)

    def choose(t: int, backlog: np.ndarray) -> int:
        expected = env.expected_workload(t)
        best_feasible = None
        best_flex = np.inf
        best_proxy = np.inf
        fallback = 0
        fallback_violation = np.inf
        fallback_proxy = np.inf

        for action_index, action in enumerate(env.actions):
            allocation = np.asarray(action, dtype=int)
            capacity = (
                np.asarray(config.workforce.base_capacity, dtype=float)
                + config.workforce.flex_productivity * allocation
            )
            expected_end = np.maximum(
                np.asarray(backlog, dtype=float) + expected - capacity,
                0.0,
            )

            flex = int(allocation.sum())
            proxy = (
                config.workforce.flex_worker_cost * flex
                + config.soft_backlog_cost * expected_end.sum()
            )
            excess = float(
                np.maximum(
                    expected_end - config.service_backlog_limit,
                    0.0,
                ).sum()
            )

            if np.all(expected_end <= config.service_backlog_limit + 1e-12):
                if flex < best_flex or (
                    flex == best_flex and proxy < best_proxy - 1e-12
                ):
                    best_feasible = action_index
                    best_flex = flex
                    best_proxy = float(proxy)

            if excess < fallback_violation - 1e-12 or (
                abs(excess - fallback_violation) <= 1e-12
                and proxy < fallback_proxy - 1e-12
            ):
                fallback = action_index
                fallback_violation = excess
                fallback_proxy = float(proxy)

        return int(fallback if best_feasible is None else best_feasible)

    return choose


def shield_action(
    proposed_action: int,
    *,
    t: int,
    backlog: np.ndarray,
    config: SafeWorkforceConfig,
) -> int:
    """Repair an action when expected next backlog violates the service limit.

    The shield only uses current backlog and expected next-period workload.
    If the proposed action is expected-feasible, it is preserved. Otherwise the
    shield selects the lowest-flex action that is expected-feasible. If no
    action can satisfy the limit, it minimizes expected violation magnitude.
    """

    env = SafeWorkforceEnv(config)
    expected = env.expected_workload(t)

    def projected(action_index: int):
        allocation = np.asarray(env.actions[int(action_index)], dtype=float)
        capacity = (
            np.asarray(config.workforce.base_capacity, dtype=float)
            + config.workforce.flex_productivity * allocation
        )
        expected_end = np.maximum(
            np.asarray(backlog, dtype=float) + expected - capacity,
            0.0,
        )
        excess = float(
            np.maximum(
                expected_end - config.service_backlog_limit,
                0.0,
            ).sum()
        )
        return expected_end, excess, int(allocation.sum())

    proposed_end, proposed_excess, _ = projected(int(proposed_action))
    if np.all(proposed_end <= config.service_backlog_limit + 1e-12):
        return int(proposed_action)

    candidates = []
    fallbacks = []
    for action_index in range(env.n_actions):
        end, excess, flex = projected(action_index)
        if np.all(end <= config.service_backlog_limit + 1e-12):
            candidates.append((flex, action_index))
        fallbacks.append((excess, flex, action_index))

    if candidates:
        candidates.sort()
        return int(candidates[0][1])

    fallbacks.sort()
    return int(fallbacks[0][2])


def evaluate_safe_workforce_policy(
    policy: Callable[[int, np.ndarray], int],
    config: SafeWorkforceConfig = SafeWorkforceConfig(),
    *,
    episodes: int = 1000,
    seed: int = 110_000,
    use_shield: bool = False,
) -> dict[str, float]:
    env = SafeWorkforceEnv(config)
    economic_costs = []
    violations = []
    any_violation = []
    flex = []
    final_backlog = []
    shield_interventions = 0
    decisions = 0

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_cost = 0.0
        total_violations = 0.0
        total_flex = 0
        episode_violation = False
        done = False

        while not done:
            action = int(policy(env.t, env.backlog.copy()))
            if use_shield:
                repaired = shield_action(
                    action,
                    t=env.t,
                    backlog=env.backlog.copy(),
                    config=config,
                )
                shield_interventions += int(repaired != action)
                action = repaired
            decisions += 1

            _, reward, constraint_cost, done, info = env.step(action)
            total_cost += -reward
            total_violations += constraint_cost
            total_flex += info["flex_used"]
            episode_violation = episode_violation or info["service_violation"]

        economic_costs.append(total_cost)
        violations.append(total_violations)
        any_violation.append(float(episode_violation))
        flex.append(total_flex / config.workforce.horizon)
        final_backlog.append(float(env.backlog.sum()))

    a = np.asarray(economic_costs, dtype=float)
    v = np.asarray(violations, dtype=float)

    return {
        "mean_economic_cost": float(a.mean()),
        "p90_economic_cost": float(np.quantile(a, 0.90)),
        "mean_violation_periods": float(v.mean()),
        "constraint_gap": float(
            v.mean() - config.expected_violation_budget
        ),
        "episode_violation_probability": float(np.mean(any_violation)),
        "mean_flex_workers": float(np.mean(flex)),
        "mean_final_backlog": float(np.mean(final_backlog)),
        "shield_intervention_rate": float(
            shield_interventions / max(decisions, 1)
        ),
    }


class DualActorCritic(nn.Module):
    """Shared representation with separate reward and constraint critics."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.reward_critic = nn.Linear(hidden_dim, 1)
        self.constraint_critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor):
        h = self.body(states)
        return (
            self.actor(h),
            self.reward_critic(h).squeeze(-1),
            self.constraint_critic(h).squeeze(-1),
        )


@dataclass(frozen=True)
class PrimalDualPPOConfig:
    updates: int = 180
    rollout_episodes: int = 24
    ppo_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    dual_learning_rate: float = 0.08
    gamma: float = 1.0
    clip_ratio: float = 0.2
    reward_value_coef: float = 0.5
    constraint_value_coef: float = 0.5
    entropy_coef: float = 0.01
    initial_lagrange: float = 0.5
    max_lagrange: float = 20.0
    seed: int = 404


class PrimalDualPPOAgent:
    """PPO with an explicit constraint critic and projected dual update."""

    def __init__(
        self,
        env: SafeWorkforceEnv,
        config: PrimalDualPPOConfig = PrimalDualPPOConfig(),
    ):
        self.env = env
        self.config = config
        set_global_seed(config.seed)
        self.model = DualActorCritic(env.state_dim, env.n_actions)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)
        self.lagrange_multiplier = float(config.initial_lagrange)

    def _collect(self, update: int):
        states = []
        actions = []
        old_log_probs = []
        reward_returns = []
        constraint_returns = []
        episode_constraint_totals = []

        for episode in range(self.config.rollout_episodes):
            obs = self.env.reset(
                seed=self.config.seed + update * 10_000 + episode
            )
            ep_states = []
            ep_actions = []
            ep_logs = []
            ep_rewards = []
            ep_constraints = []
            done = False

            while not done:
                state_t = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                ).unsqueeze(0)
                with torch.no_grad():
                    logits, _, _ = self.model(state_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_obs, reward, constraint_cost, done, _ = self.env.step(
                    int(action.item())
                )
                ep_states.append(obs.copy())
                ep_actions.append(int(action.item()))
                ep_logs.append(float(log_prob.item()))
                ep_rewards.append(float(reward))
                ep_constraints.append(float(constraint_cost))
                obs = next_obs

            reward_running = 0.0
            constraint_running = 0.0
            ep_reward_returns = [0.0] * len(ep_rewards)
            ep_constraint_returns = [0.0] * len(ep_constraints)

            for i in range(len(ep_rewards) - 1, -1, -1):
                reward_running = (
                    ep_rewards[i]
                    + self.config.gamma * reward_running
                )
                constraint_running = (
                    ep_constraints[i]
                    + self.config.gamma * constraint_running
                )
                ep_reward_returns[i] = reward_running
                ep_constraint_returns[i] = constraint_running

            states.extend(ep_states)
            actions.extend(ep_actions)
            old_log_probs.extend(ep_logs)
            reward_returns.extend(ep_reward_returns)
            constraint_returns.extend(ep_constraint_returns)
            episode_constraint_totals.append(sum(ep_constraints))

        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(old_log_probs, dtype=torch.float32),
            torch.as_tensor(reward_returns, dtype=torch.float32),
            torch.as_tensor(constraint_returns, dtype=torch.float32),
            float(np.mean(episode_constraint_totals)),
        )

    def fit(self):
        for update in range(self.config.updates):
            (
                states,
                actions,
                old_log_probs,
                reward_returns,
                constraint_returns,
                mean_episode_constraint,
            ) = self._collect(update)

            with torch.no_grad():
                _, reward_values, constraint_values = self.model(states)
                reward_adv = reward_returns - reward_values
                constraint_adv = constraint_returns - constraint_values

                reward_adv = (
                    reward_adv - reward_adv.mean()
                ) / (reward_adv.std(unbiased=False) + 1e-8)
                constraint_adv = (
                    constraint_adv - constraint_adv.mean()
                ) / (constraint_adv.std(unbiased=False) + 1e-8)

                combined_adv = (
                    reward_adv
                    - self.lagrange_multiplier * constraint_adv
                )
                combined_adv = (
                    combined_adv - combined_adv.mean()
                ) / (combined_adv.std(unbiased=False) + 1e-8)

            n = len(states)
            for _ in range(self.config.ppo_epochs):
                order = self.rng.permutation(n)
                for start in range(0, n, self.config.minibatch_size):
                    idx = torch.as_tensor(
                        order[start:start + self.config.minibatch_size],
                        dtype=torch.long,
                    )
                    logits, reward_values, constraint_values = self.model(
                        states[idx]
                    )
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[idx])
                    ratio = torch.exp(
                        new_log_probs - old_log_probs[idx]
                    )
                    adv = combined_adv[idx]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    policy_loss = -torch.min(
                        ratio * adv,
                        clipped * adv,
                    ).mean()
                    reward_value_loss = (
                        reward_values - reward_returns[idx]
                    ).square().mean()
                    constraint_value_loss = (
                        constraint_values - constraint_returns[idx]
                    ).square().mean()
                    entropy = dist.entropy().mean()

                    loss = (
                        policy_loss
                        + self.config.reward_value_coef
                        * reward_value_loss
                        + self.config.constraint_value_coef
                        * constraint_value_loss
                        - self.config.entropy_coef * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        5.0,
                    )
                    self.optimizer.step()

            self.lagrange_multiplier = float(
                np.clip(
                    self.lagrange_multiplier
                    + self.config.dual_learning_rate
                    * (
                        mean_episode_constraint
                        - self.env.config.expected_violation_budget
                    ),
                    0.0,
                    self.config.max_lagrange,
                )
            )

        return self

    def policy(self):
        config = self.env.config.workforce

        def choose(t: int, backlog: np.ndarray) -> int:
            means = self.env.expected_workload(t)
            max_mean = max(
                max(config.mean_workload) + 1.0,
                1.0,
            )
            obs = np.concatenate(
                [
                    np.asarray(
                        [t / config.horizon],
                        dtype=np.float32,
                    ),
                    (
                        np.asarray(backlog, dtype=float)
                        / config.max_backlog
                    ).astype(np.float32),
                    (
                        means / max_mean
                    ).astype(np.float32),
                ]
            )
            with torch.no_grad():
                logits, _, _ = self.model(
                    torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )
            return int(torch.argmax(logits, dim=-1).item())

        return choose
