from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural_common import set_global_seed


OPERATE = 0
MINOR_MAINTENANCE = 1
REPLACE = 2

HEALTHY = 0
DEGRADED = 1
CRITICAL = 2
FAILED = 3

GOOD = 0
WARNING = 1
ALARM = 2
FAILURE_SIGNAL = 3


@dataclass(frozen=True)
class MaintenancePOMDPConfig:
    horizon: int = 15
    discount: float = 1.0
    initial_prior: tuple[float, float, float, float] = (
        0.92,
        0.08,
        0.0,
        0.0,
    )
    state_costs: tuple[float, float, float, float] = (
        0.0,
        8.0,
        45.0,
        240.0,
    )
    action_costs: tuple[float, float, float] = (
        0.0,
        28.0,
        115.0,
    )
    operate_transition: tuple[tuple[float, float, float, float], ...] = (
        (0.88, 0.105, 0.014, 0.001),
        (0.00, 0.73, 0.22, 0.05),
        (0.00, 0.00, 0.52, 0.48),
        (0.00, 0.00, 0.00, 1.00),
    )
    minor_transition: tuple[tuple[float, float, float, float], ...] = (
        (0.96, 0.039, 0.001, 0.00),
        (0.55, 0.40, 0.045, 0.005),
        (0.08, 0.55, 0.32, 0.05),
        (0.00, 0.00, 0.00, 1.00),
    )
    replace_transition: tuple[tuple[float, float, float, float], ...] = (
        (0.995, 0.005, 0.00, 0.00),
        (0.995, 0.005, 0.00, 0.00),
        (0.995, 0.005, 0.00, 0.00),
        (0.995, 0.005, 0.00, 0.00),
    )
    emission_matrix: tuple[tuple[float, float, float, float], ...] = (
        (0.88, 0.11, 0.01, 0.00),
        (0.18, 0.62, 0.19, 0.01),
        (0.03, 0.25, 0.62, 0.10),
        (0.00, 0.02, 0.08, 0.90),
    )

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0,1]")

        prior = np.asarray(self.initial_prior, dtype=float)
        if prior.shape != (4,) or np.any(prior < 0) or not np.isclose(prior.sum(), 1.0):
            raise ValueError("initial_prior must be a probability vector")

        for matrix in (
            self.operate_transition,
            self.minor_transition,
            self.replace_transition,
            self.emission_matrix,
        ):
            arr = np.asarray(matrix, dtype=float)
            if arr.shape != (4, 4):
                raise ValueError("transition/emission matrices must be 4x4")
            if np.any(arr < 0) or not np.allclose(arr.sum(axis=1), 1.0):
                raise ValueError("matrix rows must be non-negative and sum to one")

        if len(self.state_costs) != 4 or len(self.action_costs) != 3:
            raise ValueError("cost vectors have incompatible dimensions")

    @property
    def transition_matrices(self) -> np.ndarray:
        return np.asarray(
            [
                self.operate_transition,
                self.minor_transition,
                self.replace_transition,
            ],
            dtype=float,
        )

    @property
    def emissions(self) -> np.ndarray:
        return np.asarray(self.emission_matrix, dtype=float)


class MaintenancePOMDPEnv:
    """Hidden degradation model with noisy sensor observations and Bayesian belief."""

    def __init__(self, config: MaintenancePOMDPConfig = MaintenancePOMDPConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.hidden_state = HEALTHY
        self.observation = GOOD
        self.belief = np.asarray(config.initial_prior, dtype=float)

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def state_dim(self) -> int:
        return 5

    def state_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray([self.t / self.config.horizon], dtype=np.float32),
                self.belief.astype(np.float32),
            ]
        )

    def predict_belief(self, belief: np.ndarray, action: int) -> np.ndarray:
        if not 0 <= int(action) < self.n_actions:
            raise ValueError("invalid maintenance action")
        b = np.asarray(belief, dtype=float)
        if b.shape != (4,) or np.any(b < 0) or not np.isclose(b.sum(), 1.0):
            raise ValueError("belief must be a probability vector")
        return b @ self.config.transition_matrices[int(action)]

    def bayes_update(
        self,
        predicted_belief: np.ndarray,
        observation: int,
    ) -> np.ndarray:
        if not 0 <= int(observation) < 4:
            raise ValueError("invalid sensor observation")
        predicted = np.asarray(predicted_belief, dtype=float)
        likelihood = self.config.emissions[:, int(observation)]
        posterior = predicted * likelihood
        total = float(posterior.sum())
        if total <= 0.0:
            return predicted / predicted.sum()
        return posterior / total

    def next_belief(
        self,
        belief: np.ndarray,
        action: int,
        observation: int,
    ) -> np.ndarray:
        return self.bayes_update(
            self.predict_belief(belief, action),
            observation,
        )

    def observation_probability(
        self,
        predicted_belief: np.ndarray,
        observation: int,
    ) -> float:
        return float(
            np.dot(
                np.asarray(predicted_belief, dtype=float),
                self.config.emissions[:, int(observation)],
            )
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.t = 0
        prior = np.asarray(self.config.initial_prior, dtype=float)
        self.hidden_state = int(self.rng.choice(4, p=prior))
        self.observation = int(
            self.rng.choice(
                4,
                p=self.config.emissions[self.hidden_state],
            )
        )
        posterior = prior * self.config.emissions[:, self.observation]
        self.belief = posterior / posterior.sum()
        return self.state_vector()

    def step(self, action: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode finished")
        if not 0 <= int(action) < self.n_actions:
            raise ValueError("invalid maintenance action")

        action = int(action)
        transition = self.config.transition_matrices[action]
        next_hidden = int(
            self.rng.choice(
                4,
                p=transition[self.hidden_state],
            )
        )
        next_observation = int(
            self.rng.choice(
                4,
                p=self.config.emissions[next_hidden],
            )
        )

        predicted = self.predict_belief(self.belief, action)
        next_belief = self.bayes_update(
            predicted,
            next_observation,
        )

        cost = (
            self.config.action_costs[action]
            + self.config.state_costs[next_hidden]
        )

        self.hidden_state = next_hidden
        self.observation = next_observation
        self.belief = next_belief
        self.t += 1
        done = self.t >= self.config.horizon

        info = {
            "cost": float(cost),
            "hidden_state": int(next_hidden),
            "observation": int(next_observation),
            "belief": next_belief.copy(),
            "failed": bool(next_hidden == FAILED),
            "action": action,
        }
        return self.state_vector(), -float(cost), done, info


def reactive_sensor_policy(
    t: int,
    belief: np.ndarray,
    observation: int,
) -> int:
    del t, belief
    if observation == GOOD:
        return OPERATE
    if observation == WARNING:
        return MINOR_MAINTENANCE
    return REPLACE


def belief_threshold_policy(
    *,
    replace_threshold: float = 0.34,
    maintenance_threshold: float = 0.42,
):
    def choose(
        t: int,
        belief: np.ndarray,
        observation: int,
    ) -> int:
        del t, observation
        b = np.asarray(belief, dtype=float)
        severe = float(b[CRITICAL] + b[FAILED])
        degraded_or_worse = float(
            b[DEGRADED] + b[CRITICAL] + b[FAILED]
        )
        if severe >= replace_threshold:
            return REPLACE
        if degraded_or_worse >= maintenance_threshold:
            return MINOR_MAINTENANCE
        return OPERATE

    return choose


def belief_grid(resolution: int = 10) -> np.ndarray:
    if resolution < 1:
        raise ValueError("resolution must be positive")
    points = []
    for a in range(resolution + 1):
        for b in range(resolution - a + 1):
            for c in range(resolution - a - b + 1):
                d = resolution - a - b - c
                points.append((a, b, c, d))
    return np.asarray(points, dtype=float) / float(resolution)


def nearest_belief_index(
    grid: np.ndarray,
    belief: np.ndarray,
) -> int:
    diff = grid - np.asarray(belief, dtype=float)
    return int(np.argmin(np.sum(diff * diff, axis=1)))


def discretized_belief_dp(
    config: MaintenancePOMDPConfig = MaintenancePOMDPConfig(),
    *,
    resolution: int = 10,
):
    """Approximate finite-horizon belief-MDP DP on a simplex grid."""
    config.validate()
    env = MaintenancePOMDPEnv(config)
    grid = belief_grid(resolution)
    n_grid = len(grid)

    values = np.zeros(
        (config.horizon + 1, n_grid),
        dtype=float,
    )
    policy = np.zeros(
        (config.horizon, n_grid),
        dtype=int,
    )

    next_index = np.zeros((n_grid, 3, 4), dtype=int)
    obs_prob = np.zeros((n_grid, 3, 4), dtype=float)
    immediate_cost = np.zeros((n_grid, 3), dtype=float)

    state_cost = np.asarray(config.state_costs, dtype=float)

    for i, belief in enumerate(grid):
        for action in range(3):
            predicted = env.predict_belief(belief, action)
            immediate_cost[i, action] = (
                config.action_costs[action]
                + float(np.dot(predicted, state_cost))
            )
            for obs in range(4):
                p_obs = env.observation_probability(
                    predicted,
                    obs,
                )
                obs_prob[i, action, obs] = p_obs
                if p_obs > 0.0:
                    posterior = env.bayes_update(
                        predicted,
                        obs,
                    )
                else:
                    posterior = predicted
                next_index[i, action, obs] = nearest_belief_index(
                    grid,
                    posterior,
                )

    for t in range(config.horizon - 1, -1, -1):
        for i in range(n_grid):
            action_values = np.empty(3, dtype=float)
            for action in range(3):
                continuation = 0.0
                for obs in range(4):
                    continuation += (
                        obs_prob[i, action, obs]
                        * values[
                            t + 1,
                            next_index[i, action, obs],
                        ]
                    )
                action_values[action] = (
                    immediate_cost[i, action]
                    + config.discount * continuation
                )
            policy[t, i] = int(np.argmin(action_values))
            values[t, i] = float(action_values[policy[t, i]])

    return grid, values, policy


def discretized_belief_policy(
    grid: np.ndarray,
    policy_table: np.ndarray,
):
    def choose(
        t: int,
        belief: np.ndarray,
        observation: int,
    ) -> int:
        del observation
        idx = nearest_belief_index(
            grid,
            belief,
        )
        return int(policy_table[t, idx])

    return choose


def evaluate_maintenance_policy(
    policy: Callable[[int, np.ndarray, int], int],
    config: MaintenancePOMDPConfig = MaintenancePOMDPConfig(),
    *,
    episodes: int = 1000,
    seed: int = 150_000,
) -> dict[str, float]:
    env = MaintenancePOMDPEnv(config)
    costs = []
    failure_period_rates = []
    replacement_rates = []
    maintenance_rates = []
    belief_entropies = []

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_cost = 0.0
        failure_periods = 0
        replacements = 0
        maintenance = 0
        entropy_sum = 0.0
        done = False

        while not done:
            action = int(
                policy(
                    env.t,
                    env.belief.copy(),
                    env.observation,
                )
            )
            _, reward, done, info = env.step(action)
            total_cost += -reward
            failure_periods += int(info["failed"])
            replacements += int(action == REPLACE)
            maintenance += int(action == MINOR_MAINTENANCE)

            b = np.clip(
                np.asarray(info["belief"], dtype=float),
                1e-12,
                1.0,
            )
            entropy_sum += float(
                -np.sum(b * np.log(b))
            )

        costs.append(total_cost)
        failure_period_rates.append(
            failure_periods / config.horizon
        )
        replacement_rates.append(
            replacements / config.horizon
        )
        maintenance_rates.append(
            maintenance / config.horizon
        )
        belief_entropies.append(
            entropy_sum / config.horizon
        )

    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_failure_period_rate": float(
            np.mean(failure_period_rates)
        ),
        "mean_replacement_rate": float(
            np.mean(replacement_rates)
        ),
        "mean_minor_maintenance_rate": float(
            np.mean(maintenance_rates)
        ),
        "mean_belief_entropy": float(
            np.mean(belief_entropies)
        ),
    }


class BeliefQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class BeliefDQNConfig:
    episodes: int = 3500
    batch_size: int = 64
    replay_size: int = 25_000
    warmup: int = 500
    learning_rate: float = 1e-3
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.04
    target_update: int = 250
    seed: int = 606


class BeliefDQNAgent:
    def __init__(
        self,
        env: MaintenancePOMDPEnv,
        config: BeliefDQNConfig = BeliefDQNConfig(),
    ):
        self.env = env
        self.config = config
        set_global_seed(config.seed)

        self.online = BeliefQNetwork(
            env.state_dim,
            env.n_actions,
        )
        self.target = BeliefQNetwork(
            env.state_dim,
            env.n_actions,
        )
        self.target.load_state_dict(
            self.online.state_dict()
        )
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
        )
        self.replay = deque(
            maxlen=config.replay_size
        )
        self.rng = np.random.default_rng(config.seed)
        self.update_count = 0

    def _epsilon(self, episode: int) -> float:
        frac = episode / max(
            self.config.episodes - 1,
            1,
        )
        return (
            self.config.epsilon_start
            + frac
            * (
                self.config.epsilon_end
                - self.config.epsilon_start
            )
        )

    def _greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            q = self.online(
                torch.as_tensor(
                    state,
                    dtype=torch.float32,
                ).unsqueeze(0)
            )
        return int(torch.argmax(q, dim=-1).item())

    def _optimize(self):
        if len(self.replay) < max(
            self.config.warmup,
            self.config.batch_size,
        ):
            return

        idx = self.rng.integers(
            0,
            len(self.replay),
            size=self.config.batch_size,
        )
        batch = [
            self.replay[int(i)]
            for i in idx
        ]

        states = torch.as_tensor(
            np.stack([b[0] for b in batch]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            [b[1] for b in batch],
            dtype=torch.long,
        )
        rewards = torch.as_tensor(
            [b[2] for b in batch],
            dtype=torch.float32,
        )
        next_states = torch.as_tensor(
            np.stack([b[3] for b in batch]),
            dtype=torch.float32,
        )
        dones = torch.as_tensor(
            [b[4] for b in batch],
            dtype=torch.float32,
        )

        q = self.online(states).gather(
            1,
            actions[:, None],
        ).squeeze(1)

        with torch.no_grad():
            target = (
                rewards
                + (1.0 - dones)
                * self.config.gamma
                * self.target(next_states).max(dim=1).values
            )

        loss = F.smooth_l1_loss(
            q,
            target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            5.0,
        )
        self.optimizer.step()

        self.update_count += 1
        if (
            self.update_count
            % self.config.target_update
            == 0
        ):
            self.target.load_state_dict(
                self.online.state_dict()
            )

    def fit(self):
        for episode in range(self.config.episodes):
            state = self.env.reset(
                seed=self.config.seed + episode
            )
            done = False

            while not done:
                if (
                    self.rng.random()
                    < self._epsilon(episode)
                ):
                    action = int(
                        self.rng.integers(
                            0,
                            self.env.n_actions,
                        )
                    )
                else:
                    action = self._greedy(state)

                next_state, reward, done, _ = self.env.step(
                    action
                )

                self.replay.append(
                    (
                        state.copy(),
                        action,
                        float(reward),
                        next_state.copy(),
                        bool(done),
                    )
                )
                state = next_state
                self._optimize()

        return self

    def policy(self):
        config = self.env.config

        def choose(
            t: int,
            belief: np.ndarray,
            observation: int,
        ) -> int:
            del observation
            state = np.concatenate(
                [
                    np.asarray(
                        [t / config.horizon],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        belief,
                        dtype=np.float32,
                    ),
                ]
            )
            return self._greedy(state)

        return choose
