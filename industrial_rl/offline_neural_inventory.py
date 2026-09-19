from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dqn_inventory import RegimeInventoryConfig, RegimeInventoryEnv
from .neural_common import set_global_seed


@dataclass(frozen=True)
class LoggedRegimeTransition:
    state: tuple[float, float, float, float]
    action: int
    reward: float
    next_state: tuple[float, float, float, float]
    done: bool
    t: int
    inventory: int
    regime: int
    next_t: int
    next_inventory: int
    next_regime: int


def discrete_state_id(
    config: RegimeInventoryConfig,
    t: int,
    inventory: int,
    regime: int,
) -> int:
    return (
        (int(t) * (config.max_inventory + 1) + int(inventory)) * 2
        + int(regime)
    )


def generate_regime_inventory_logs(
    behavior_policy: Callable[[int, int, int], int],
    config: RegimeInventoryConfig = RegimeInventoryConfig(),
    *,
    episodes: int = 1200,
    epsilon: float = 0.12,
    seed: int = 170_000,
) -> tuple[LoggedRegimeTransition, ...]:
    """Generate a fixed operations log for offline learning.

    The dataset generator can interact with the simulator. Offline learners
    receive only the returned transition table and never call env.step().
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0,1]")

    env = RegimeInventoryEnv(config)
    rng = np.random.default_rng(seed)
    rows: list[LoggedRegimeTransition] = []

    for episode in range(episodes):
        obs = env.reset(seed=seed + episode)
        done = False

        while not done:
            t = env.t
            inventory = env.inventory
            regime = env.regime
            feasible = env.feasible_actions(inventory)

            if rng.random() < epsilon:
                action = int(rng.choice(feasible))
            else:
                action = int(
                    behavior_policy(t, inventory, regime)
                )
                if action not in feasible:
                    action = int(feasible[-1])

            next_obs, reward, done, _ = env.step(action)
            rows.append(
                LoggedRegimeTransition(
                    state=tuple(float(x) for x in obs),
                    action=action,
                    reward=float(reward),
                    next_state=tuple(float(x) for x in next_obs),
                    done=bool(done),
                    t=int(t),
                    inventory=int(inventory),
                    regime=int(regime),
                    next_t=int(env.t),
                    next_inventory=int(env.inventory),
                    next_regime=int(env.regime),
                )
            )
            obs = next_obs

    return tuple(rows)


def dataset_coverage(
    dataset: tuple[LoggedRegimeTransition, ...],
    config: RegimeInventoryConfig,
) -> dict[str, float]:
    if not dataset:
        raise ValueError("dataset must be non-empty")

    visited_states = {
        (row.t, row.inventory, row.regime)
        for row in dataset
    }
    visited_pairs = {
        (row.t, row.inventory, row.regime, row.action)
        for row in dataset
    }

    feasible_states = 0
    feasible_pairs = 0
    env = RegimeInventoryEnv(config)
    for t in range(config.horizon):
        for inventory in range(config.max_inventory + 1):
            for regime in (0, 1):
                feasible_states += 1
                feasible_pairs += len(env.feasible_actions(inventory))

    counts: dict[tuple[int, int, int], np.ndarray] = {}
    for row in dataset:
        key = (row.t, row.inventory, row.regime)
        if key not in counts:
            counts[key] = np.zeros(env.n_actions, dtype=int)
        counts[key][row.action] += 1

    entropies = []
    for c in counts.values():
        p = c[c > 0].astype(float)
        p /= p.sum()
        entropies.append(float(-np.sum(p * np.log(p))))

    return {
        "transitions": float(len(dataset)),
        "state_coverage": float(len(visited_states) / feasible_states),
        "state_action_coverage": float(len(visited_pairs) / feasible_pairs),
        "mean_behavior_entropy": float(np.mean(entropies)),
    }


def unsupported_policy_action_rate(
    dataset: tuple[LoggedRegimeTransition, ...],
    policy: Callable[[int, int, int], int],
) -> float:
    if not dataset:
        raise ValueError("dataset must be non-empty")

    observed: dict[tuple[int, int, int], set[int]] = {}
    for row in dataset:
        observed.setdefault(
            (row.t, row.inventory, row.regime),
            set(),
        ).add(row.action)

    unsupported = 0
    for t, inventory, regime in observed:
        action = int(policy(t, inventory, regime))
        unsupported += int(
            action not in observed[(t, inventory, regime)]
        )

    return float(unsupported / max(len(observed), 1))


def _arrays(
    dataset: tuple[LoggedRegimeTransition, ...],
):
    states = np.asarray([row.state for row in dataset], dtype=np.float32)
    actions = np.asarray([row.action for row in dataset], dtype=np.int64)
    rewards = np.asarray([row.reward for row in dataset], dtype=np.float32)
    next_states = np.asarray(
        [row.next_state for row in dataset],
        dtype=np.float32,
    )
    dones = np.asarray([row.done for row in dataset], dtype=np.float32)
    inventories = np.asarray(
        [row.inventory for row in dataset],
        dtype=np.int64,
    )
    next_inventories = np.asarray(
        [row.next_inventory for row in dataset],
        dtype=np.int64,
    )
    return (
        states,
        actions,
        rewards,
        next_states,
        dones,
        inventories,
        next_inventories,
    )


class OfflineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mask_from_inventory(
    inventories: np.ndarray,
    config: RegimeInventoryConfig,
) -> torch.Tensor:
    n_actions = config.max_order + 1
    masks = np.zeros((len(inventories), n_actions), dtype=bool)
    for i, inventory in enumerate(inventories):
        upper = min(
            config.max_order,
            config.max_inventory - int(inventory),
        )
        masks[i, : upper + 1] = True
    return torch.as_tensor(masks, dtype=torch.bool)


@dataclass(frozen=True)
class OfflineBCConfig:
    gradient_steps: int = 1200
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 707


class NeuralBehaviorCloning:
    def __init__(
        self,
        config: RegimeInventoryConfig,
        train_config: OfflineBCConfig = OfflineBCConfig(),
    ):
        self.env_config = config
        self.train_config = train_config
        set_global_seed(train_config.seed)
        self.model = OfflineMLP(4, config.max_order + 1)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config.learning_rate,
        )
        self.rng = np.random.default_rng(train_config.seed)

    def fit(
        self,
        dataset: tuple[LoggedRegimeTransition, ...],
    ):
        if not dataset:
            raise ValueError("dataset must be non-empty")
        states, actions, *_ = _arrays(dataset)

        for _ in range(self.train_config.gradient_steps):
            idx = self.rng.integers(
                0,
                len(dataset),
                size=self.train_config.batch_size,
            )
            s = torch.as_tensor(states[idx], dtype=torch.float32)
            a = torch.as_tensor(actions[idx], dtype=torch.long)
            logits = self.model(s)
            loss = F.cross_entropy(logits, a)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
        return self

    def policy(self):
        config = self.env_config
        env = RegimeInventoryEnv(config)

        def choose(t: int, inventory: int, regime: int) -> int:
            obs = np.asarray(
                [
                    t / config.horizon,
                    inventory / config.max_inventory,
                    float(regime == 0),
                    float(regime == 1),
                ],
                dtype=np.float32,
            )
            feasible = env.feasible_actions(inventory)
            with torch.no_grad():
                logits = self.model(
                    torch.as_tensor(obs).unsqueeze(0)
                )[0]
            feasible_t = torch.as_tensor(feasible, dtype=torch.long)
            return int(
                feasible[int(torch.argmax(logits[feasible_t]))]
            )

        return choose


@dataclass(frozen=True)
class CQLConfig:
    gradient_steps: int = 1800
    batch_size: int = 128
    learning_rate: float = 7e-4
    gamma: float = 1.0
    conservative_weight: float = 1.0
    target_update: int = 100
    seed: int = 708


class DiscreteCQL:
    """Compact discrete Conservative Q-Learning implementation."""

    def __init__(
        self,
        config: RegimeInventoryConfig,
        train_config: CQLConfig = CQLConfig(),
    ):
        self.env_config = config
        self.train_config = train_config
        set_global_seed(train_config.seed)
        self.q = OfflineMLP(4, config.max_order + 1)
        self.target_q = OfflineMLP(4, config.max_order + 1)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(
            self.q.parameters(),
            lr=train_config.learning_rate,
        )
        self.rng = np.random.default_rng(train_config.seed)

    def fit(
        self,
        dataset: tuple[LoggedRegimeTransition, ...],
    ):
        if not dataset:
            raise ValueError("dataset must be non-empty")
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            inventories,
            next_inventories,
        ) = _arrays(dataset)

        masks = _mask_from_inventory(inventories, self.env_config)
        next_masks = _mask_from_inventory(
            next_inventories,
            self.env_config,
        )

        for step in range(self.train_config.gradient_steps):
            idx_np = self.rng.integers(
                0,
                len(dataset),
                size=self.train_config.batch_size,
            )
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            s = torch.as_tensor(states[idx_np], dtype=torch.float32)
            a = torch.as_tensor(actions[idx_np], dtype=torch.long)
            r = torch.as_tensor(rewards[idx_np], dtype=torch.float32)
            ns = torch.as_tensor(
                next_states[idx_np],
                dtype=torch.float32,
            )
            d = torch.as_tensor(dones[idx_np], dtype=torch.float32)
            mask = masks[idx]
            next_mask = next_masks[idx]

            q_all = self.q(s)
            q_data = q_all.gather(1, a[:, None]).squeeze(1)

            with torch.no_grad():
                next_q = self.target_q(ns).masked_fill(
                    ~next_mask,
                    -1e9,
                )
                target = (
                    r
                    + (1.0 - d)
                    * self.train_config.gamma
                    * next_q.max(dim=1).values
                )

            td_loss = F.smooth_l1_loss(q_data, target)
            feasible_q = q_all.masked_fill(~mask, -1e9)
            conservative = (
                torch.logsumexp(feasible_q, dim=1)
                - q_data
            ).mean()
            loss = (
                td_loss
                + self.train_config.conservative_weight
                * conservative
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.optimizer.step()

            if (step + 1) % self.train_config.target_update == 0:
                self.target_q.load_state_dict(self.q.state_dict())

        return self

    def policy(self):
        config = self.env_config
        env = RegimeInventoryEnv(config)

        def choose(t: int, inventory: int, regime: int) -> int:
            obs = np.asarray(
                [
                    t / config.horizon,
                    inventory / config.max_inventory,
                    float(regime == 0),
                    float(regime == 1),
                ],
                dtype=np.float32,
            )
            feasible = env.feasible_actions(inventory)
            with torch.no_grad():
                q = self.q(
                    torch.as_tensor(obs).unsqueeze(0)
                )[0]
            feasible_t = torch.as_tensor(feasible, dtype=torch.long)
            return int(feasible[int(torch.argmax(q[feasible_t]))])

        return choose


@dataclass(frozen=True)
class IQLConfig:
    gradient_steps: int = 2000
    batch_size: int = 128
    learning_rate: float = 7e-4
    gamma: float = 1.0
    expectile: float = 0.70
    temperature: float = 2.0
    max_weight: float = 20.0
    target_update: int = 100
    seed: int = 709


class DiscreteIQL:
    """Discrete Implicit Q-Learning with expectile V and weighted BC."""

    def __init__(
        self,
        config: RegimeInventoryConfig,
        train_config: IQLConfig = IQLConfig(),
    ):
        self.env_config = config
        self.train_config = train_config
        set_global_seed(train_config.seed)

        self.q = OfflineMLP(4, config.max_order + 1)
        self.target_q = OfflineMLP(4, config.max_order + 1)
        self.target_q.load_state_dict(self.q.state_dict())
        self.v = OfflineMLP(4, 1)
        self.actor = OfflineMLP(4, config.max_order + 1)

        self.q_opt = torch.optim.Adam(
            self.q.parameters(),
            lr=train_config.learning_rate,
        )
        self.v_opt = torch.optim.Adam(
            self.v.parameters(),
            lr=train_config.learning_rate,
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=train_config.learning_rate,
        )
        self.rng = np.random.default_rng(train_config.seed)

    def fit(
        self,
        dataset: tuple[LoggedRegimeTransition, ...],
    ):
        if not dataset:
            raise ValueError("dataset must be non-empty")
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            _,
            _,
        ) = _arrays(dataset)

        tau = float(self.train_config.expectile)
        if not 0.0 < tau < 1.0:
            raise ValueError("expectile must be in (0,1)")

        for step in range(self.train_config.gradient_steps):
            idx = self.rng.integers(
                0,
                len(dataset),
                size=self.train_config.batch_size,
            )
            s = torch.as_tensor(states[idx], dtype=torch.float32)
            a = torch.as_tensor(actions[idx], dtype=torch.long)
            r = torch.as_tensor(rewards[idx], dtype=torch.float32)
            ns = torch.as_tensor(
                next_states[idx],
                dtype=torch.float32,
            )
            d = torch.as_tensor(dones[idx], dtype=torch.float32)

            with torch.no_grad():
                target_q_data = self.target_q(s).gather(
                    1,
                    a[:, None],
                ).squeeze(1)

            v = self.v(s).squeeze(1)
            diff = target_q_data - v
            weight = torch.where(
                diff > 0,
                tau,
                1.0 - tau,
            )
            v_loss = (weight * diff.square()).mean()

            self.v_opt.zero_grad()
            v_loss.backward()
            self.v_opt.step()

            with torch.no_grad():
                next_v = self.v(ns).squeeze(1)
                target = (
                    r
                    + (1.0 - d)
                    * self.train_config.gamma
                    * next_v
                )

            q_data = self.q(s).gather(
                1,
                a[:, None],
            ).squeeze(1)
            q_loss = F.smooth_l1_loss(q_data, target)

            self.q_opt.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.q_opt.step()

            with torch.no_grad():
                advantage = (
                    self.target_q(s).gather(
                        1,
                        a[:, None],
                    ).squeeze(1)
                    - self.v(s).squeeze(1)
                )
                actor_weight = torch.exp(
                    self.train_config.temperature * advantage
                ).clamp(max=self.train_config.max_weight)

            logits = self.actor(s)
            log_prob = F.log_softmax(logits, dim=1).gather(
                1,
                a[:, None],
            ).squeeze(1)
            actor_loss = -(actor_weight * log_prob).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_opt.step()

            if (step + 1) % self.train_config.target_update == 0:
                self.target_q.load_state_dict(self.q.state_dict())

        return self

    def policy(self):
        config = self.env_config
        env = RegimeInventoryEnv(config)

        def choose(t: int, inventory: int, regime: int) -> int:
            obs = np.asarray(
                [
                    t / config.horizon,
                    inventory / config.max_inventory,
                    float(regime == 0),
                    float(regime == 1),
                ],
                dtype=np.float32,
            )
            feasible = env.feasible_actions(inventory)
            with torch.no_grad():
                logits = self.actor(
                    torch.as_tensor(obs).unsqueeze(0)
                )[0]
            feasible_t = torch.as_tensor(feasible, dtype=torch.long)
            return int(
                feasible[int(torch.argmax(logits[feasible_t]))]
            )

        return choose


def tabular_fqe(
    dataset: tuple[LoggedRegimeTransition, ...],
    policy: Callable[[int, int, int], int],
    config: RegimeInventoryConfig,
    *,
    iterations: int = 80,
    gamma: float = 1.0,
) -> float:
    """Model-free fitted Q evaluation on the fixed logged transition table.

    Returns an estimated expected *cost* from the fixed initial state.
    The estimate is unreliable when the evaluated policy selects unsupported
    actions, so it must be read alongside unsupported_policy_action_rate().
    """
    if not dataset:
        raise ValueError("dataset must be non-empty")

    n_states = (
        (config.horizon + 1)
        * (config.max_inventory + 1)
        * 2
    )
    n_actions = config.max_order + 1
    q = np.zeros((n_states, n_actions), dtype=float)

    groups: dict[tuple[int, int], list[LoggedRegimeTransition]] = {}
    for row in dataset:
        sid = discrete_state_id(
            config,
            row.t,
            row.inventory,
            row.regime,
        )
        groups.setdefault((sid, row.action), []).append(row)

    for _ in range(iterations):
        new_q = q.copy()
        for (sid, action), rows in groups.items():
            targets = []
            for row in rows:
                if row.done:
                    targets.append(row.reward)
                else:
                    next_sid = discrete_state_id(
                        config,
                        row.next_t,
                        row.next_inventory,
                        row.next_regime,
                    )
                    next_action = int(
                        policy(
                            row.next_t,
                            row.next_inventory,
                            row.next_regime,
                        )
                    )
                    targets.append(
                        row.reward
                        + gamma * q[next_sid, next_action]
                    )
            new_q[sid, action] = float(np.mean(targets))
        q = new_q

    initial_sid = discrete_state_id(
        config,
        0,
        config.initial_inventory,
        config.initial_regime,
    )
    initial_action = int(
        policy(
            0,
            config.initial_inventory,
            config.initial_regime,
        )
    )
    return float(-q[initial_sid, initial_action])
