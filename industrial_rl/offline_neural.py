from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .inventory import InventoryConfig, InventoryEnv
from .neural_common import set_global_seed
from .offline import OfflineTransition


def inventory_state_features(env: InventoryEnv, state: int) -> np.ndarray:
    t, inventory = env.decode_state(int(state))
    return np.asarray(
        [
            t / env.config.horizon,
            inventory / env.config.max_inventory,
        ],
        dtype=np.float32,
    )


def offline_dataset_diagnostics(
    dataset: tuple[OfflineTransition, ...],
    env: InventoryEnv,
    *,
    target_policy: Callable[[int, int], int] | None = None,
) -> dict[str, float]:
    if not dataset:
        raise ValueError("dataset must be non-empty")

    visited_states = set()
    visited_pairs = set()
    feasible_pairs = set()

    for t in range(env.config.horizon):
        for inventory in range(env.config.max_inventory + 1):
            state = env.state_index(t, inventory)
            for action in env.feasible_actions(inventory):
                feasible_pairs.add((state, int(action)))

    for row in dataset:
        visited_states.add(int(row.state))
        visited_pairs.add((int(row.state), int(row.action)))

    all_nonterminal_states = (
        env.config.horizon * (env.config.max_inventory + 1)
    )

    metrics = {
        "state_coverage": float(
            len(visited_states) / all_nonterminal_states
        ),
        "state_action_coverage": float(
            len(visited_pairs) / max(len(feasible_pairs), 1)
        ),
    }

    if target_policy is not None:
        relevant_states = sorted(visited_states)
        supported = 0
        for state in relevant_states:
            t, inventory = env.decode_state(state)
            action = int(target_policy(t, inventory))
            supported += int((state, action) in visited_pairs)
        metrics["target_action_support_rate"] = float(
            supported / max(len(relevant_states), 1)
        )

    return metrics


class OfflineQNetwork(nn.Module):
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

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


@dataclass(frozen=True)
class ConservativeOfflineConfig:
    gradient_steps: int = 4000
    batch_size: int = 128
    learning_rate: float = 8e-4
    gamma: float = 1.0
    conservative_alpha: float = 1.0
    target_update: int = 200
    hidden_dim: int = 64
    seed: int = 707


class ConservativeOfflineDQN:
    """Discrete CQL-style offline Q learner for finite inventory actions.

    The objective is:

        TD loss
        + alpha * (logsumexp_a Q(s,a) - Q(s,a_data))

    with infeasible order quantities masked before log-sum-exp and target max.

    This compact implementation captures the conservative discrete-Q principle.
    It is not presented as a reproduction of every CQL variant or benchmark.
    """

    def __init__(
        self,
        env: InventoryEnv,
        config: ConservativeOfflineConfig = ConservativeOfflineConfig(),
    ):
        self.env = env
        self.config = config
        set_global_seed(config.seed)

        self.online = OfflineQNetwork(
            state_dim=2,
            n_actions=env.n_actions,
            hidden_dim=config.hidden_dim,
        )
        self.target = OfflineQNetwork(
            state_dim=2,
            n_actions=env.n_actions,
            hidden_dim=config.hidden_dim,
        )
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)
        self.loss_history: list[float] = []

    def _action_mask(
        self,
        states: list[int] | np.ndarray,
    ) -> torch.Tensor:
        mask = np.zeros(
            (len(states), self.env.n_actions),
            dtype=bool,
        )
        for i, state in enumerate(states):
            _, inventory = self.env.decode_state(int(state))
            mask[i, self.env.feasible_actions(inventory)] = True
        return torch.as_tensor(mask, dtype=torch.bool)

    def fit(
        self,
        dataset: tuple[OfflineTransition, ...],
    ) -> "ConservativeOfflineDQN":
        if not dataset:
            raise ValueError("dataset must be non-empty")

        states_int = np.asarray(
            [row.state for row in dataset],
            dtype=int,
        )
        next_states_int = np.asarray(
            [row.next_state for row in dataset],
            dtype=int,
        )
        states = np.stack(
            [
                inventory_state_features(
                    self.env,
                    row.state,
                )
                for row in dataset
            ]
        )
        next_states = np.stack(
            [
                inventory_state_features(
                    self.env,
                    row.next_state,
                )
                for row in dataset
            ]
        )
        actions = np.asarray(
            [row.action for row in dataset],
            dtype=np.int64,
        )
        rewards = np.asarray(
            [row.reward for row in dataset],
            dtype=np.float32,
        )
        dones = np.asarray(
            [row.done for row in dataset],
            dtype=np.float32,
        )

        n = len(dataset)
        batch_size = min(self.config.batch_size, n)

        for step in range(self.config.gradient_steps):
            idx = self.rng.integers(
                0,
                n,
                size=batch_size,
            )

            s = torch.as_tensor(
                states[idx],
                dtype=torch.float32,
            )
            ns = torch.as_tensor(
                next_states[idx],
                dtype=torch.float32,
            )
            a = torch.as_tensor(
                actions[idx],
                dtype=torch.long,
            )
            r = torch.as_tensor(
                rewards[idx],
                dtype=torch.float32,
            )
            d = torch.as_tensor(
                dones[idx],
                dtype=torch.float32,
            )

            q_all = self.online(s)
            q_data = q_all.gather(
                1,
                a[:, None],
            ).squeeze(1)

            with torch.no_grad():
                next_q_all = self.target(ns)
                next_mask = self._action_mask(
                    next_states_int[idx]
                )
                next_q_all = next_q_all.masked_fill(
                    ~next_mask,
                    -1e9,
                )
                next_value = next_q_all.max(
                    dim=1
                ).values
                target = (
                    r
                    + (1.0 - d)
                    * self.config.gamma
                    * next_value
                )

            td_loss = F.smooth_l1_loss(
                q_data,
                target,
            )

            current_mask = self._action_mask(
                states_int[idx]
            )
            masked_q = q_all.masked_fill(
                ~current_mask,
                -1e9,
            )
            conservative_gap = (
                torch.logsumexp(
                    masked_q,
                    dim=1,
                )
                - q_data
            ).mean()

            loss = (
                td_loss
                + self.config.conservative_alpha
                * conservative_gap
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.online.parameters(),
                5.0,
            )
            self.optimizer.step()

            self.loss_history.append(
                float(loss.item())
            )

            if (
                (step + 1)
                % self.config.target_update
                == 0
            ):
                self.target.load_state_dict(
                    self.online.state_dict()
                )

        return self

    def policy(self):
        def choose(
            t: int,
            inventory: int,
        ) -> int:
            state = np.asarray(
                [
                    t / self.env.config.horizon,
                    inventory
                    / self.env.config.max_inventory,
                ],
                dtype=np.float32,
            )
            feasible = self.env.feasible_actions(
                inventory
            )
            with torch.no_grad():
                q = self.online(
                    torch.as_tensor(
                        state,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )[0]
            feasible_t = torch.as_tensor(
                feasible,
                dtype=torch.long,
            )
            return int(
                feasible[
                    int(
                        torch.argmax(
                            q[feasible_t]
                        )
                    )
                ]
            )

        return choose


@dataclass(frozen=True)
class FQEConfig:
    iterations: int = 120
    gamma: float = 1.0


class TabularFQE:
    """Fitted Q Evaluation using only the logged transition table."""

    def __init__(
        self,
        env: InventoryEnv,
        config: FQEConfig = FQEConfig(),
    ):
        self.env = env
        self.config = config
        self.q = np.zeros(
            (env.n_states, env.n_actions),
            dtype=float,
        )
        self.counts = np.zeros_like(
            self.q,
            dtype=int,
        )

    def fit(
        self,
        dataset: tuple[OfflineTransition, ...],
        target_policy: Callable[[int, int], int],
    ) -> "TabularFQE":
        if not dataset:
            raise ValueError("dataset must be non-empty")

        by_pair: dict[
            tuple[int, int],
            list[OfflineTransition],
        ] = {}

        for row in dataset:
            key = (
                int(row.state),
                int(row.action),
            )
            by_pair.setdefault(
                key,
                [],
            ).append(row)
            self.counts[key] += 1

        for _ in range(self.config.iterations):
            new_q = self.q.copy()

            for (state, action), rows in by_pair.items():
                targets = []
                for row in rows:
                    if row.done:
                        targets.append(
                            row.reward
                        )
                        continue

                    t, inventory = self.env.decode_state(
                        row.next_state
                    )
                    next_action = int(
                        target_policy(
                            t,
                            inventory,
                        )
                    )
                    targets.append(
                        row.reward
                        + self.config.gamma
                        * self.q[
                            row.next_state,
                            next_action,
                        ]
                    )

                new_q[
                    state,
                    action,
                ] = float(
                    np.mean(targets)
                )

            self.q = new_q

        return self

    def initial_value(
        self,
        target_policy: Callable[[int, int], int],
    ) -> float:
        state = self.env.state_index(
            0,
            self.env.config.initial_inventory,
        )
        action = int(
            target_policy(
                0,
                self.env.config.initial_inventory,
            )
        )
        return float(
            self.q[state, action]
        )

    def initial_cost(
        self,
        target_policy: Callable[[int, int], int],
    ) -> float:
        return -self.initial_value(
            target_policy
        )
