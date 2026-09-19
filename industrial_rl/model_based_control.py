from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .neural_common import set_global_seed
from .sac_energy import EnergyProductionConfig, EnergyProductionEnv, evaluate_energy_policy


@dataclass(frozen=True)
class DynamicsTrainingConfig:
    transitions: int = 5000
    ensemble_size: int = 4
    hidden_dim: int = 96
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 501


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 4
    candidates: int = 96
    iterations: int = 3
    elite_fraction: float = 0.15
    uncertainty_penalty: float = 0.15
    min_std: float = 0.08
    seed: int = 777


class DynamicsNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedDynamicsEnsemble:
    """Bootstrap ensemble predicting next observation and one-step reward."""

    def __init__(self, state_dim: int, config: DynamicsTrainingConfig):
        self.state_dim = int(state_dim)
        self.config = config
        self.members = nn.ModuleList(
            [DynamicsNet(state_dim + 1, state_dim + 1, config.hidden_dim) for _ in range(config.ensemble_size)]
        )
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray) -> "LearnedDynamicsEnsemble":
        set_global_seed(self.config.seed)
        x = np.concatenate([states, actions.reshape(-1, 1)], axis=1).astype(np.float32)
        y = np.concatenate([next_states, rewards.reshape(-1, 1)], axis=1).astype(np.float32)

        x_t = torch.as_tensor(x)
        y_t = torch.as_tensor(y)
        self.x_mean = x_t.mean(0, keepdim=True)
        self.x_std = x_t.std(0, keepdim=True).clamp_min(1e-5)
        self.y_mean = y_t.mean(0, keepdim=True)
        self.y_std = y_t.std(0, keepdim=True).clamp_min(1e-5)
        x_n = (x_t - self.x_mean) / self.x_std
        y_n = (y_t - self.y_mean) / self.y_std

        rng = np.random.default_rng(self.config.seed)
        n = len(x)
        for model in self.members:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            boot = rng.integers(0, n, size=n)
            for _ in range(self.config.epochs):
                order = rng.permutation(n)
                for start in range(0, n, self.config.batch_size):
                    idx_np = boot[order[start : start + self.config.batch_size]]
                    idx = torch.as_tensor(idx_np, dtype=torch.long)
                    pred = model(x_n[idx])
                    loss = torch.mean((pred - y_n[idx]) ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            model.eval()
        return self

    def _check_fitted(self) -> None:
        if any(v is None for v in (self.x_mean, self.x_std, self.y_mean, self.y_std)):
            raise RuntimeError("dynamics ensemble is not fitted")

    @torch.no_grad()
    def predict_members(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_fitted()
        if actions.ndim == 1:
            actions = actions[:, None]
        x = torch.cat([states, actions], dim=-1)
        x_n = (x - self.x_mean) / self.x_std
        preds = []
        for model in self.members:
            y_n = model(x_n)
            preds.append(y_n * self.y_std + self.y_mean)
        stacked = torch.stack(preds, dim=0)
        return stacked[..., : self.state_dim], stacked[..., self.state_dim]


def collect_random_transitions(
    config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    transitions: int = 5000,
    seed: int = 501,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = EnergyProductionEnv(config)
    rng = np.random.default_rng(seed)
    state = env.reset(seed=seed)
    states, actions, next_states, rewards = [], [], [], []
    for i in range(int(transitions)):
        action = float(rng.uniform(-1.0, 1.0))
        next_state, reward, done, _ = env.step(action)
        states.append(state.copy())
        actions.append(action)
        next_states.append(next_state.copy())
        rewards.append(float(reward))
        state = env.reset(seed=seed + i + 1) if done else next_state
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
    )


def train_dynamics_ensemble(
    env_config: EnergyProductionConfig = EnergyProductionConfig(),
    training: DynamicsTrainingConfig = DynamicsTrainingConfig(),
) -> LearnedDynamicsEnsemble:
    data = collect_random_transitions(env_config, transitions=training.transitions, seed=training.seed)
    model = LearnedDynamicsEnsemble(state_dim=data[0].shape[1], config=training)
    return model.fit(*data)


def cem_mpc_policy(
    model: LearnedDynamicsEnsemble,
    config: CEMConfig = CEMConfig(),
) -> Callable[[np.ndarray], float]:
    rng = np.random.default_rng(config.seed)
    elite_count = max(2, int(round(config.candidates * config.elite_fraction)))

    def choose(obs: np.ndarray) -> float:
        mean = np.zeros(config.horizon, dtype=np.float32)
        std = np.ones(config.horizon, dtype=np.float32) * 0.75
        initial = torch.as_tensor(obs, dtype=torch.float32)

        for _ in range(config.iterations):
            seq = rng.normal(mean, std, size=(config.candidates, config.horizon)).astype(np.float32)
            seq = np.clip(seq, -1.0, 1.0)
            member_scores = np.zeros((len(model.members), config.candidates), dtype=np.float32)

            for m in range(len(model.members)):
                states = initial.repeat(config.candidates, 1)
                cumulative = torch.zeros(config.candidates)
                for h in range(config.horizon):
                    actions = torch.as_tensor(seq[:, h], dtype=torch.float32)
                    next_all, reward_all = model.predict_members(states, actions)
                    next_state = next_all[m]
                    reward = reward_all[m]
                    cumulative = cumulative + reward
                    states = torch.clamp(next_state, -3.0, 3.0)
                member_scores[m] = cumulative.numpy()

            mean_score = member_scores.mean(axis=0)
            uncertainty = member_scores.std(axis=0)
            robust_score = mean_score - config.uncertainty_penalty * uncertainty
            elite = seq[np.argsort(robust_score)[-elite_count:]]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), config.min_std)

        return float(np.clip(mean[0], -1.0, 1.0))

    return choose


def evaluate_model_based_controller(
    model: LearnedDynamicsEnsemble,
    env_config: EnergyProductionConfig = EnergyProductionConfig(),
    cem_config: CEMConfig = CEMConfig(),
    *,
    episodes: int = 100,
    seed: int = 95_000,
) -> dict[str, float]:
    return evaluate_energy_policy(cem_mpc_policy(model, cem_config), env_config, episodes=episodes, seed=seed)
