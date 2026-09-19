from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .neural_common import set_global_seed
from .sac_energy import EnergyProductionConfig, EnergyProductionEnv, evaluate_energy_policy, mpc_grid_policy


@dataclass(frozen=True)
class LatentControlConfig:
    high_dim: int = 64
    latent_dim: int = 8
    samples: int = 1800
    autoencoder_epochs: int = 80
    policy_epochs: int = 70
    learning_rate: float = 1e-3
    action_levels: int = 9
    seed: int = 923


class ObservationMap:
    """Deterministic high-dimensional sensor map of a compact process state."""

    def __init__(self, state_dim: int, high_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.8, size=(state_dim, high_dim // 2)).astype(np.float32)
        self.w2 = rng.normal(0.0, 0.8, size=(state_dim, high_dim - high_dim // 2)).astype(np.float32)

    def transform(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32)
        a = np.tanh(x @ self.w1)
        b = np.sin(x @ self.w2)
        return np.concatenate([a, b], axis=-1).astype(np.float32)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(), nn.Linear(48, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48), nn.ReLU(), nn.Linear(48, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PolicyClassifier(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LatentControlModels:
    observation_map: ObservationMap
    autoencoder: AutoEncoder
    raw_policy: PolicyClassifier
    latent_policy: PolicyClassifier
    action_grid: np.ndarray
    reconstruction_mse: float
    raw_accuracy: float
    latent_accuracy: float


def _collect_states(config: EnergyProductionConfig, n: int, seed: int) -> np.ndarray:
    env = EnergyProductionEnv(config)
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    states = []
    for i in range(n):
        states.append(obs.copy())
        obs, _, done, _ = env.step(float(rng.uniform(-1.0, 1.0)))
        if done:
            obs = env.reset(seed=seed + i + 1)
    return np.asarray(states, dtype=np.float32)


def _fit_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def train_latent_control(
    env_config: EnergyProductionConfig = EnergyProductionConfig(),
    config: LatentControlConfig = LatentControlConfig(),
) -> LatentControlModels:
    set_global_seed(config.seed)
    base_states = _collect_states(env_config, config.samples, config.seed)
    obs_map = ObservationMap(base_states.shape[1], config.high_dim, config.seed)
    high = np.stack([obs_map.transform(s) for s in base_states]).astype(np.float32)

    high_t = torch.as_tensor(high)
    autoencoder = AutoEncoder(config.high_dim, config.latent_dim)
    ae_opt = torch.optim.Adam(autoencoder.parameters(), lr=config.learning_rate)
    for _ in range(config.autoencoder_epochs):
        recon = autoencoder(high_t)
        loss = torch.mean((recon - high_t) ** 2)
        ae_opt.zero_grad()
        loss.backward()
        ae_opt.step()

    with torch.no_grad():
        latent = autoencoder.encoder(high_t)
        reconstruction_mse = float(torch.mean((autoencoder(high_t) - high_t) ** 2).item())

    oracle = mpc_grid_policy(env_config, lookahead=3, grid_points=config.action_levels)
    action_grid = np.linspace(-1.0, 1.0, config.action_levels, dtype=np.float32)
    labels = []
    for state in base_states:
        a = float(oracle(state))
        labels.append(int(np.argmin(np.abs(action_grid - a))))
    y = torch.as_tensor(labels, dtype=torch.long)

    split = int(0.8 * len(high))
    raw_policy = PolicyClassifier(config.high_dim, config.action_levels)
    latent_policy = PolicyClassifier(config.latent_dim, config.action_levels)
    _fit_classifier(raw_policy, high_t[:split], y[:split], config.policy_epochs, config.learning_rate)
    _fit_classifier(latent_policy, latent[:split].detach(), y[:split], config.policy_epochs, config.learning_rate)

    with torch.no_grad():
        raw_acc = float((raw_policy(high_t[split:]).argmax(1) == y[split:]).float().mean().item())
        latent_acc = float((latent_policy(latent[split:]).argmax(1) == y[split:]).float().mean().item())

    return LatentControlModels(
        observation_map=obs_map,
        autoencoder=autoencoder,
        raw_policy=raw_policy,
        latent_policy=latent_policy,
        action_grid=action_grid,
        reconstruction_mse=reconstruction_mse,
        raw_accuracy=raw_acc,
        latent_accuracy=latent_acc,
    )


def _deployed_policy(models: LatentControlModels, *, latent: bool) -> Callable[[np.ndarray], float]:
    def choose(state: np.ndarray) -> float:
        high = torch.as_tensor(models.observation_map.transform(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if latent:
                features = models.autoencoder.encoder(high)
                logits = models.latent_policy(features)
            else:
                logits = models.raw_policy(high)
        return float(models.action_grid[int(torch.argmax(logits, dim=1).item())])
    return choose


def evaluate_latent_control(
    models: LatentControlModels,
    env_config: EnergyProductionConfig = EnergyProductionConfig(),
    *,
    episodes: int = 150,
    seed: int = 140_000,
) -> dict[str, dict[str, float] | float]:
    return {
        "reconstruction_mse": models.reconstruction_mse,
        "raw_imitation_accuracy": models.raw_accuracy,
        "latent_imitation_accuracy": models.latent_accuracy,
        "raw_sensor_policy": evaluate_energy_policy(_deployed_policy(models, latent=False), env_config, episodes=episodes, seed=seed),
        "latent_policy": evaluate_energy_policy(_deployed_policy(models, latent=True), env_config, episodes=episodes, seed=seed),
    }
