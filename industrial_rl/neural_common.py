from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([torch.nn.Linear(prev, int(hidden)), torch.nn.Tanh()])
        prev = int(hidden)
    layers.append(torch.nn.Linear(prev, int(output_dim)))
    return torch.nn.Sequential(*layers)
