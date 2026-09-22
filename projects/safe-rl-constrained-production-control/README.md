# Safe RL for Constrained Production Control

Industrial production-control benchmark with explicit safety constraints, classical safety-aware control, and a PPO-Lagrangian-style reinforcement-learning path.

The central question is not only whether a controller maximizes operational reward, but whether it does so while respecting a measurable safety envelope.

## Problem formulation

A production line receives stochastic demand and electricity prices. At each period the controller selects a normalized production command `u_t in [0, 1]`.

The state contains:

- work-in-process (WIP),
- backlog,
- process temperature,
- stochastic demand,
- electricity price,
- previous control command.

The action is a continuous production-rate command.

The environment tracks three explicit constraints:

- process temperature below a maximum threshold,
- WIP below a maximum storage/process threshold,
- utilization below a maximum safe utilization level.

The operational objective penalizes holding cost, backlog, electricity consumption, control movement, and constraint violations.

## Why safe RL matters

In industrial systems, a high-return policy that violates process limits is not deployable. The benchmark therefore reports constraint cost separately from economic reward.

A policy is evaluated with both operational and safety KPIs:

- cumulative return,
- total operating cost,
- throughput,
- final WIP,
- final backlog,
- energy cost,
- constraint violations per episode,
- violation rate,
- cumulative violation severity.

## Controllers

### Safety rule controller

A transparent heuristic throttles production as temperature or WIP approaches the safety envelope. It also reacts to backlog, demand, and electricity prices.

### Aggressive controller

A throughput-seeking controller acts as a deliberately less conservative reference. It is useful for exposing the trade-off between production performance and constraint violations.

### PPO-Lagrangian-style controller

`train_lagrangian.py` wraps the environment with an additional multiplier on explicit constraint cost and trains Stable-Baselines3 PPO.

The shaped objective is conceptually

`reward_safe = reward_operational - lambda * constraint_cost`.

This is a practical **PPO-Lagrangian-style benchmark**, not a full primal-dual constrained policy-optimization implementation. A full implementation would adapt the Lagrange multiplier online to satisfy a target cost budget and may use a separate constraint critic.

## Installation

Core environment, baselines, evaluation, and tests:

```bash
pip install -e '.[dev]'
```

RL training:

```bash
pip install -e '.[rl,dev]'
```

## Evaluate classical controllers

```bash
python -m safe_production_control.evaluate --policy safe_rule --episodes 50
python -m safe_production_control.evaluate --policy aggressive --episodes 50
```

## Train PPO-Lagrangian-style policy

```bash
python -m safe_production_control.train_lagrangian \
  --timesteps 100000 \
  --lambda-constraint 10
```

Evaluate the trained model:

```bash
python -m safe_production_control.evaluate \
  --policy ppo \
  --model models/ppo_lagrangian_safe_control.zip \
  --episodes 50
```

## Experimental design

For a defensible study:

1. train on a fixed stochastic operating distribution,
2. evaluate all controllers on identical unseen seeds,
3. report economic reward and safety metrics separately,
4. sweep the Lagrangian multiplier to build a reward-safety trade-off curve,
5. stress-test higher demand, cooling degradation, and energy-price shocks,
6. define an acceptable constraint budget before selecting a controller.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/safe_production_control/
│   ├── environment.py
│   ├── controllers.py
│   ├── evaluate.py
│   └── train_lagrangian.py
├── tests/
│   ├── test_environment.py
│   └── test_controllers.py
└── .github/workflows/ci.yml
```

## Research extensions

Useful next steps include:

- adaptive primal-dual PPO-Lagrangian,
- separate reward and cost critics,
- constrained policy optimization (CPO),
- CVaR and chance-constrained RL,
- control-barrier-function safety filters,
- shielded RL and action projection,
- safe offline RL from historian/SCADA data,
- partial observability and sensor noise,
- multiple process units and coupled constraints,
- MPC safety filter combined with RL,
- sim-to-real/domain randomization studies,
- Pareto-front analysis between throughput and safety risk.

## CI

GitHub Actions validates Python 3.10, 3.11, and 3.12. CI installs only the lightweight core and development dependencies, runs unit tests, and smoke-tests the safety-aware baseline evaluation. Long PPO training is intentionally excluded.
