# Production Control with MPC vs RL

Industrial production-control benchmark comparing **rule-based control**, a lightweight **receding-horizon MPC**, **PPO**, and **SAC** on the same stochastic manufacturing environment.

The purpose of the repository is not to show that reinforcement learning always wins. It is to compare control paradigms under identical plant dynamics and business KPIs.

## Problem

A production line faces stochastic demand and time-varying electricity prices. At every decision period the controller selects a normalized production-rate command `u_t in [0,1]`.

The plant must balance:

- throughput and service level,
- work-in-process inventory,
- backlog/tardiness pressure,
- electricity cost,
- aggressive control changes.

A simplified state is

`[WIP, backlog, demand, energy price, previous control, time]`.

The continuous action is the fraction of maximum production rate to request.

The per-period objective penalizes

`holding cost + backlog cost + energy cost + control smoothness cost`.

## Controllers

### Rule-based

A base-stock-style heuristic increases production when backlog grows or WIP falls below its target.

### MPC

A dependency-light receding-horizon controller evaluates a grid of candidate production commands using a local predictive model and chooses the command with minimum predicted cost.

This is intentionally transparent rather than a full industrial nonlinear MPC implementation. It provides a meaningful model-based baseline without requiring a commercial solver.

### PPO

Stable-Baselines3 PPO learns a continuous control policy from interactions with the plant.

### SAC

Stable-Baselines3 SAC is the main off-policy continuous-control RL benchmark. It is particularly relevant when sample reuse and continuous action optimization are important.

## KPIs

All controllers are evaluated using the same seeded scenarios and the same metrics:

- total cost,
- throughput,
- utilization proxy,
- final backlog,
- WIP,
- energy cost,
- holding cost,
- backlog cost,
- control smoothness cost.

A good controller should not be selected from reward alone. The KPI decomposition shows *why* one controller performs better.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── production_control/
│       ├── environment.py
│       ├── controllers.py
│       ├── train_rl.py
│       └── evaluate.py
├── tests/
│   ├── test_environment.py
│   └── test_controllers.py
└── .github/workflows/ci.yml
```

## Installation

Classical control, evaluation and tests:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[test]'
```

For PPO/SAC experiments install the optional RL dependencies:

```bash
pip install -e '.[rl,test]'
```

## Evaluate classical controllers

```bash
python -m production_control.evaluate --episodes 30
```

Results are written to `results/comparison.csv`.

## Train SAC

```bash
python -m production_control.train_rl --algo sac --timesteps 100000
```

## Train PPO

```bash
python -m production_control.train_rl --algo ppo --timesteps 100000
```

## Compare trained RL agents with MPC and rule-based control

```bash
python -m production_control.evaluate \
  --episodes 50 \
  --ppo-model artifacts/ppo_production_control.zip \
  --sac-model artifacts/sac_production_control.zip
```

## Experimental design

For a defensible comparison:

1. train PPO and SAC on a fixed training distribution,
2. evaluate all controllers on identical unseen random seeds,
3. compare mean and variance across many episodes,
4. report the complete KPI decomposition rather than only cumulative reward,
5. test distribution shift such as demand surges or higher peak electricity prices.

## Why this matters for industrial engineering

This project connects several IE areas in one benchmark:

- production planning and control,
- inventory/WIP management,
- energy-aware manufacturing,
- model predictive control,
- reinforcement learning,
- simulation-based optimization,
- multi-objective operational KPIs.

The key engineering question is not "Which RL algorithm is best?" but "Under which plant uncertainty, model quality, response-time and data constraints should a rule, MPC or learned policy be used?"

## Research extensions

Useful next steps include:

- multi-stage production lines and bottlenecks,
- machine failures and maintenance decisions,
- explicit due-date/tardiness orders,
- nonlinear energy demand charges,
- constrained/safe RL,
- domain randomization and sim-to-real testing,
- nonlinear MPC with CasADi,
- MILP-based finite-horizon production control,
- robust MPC under uncertain forecasts,
- Pareto analysis across throughput, WIP and energy,
- statistical significance tests across evaluation seeds.

## CI

GitHub Actions tests Python 3.10, 3.11 and 3.12. The workflow installs only the lightweight classical/test dependencies, runs unit tests, and executes a fast MPC/rule-based evaluation smoke test. Long PPO/SAC training is intentionally excluded from CI.
