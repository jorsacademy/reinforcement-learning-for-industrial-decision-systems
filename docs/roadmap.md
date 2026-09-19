# Roadmap

The first release emphasizes transparent small-state benchmarks. Future additions should increase modeling complexity only when the evaluation protocol remains auditable.

## Phase 1 — transparent foundations (implemented)

- exact finite-horizon dynamic programming;
- base-stock inventory baseline;
- tabular Q-learning;
- constrained capacity allocation with a hard overtime budget;
- shadow-price / Lagrangian-shaped Q-learning;
- offline behavior cloning;
- pessimistic tabular fitted-Q learning.

## Phase 2 — function approximation (implemented)

Implemented:

- DQN for regime-switching inventory with a neural Q-network, replay buffer, target network, and feasibility masking;
- PPO for dynamic flex-workforce allocation across three work centers;
- SAC for continuous energy-aware production-rate decisions with twin critics and short-horizon MPC-style comparison.

Future neural expansion:

- larger masked-action scheduling/resource-allocation benchmarks;
- multi-output or recurrent policies where partial history is operationally meaningful.

Every neural benchmark should retain an exact or optimization reference on a smaller validation regime.

## Phase 3 — constrained and safe RL (in progress)

Implemented:

- CMDP formulation for stochastic flex-workforce allocation;
- explicit reward critic and constraint critic;
- primal-dual / Lagrangian PPO with projected dual updates;
- expected service-violation budget reported separately from economic reward;
- safety shield / action repair based on expected next-period backlog;
- shield intervention rate reported separately from raw policy quality.

Remaining:

- chance-constrained or CVaR-aware evaluation;
- continuous-action constrained RL for production/energy systems;
- stronger statistical validation of constraint satisfaction across training seeds.

## Phase 4 — offline RL

- CQL;
- IQL;
- behavior-policy diagnostics;
- dataset coverage analysis;
- off-policy evaluation before simulator testing.

## Phase 5 — model-based and hybrid decision systems

- learned dynamics + model predictive control;
- RL selecting among optimization/heuristic operators;
- RL warm-starting or parameterizing mathematical optimization;
- digital-twin policy evaluation under distribution shift.

## Planned industrial benchmark families

- multi-echelon inventory;
- predictive maintenance with partial observability;
- energy-aware production;
- dynamic workforce/capacity allocation;
- rolling-horizon scheduling with stochastic arrivals;
- hybrid RL + OR decision systems.
