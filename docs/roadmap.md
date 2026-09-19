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

## Phase 2 — function approximation

- DQN for larger discrete inventory state spaces;
- PPO for dynamic discrete decision systems;
- SAC for continuous production/energy decisions;
- neural policies with action masking where feasibility is state-dependent.

Every neural benchmark should retain an exact or optimization reference on a smaller validation regime.

## Phase 3 — constrained and safe RL

- primal-dual/Lagrangian policy optimization;
- explicit cost critics;
- chance-constraint or CVaR-aware evaluation;
- safety shields or optimization-based action repair.

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
