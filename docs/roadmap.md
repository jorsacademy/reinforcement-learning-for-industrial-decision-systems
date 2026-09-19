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

Implemented risk-aware extension:

- empirical VaR/CVaR reporting for rare-demand inventory;
- mean-optimal versus CVaR-optimal base-stock policy search;
- tail-weighted PPO as a transparent worst-episode training heuristic.

Remaining:

- formal chance-constrained policy optimization;
- continuous-action constrained RL for production/energy systems;
- stronger statistical validation of constraint satisfaction across training seeds.

## Partial observability / maintenance benchmark (implemented)

- hidden four-state equipment degradation model;
- noisy categorical condition sensor;
- exact Bayesian belief filtering;
- reactive sensor and posterior-threshold maintenance rules;
- discretized finite-horizon belief-state dynamic programming;
- belief-state DQN that never receives hidden health labels;
- held-out cost, failure exposure, intervention-rate, and belief-entropy evaluation.

This extends the separate fully observed maintenance MDP project rather than duplicating it.

## Phase 4 — offline RL (in progress)

Implemented:

- fixed logged inventory trajectories with no simulator access during fitting;
- behavior-policy and state/action coverage diagnostics;
- tabular behavior cloning;
- pessimistic tabular fitted-Q iteration;
- neural discrete CQL-style conservative offline Q-learning;
- target-policy support reporting;
- Fitted Q Evaluation (FQE) from logged transitions before simulator testing.

Additional implemented scheduling extension:

- dynamic shop-floor dispatch logs with stochastic arrivals, due dates, priorities, and family setups;
- offline behavior cloning and CQL-style rule selection;
- behavior-probability support guardrails;
- neural FQE before simulator testing;
- fixed FIFO/EDD/SPT/ATC baselines;
- rolling-horizon local permutation search as a model-based scheduling reference.

Remaining:

- IQL-style offline policy learning;
- richer off-policy evaluation diagnostics;
- offline RL on maintenance intervention logs;
- dataset-shift and behavior-policy sensitivity studies.

## Phase 5 — model-based and hybrid decision systems

- learned dynamics + model predictive control;
- RL selecting among optimization/heuristic operators;
- RL warm-starting or parameterizing mathematical optimization;
- digital-twin policy evaluation under distribution shift.

## Planned industrial benchmark families

- multi-echelon inventory;
- predictive maintenance with partial observability — implemented;
- energy-aware production;
- dynamic workforce/capacity allocation;
- rolling-horizon scheduling with stochastic arrivals — implemented as an offline-log benchmark;
- hybrid RL + OR decision systems.
