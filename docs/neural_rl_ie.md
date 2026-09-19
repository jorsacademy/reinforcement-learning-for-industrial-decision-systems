# Neural RL for Industrial Engineering Benchmarks

Phase 2 introduces function approximation while preserving the repository's original evaluation discipline.

## Why DQN here?

DQN is used on a regime-switching inventory problem where the state includes:

- decision period;
- on-hand inventory;
- observed demand regime.

The action is a discrete order quantity. The benchmark retains an exact dynamic-programming reference because the finite state space is still tractable. This makes it possible to evaluate approximation error rather than treating training reward as evidence of quality.

Implementation details:

- two-hidden-layer Q-network;
- experience replay;
- target network;
- epsilon-greedy exploration;
- Huber loss;
- gradient clipping;
- feasibility masking so the network cannot order beyond storage capacity.

## Why PPO here?

PPO is used for dynamic flex-workforce allocation across three work centers.

At each period the policy observes:

- current period;
- backlog at each work center;
- expected workload at each work center.

The action is one feasible allocation of a limited flex-worker pool across the three centers.

The comparison baseline is a one-step expected-cost allocation rule that explicitly evaluates every feasible flex allocation using expected workload. This is materially stronger than comparing PPO only with "no flex workers".

Implementation details:

- categorical actor;
- state-value critic;
- clipped PPO objective;
- Monte Carlo finite-horizon returns;
- normalized advantages;
- entropy regularization;
- gradient clipping.

## What is deliberately not claimed

The DQN and PPO implementations are compact research/teaching implementations, not production frameworks.

The PPO workforce benchmark is not an exact constrained MDP solution. Flex-pool feasibility is encoded in the action set, but the benchmark does not claim global optimality.

The DQN benchmark can be compared with exact DP only because this first neural inventory model remains finite and small enough to solve exactly.

## Evaluation rule

Neural RL is considered useful only if it adds value relative to a credible operational baseline on held-out stochastic trajectories. Training return is diagnostic, not the final KPI.
