# When Should an Industrial Engineer Use Reinforcement Learning?

Reinforcement learning is not a default replacement for mathematical optimization, dynamic programming, simulation, or control. It is most plausible when the operational problem is genuinely sequential and decisions change future states.

## A practical decision test

RL becomes more defensible when several of the following are true:

1. **Repeated decisions matter.** A decision changes inventory, backlog, equipment condition, queue composition, or another state that affects later decisions.
2. **The exact transition model is unavailable or expensive.** The system can be sampled through historical trajectories or a simulator, but writing a complete transition model is difficult.
3. **The decision must be made repeatedly and quickly.** Expensive optimization can be shifted into an offline training stage if a learned policy is sufficiently reliable.
4. **The environment is stochastic.** Demand, processing time, failures, arrivals, prices, or yields vary over time.
5. **A simulator or logged dataset exists.** Online exploration in a real factory is rarely an acceptable starting point.

RL should be treated cautiously when an exact LP/MILP/CP-SAT/DP model is small enough to solve reliably at decision time. In that case, the exact or optimization-based policy is an important reference and may remain the operational choice.

## Industrial problem classes

### Inventory and replenishment

Useful RL questions include state-dependent ordering, lost-sales systems, non-stationary demand, large multi-echelon state spaces, and decisions with delayed consequences.

Start with a base-stock policy and exact dynamic programming on small instances before introducing Q-learning, DQN, or offline RL.

### Production and capacity allocation

RL can be useful when capacity decisions repeat under uncertain demand, energy prices, changeovers, failures, or uncertain processing times.

Constraints should not be hidden inside an arbitrary reward coefficient. Report constraint violations explicitly and compare with a solver or a policy that enforces feasibility by construction.

### Maintenance and reliability

A maintenance problem naturally becomes an MDP when equipment condition evolves over time and maintenance changes that evolution. If true health is not directly observed, the problem moves toward partial observability rather than a fully observed MDP.

### Scheduling

RL is plausible for repeated dispatch decisions under dynamic arrivals and disruptions. Strong dispatch rules and rolling-horizon mathematical optimization are mandatory baselines. A learned policy that only beats FIFO is not strong evidence.

### Energy-aware operations

Continuous decisions such as production rate, charging power, thermal setpoints, or flexible load allocation can motivate actor-critic methods such as PPO or SAC. Industrial constraints and safe operating envelopes remain part of the decision problem.

## What this repository does

The repository begins with three deliberately small benchmarks where exact or transparent references are available:

- finite-horizon inventory control;
- production/capacity allocation with an episode-wide overtime budget;
- offline inventory learning from logged trajectories.

The purpose is to learn when RL adds value, not to assume that RL should win.
