# Evaluation Protocol

Industrial RL should be evaluated as a decision method, not only as a learning curve.

## 1. Define the decision problem first

For every experiment, state explicitly:

- state variables;
- actions;
- transition uncertainty;
- horizon;
- economic objective;
- hard constraints;
- operational KPIs.

A reward function is an implementation device. It is not a substitute for a clear operational objective.

## 2. Use strong non-RL baselines

Each experiment should include the strongest transparent reference that is computationally reasonable:

- exact dynamic programming for small finite MDPs;
- base-stock or threshold policies;
- rolling-horizon LP/MILP/CP-SAT;
- domain heuristics;
- behavior policy for offline RL.

Weak baselines make RL results difficult to interpret.

## 3. Separate fitting and evaluation randomness

Training seeds and evaluation seeds must be disjoint. Policies compared in the same benchmark should see the same evaluation trajectories whenever possible (common random numbers).

## 4. Report decision metrics

At minimum report metrics tied to the industrial objective:

- mean cost or reward;
- tail cost such as p90/p95;
- service/fill rate where relevant;
- stockout/backlog metrics;
- constraint violation rate;
- resource use such as overtime;
- inference/decision latency for real-time claims.

Training reward alone is not an operational result.

## 5. Treat exact references carefully

An exact DP or mathematical-programming result is exact only for the declared model and numerical formulation. It does not prove that the model represents a real plant correctly.

## 6. Treat offline RL carefully

An offline learner must not interact with the simulator during fitting. Evaluation can use the simulator only after the policy is frozen. Unsupported actions are a major risk; this repository's first offline baseline uses count-based pessimism and is explicitly not presented as a full CQL or IQL implementation.

## 7. Negative results are valid

If a heuristic or optimizer outperforms RL, keep the result. The correct industrial conclusion may be that the simpler policy is better for the declared operating regime.
