# Constrained and Safe RL for Industrial Workforce Allocation

This benchmark treats an operational service requirement as a separate constraint rather than burying it inside one reward coefficient.

## Problem

Three work centers receive stochastic workloads. A limited flex-worker pool can be reassigned every period.

The economic objective is to reduce flexible-labor cost while avoiding excessive work-in-process/backlog.

## CMDP formulation

The environment returns two signals at every decision:

### Economic reward

```text
reward
=
- (
    flex labor cost
    + small soft backlog cost
    + overflow cost
  )
```

### Constraint cost

```text
constraint_cost = 1
if any center backlog > service_backlog_limit
else 0
```

The default benchmark limits the **expected cumulative number of violation periods per episode**.

This is an expected-cost CMDP constraint. It is not described as a chance constraint or a formal service-level guarantee.

## Primal-dual PPO

The policy uses:

- one categorical actor;
- one reward-value critic;
- one constraint-value critic;
- PPO clipping;
- entropy regularization;
- projected non-negative Lagrange multiplier.

The actor is trained using a combined advantage:

```text
A_reward - lambda * A_constraint
```

After every rollout batch:

```text
lambda <- projection[
    lambda
    + dual_step
      * (observed constraint return - constraint budget)
]
```

The dual update therefore reacts to explicit service-constraint performance rather than a fixed hand-tuned reward penalty.

## Safety shield

A separate action-repair layer checks the proposed action using:

- current backlog;
- expected next-period workload;
- declared service backlog threshold.

If the action is expected to violate the threshold and another feasible allocation avoids that violation, the shield substitutes the lowest-flex expected-feasible allocation.

The shield is deliberately reported separately from the RL algorithm. This makes it possible to distinguish:

```text
policy learning quality
vs
safety-repair intervention
```

## Baselines

The benchmark compares:

1. no flex workers;
2. constrained expected-workload allocation;
3. raw primal-dual PPO;
4. the same PPO policy with the safety shield.

The constrained expected-workload policy is a strong one-step operations baseline: it enumerates every feasible workforce allocation and chooses the lowest-flex option expected to remain within the service threshold.

## Reported metrics

- mean economic cost;
- p90 economic cost;
- expected violation periods per episode;
- gap relative to the declared constraint budget;
- probability an episode contains at least one violation;
- average flex-worker use;
- safety-shield intervention rate.

## Interpretation

A policy can have a low economic cost and still be unacceptable if it violates the service constraint.

Conversely, a shielded policy can meet the service target by intervening frequently. In that case the safety layer, not the learned policy alone, deserves credit.

This distinction is central to industrial RL deployment.
