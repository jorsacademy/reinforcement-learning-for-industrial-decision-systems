# Predictive Maintenance under Partial Observability

This benchmark extends the fully observed maintenance-MDP idea into a partially observable decision problem.

The physical equipment health is hidden:

```text
Healthy -> Degraded -> Critical -> Failed
```

The decision-maker does not observe that state directly. Instead, a noisy sensor emits one of:

```text
Good
Warning
Alarm
Failure signal
```

## Why this matters for Industrial Engineering

Condition-based maintenance often relies on imperfect inspections, vibration features, temperatures, alarms, and health scores rather than a perfectly observed degradation state.

A maintenance policy therefore needs to distinguish:

```text
physical state
vs
information available to the planner
```

The repository represents that uncertainty with a Bayesian belief state.

## Decision sequence

At each review period:

1. the planner has a posterior belief over the four hidden health states;
2. it selects `Operate`, `Minor maintenance`, or `Replace`;
3. the hidden asset condition changes according to the action-specific transition matrix;
4. a noisy sensor observation is generated;
5. the belief is updated with Bayes' rule;
6. the next maintenance decision uses that posterior.

The RL policy never receives the hidden health state.

## Bayesian filter

For action `a` and current belief `b`:

```text
predicted belief:
b_minus = b T_a
```

After observing sensor signal `o`:

```text
b_next(s)
proportional to
P(o | s) * b_minus(s)
```

The normalized posterior is used as the belief-MDP state.

## Compared policies

### Always operate

A deliberately weak run-to-failure-style sanity baseline.

### Reactive sensor rule

Uses only the current sensor category:

```text
Good    -> Operate
Warning -> Minor maintenance
Alarm   -> Replace
Failure -> Replace
```

### Belief-threshold rule

Uses posterior degradation/severity probabilities rather than the raw sensor label.

This illustrates the value of information aggregation without requiring RL.

### Discretized belief dynamic programming

The continuous four-state belief simplex is discretized onto a finite probability grid.

Finite-horizon dynamic programming is then run on that grid. Posterior beliefs are projected to the nearest grid point.

This is a model-based approximate POMDP reference. It is **not** an exact POMDP solution, because the belief space has been discretized.

### Belief-state DQN

DQN receives:

```text
normalized period
+ posterior P(Healthy)
+ posterior P(Degraded)
+ posterior P(Critical)
+ posterior P(Failed)
```

The hidden health state is never included in the network input.

## Evaluation

Held-out simulation reports:

- mean lifecycle cost;
- p90 lifecycle cost;
- fraction of periods in the failed state;
- replacement frequency;
- minor-maintenance frequency;
- average posterior entropy.

Posterior entropy is an information-state diagnostic, not an operational objective.

## Relation to the separate fully observed maintenance repository

The separate `industrial-maintenance-markov-decision-process-python` project assumes the health state is observed directly and solves that finite MDP with exact methods.

This benchmark addresses the next modeling layer:

```text
fully observed MDP
        ↓
noisy condition observations
        ↓
Bayesian belief state
        ↓
POMDP / belief-MDP decision making
```

The two projects therefore complement rather than duplicate each other.

## Scope

Transition probabilities, sensor confusion probabilities, and costs are stylized engineering assumptions.

A production maintenance application would require those quantities to be estimated and validated from reliability data, inspection data, sensor histories, maintenance records, and failure consequences.
