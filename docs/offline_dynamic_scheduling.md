# Offline Dynamic Production Scheduling from Shop-Floor Logs

This benchmark studies dynamic dispatching when online exploration is not available.

Jobs arrive over time to a single production resource. Each job has:

- release time;
- processing time;
- due date;
- product family;
- tardiness weight.

Changing family creates sequence-dependent setup time.

## Decision structure

The offline learner does not select a changing job identifier directly. It selects one dispatching rule:

```text
EDD
SPT
Setup-aware
ATC
```

The selected rule then chooses the next available job.

This fixed action space is suitable for historical dispatch logs and avoids an unbounded job-ID action space.

## Historical behavior

The synthetic historical log is ATC-heavy, with smaller probabilities assigned to EDD, setup-aware, and SPT decisions.

The dataset records:

```text
state features
+ selected dispatch rule
+ immediate objective reward
+ next state
+ terminal flag
```

Offline behavior cloning and a conservative CQL-style neural Q learner are fitted only to this table.

## Industrial objective

The per-job objective increment is:

```text
weighted tardiness
+ setup_cost_weight * setup_time
```

Evaluation also reports:

- p90 objective;
- total weighted tardiness;
- setup time;
- mean flow time;
- on-time completion rate.

## Baselines

Fixed dispatch rules:

- Earliest Due Date (EDD);
- Shortest Processing Time (SPT);
- setup-aware dispatch;
- Apparent Tardiness Cost (ATC).

A two-step current-information lookahead baseline also evaluates the rule-induced first dispatch and the best available second dispatch over the current queue.

The lookahead is intentionally described as a short-horizon heuristic, not as a globally optimal rolling-horizon schedule.

## Relation to the other scheduling repositories

The separate job-shop repository solves a classical JSSP with PPO and CP-SAT benchmarking.

The separate automotive paint-shop repository trains an online PPO hyper-heuristic in an event-driven simulation.

This benchmark addresses a different question:

```text
historical shop-floor dispatch logs
        ↓
offline policy learning
        ↓
frozen dispatch-rule selector
        ↓
held-out dynamic scheduling evaluation
```

## Scope

The benchmark is a stylized single-machine dynamic dispatching model. It does not yet include parallel machines, machine eligibility, preventive maintenance windows, material constraints, or exact rolling-horizon MILP/CP-SAT optimization.

Those are natural future extensions.
