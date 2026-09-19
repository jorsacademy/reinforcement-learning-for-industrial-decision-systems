# Tail-Risk and CVaR-Aware Inventory Control

Industrial inventory decisions are often judged by more than average cost. Rare demand surges can create severe lost-sales or service failures even when mean performance looks acceptable.

This benchmark adds an explicit tail-risk view.

## Demand model

Each period has ordinary demand plus a rare surge component.

The environment is still synthetic and finite, so a risk-neutral exact dynamic-programming reference remains available.

## Reported risk metrics

For every frozen policy the held-out evaluator reports:

- mean total cost;
- p90 total cost;
- p95 total cost;
- empirical CVaR90 cost;
- fill rate;
- stockout-period rate.

For a cost sample `L`, the repository uses the empirical upper-tail definition:

```text
VaR_alpha = empirical alpha-quantile of L

CVaR_alpha
=
mean(
    L_i
    for L_i >= VaR_alpha
)
```

This is a finite-sample empirical tail metric. It is not a confidence interval.

## Parametric operations baselines

Two base-stock policies are selected using common random numbers:

1. **mean-optimal base-stock target** — minimizes empirical mean cost;
2. **CVaR-optimal base-stock target** — minimizes empirical CVaR90 cost.

Searching the same interpretable policy family under different objectives makes the risk/mean trade-off visible without adding neural complexity.

## Risk-neutral exact DP

The exact finite-horizon DP minimizes expected cost for the declared model.

It is an important reference but is not described as CVaR-optimal. Dynamic CVaR control is generally not equivalent to ordinary Bellman recursion over the original physical state alone.

## PPO comparison

The benchmark includes two PPO variants:

- standard mean-focused PPO;
- tail-weighted PPO.

The tail-weighted version identifies the empirical worst-cost episode tail in each rollout batch and increases the policy-gradient weight of transitions from those episodes.

This is intentionally called **tail-weighted PPO**, not an exact CVaR policy-gradient algorithm. It is a transparent heuristic for testing whether additional training emphasis on bad episodes improves held-out tail metrics.

## Interpretation

A risk-aware policy may accept a somewhat higher mean cost in exchange for lower p95/CVaR cost or fewer stockout periods.

A policy should not be called better solely because its CVaR is lower. The economic and service trade-off must be reported together.

## Scope

The benchmark represents single-echelon finite-horizon inventory under rare demand surges. It does not yet model:

- lead-time uncertainty;
- multi-echelon networks;
- substitution;
- perishability;
- correlated multi-SKU demand;
- formal chance constraints.

Those are natural later extensions.
