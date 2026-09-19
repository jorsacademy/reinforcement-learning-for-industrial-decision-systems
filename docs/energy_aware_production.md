# Energy-Aware Production with Continuous-Action SAC

This benchmark represents a production-planning problem in which the operational decision is naturally continuous rather than a discrete dispatch choice.

## Industrial decision

At each planning period the controller observes:

- current net inventory or backlog;
- current electricity/energy price;
- current point in the planning horizon;
- the known seasonal mean-demand signal.

It selects a continuous production quantity:

```text
0 <= production_t <= maximum production rate
```

Demand is stochastic. Energy price follows a bounded mean-reverting stochastic process.

## Cost model

The benchmark charges:

- direct production cost;
- energy cost;
- holding cost;
- backlog/shortage cost.

Energy consumption is convex in production rate:

```text
energy(q) = a*q + b*q^2
```

so pushing production into one expensive period can be economically undesirable even when more inventory would reduce future shortage risk.

## Why SAC?

SAC is used because the action is continuous and the future value of producing now depends on inventory, expected demand, and stochastic future energy prices.

The implementation includes:

- squashed Gaussian actor;
- twin critics;
- target critics;
- replay buffer;
- soft target updates;
- entropy-regularized actor objective;
- Huber-free MSE critic targets with gradient clipping.

The entropy coefficient is fixed in this first benchmark for transparency. Automatic temperature tuning can be added later.

## Operational baselines

### Myopic economic dispatch

Selects the production quantity minimizing immediate expected cost under mean demand.

### Rolling-horizon grid MPC

Enumerates a short sequence of candidate production rates, propagates expected demand and mean-reverting energy-price forecasts, and executes only the first production decision before replanning next period.

This is intentionally stronger than a do-nothing or constant-rate baseline.

## Evaluation

Policies are evaluated on the same held-out stochastic seed range and report:

- mean total cost;
- p90 total cost;
- mean energy cost;
- mean final backlog;
- mean total production.

The benchmark does not assume SAC should beat MPC. If the rolling-horizon operational policy is better, that result is retained.
