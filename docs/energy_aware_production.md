# Energy-Aware Production with Continuous Control

This benchmark studies a continuous Industrial Engineering decision: how much to produce in each period when demand and electricity prices vary, inventory/backlog carries over, and rapid production-rate changes are costly.

## State

The controller observes:

- normalized decision period;
- current net inventory / backlog;
- expected demand for the current period;
- current electricity price;
- previous production rate.

## Action

The RL action is continuous in `[-1, 1]` and is mapped to a physical production rate in:

```text
0 <= production_rate <= max_rate
```

## Cost model

The per-period cost includes:

```text
electricity price
  * (linear production energy + quadratic production energy)
+ holding cost
+ backlog cost
+ production-rate ramping cost
+ overflow / excessive-backlog penalty
```

The quadratic term represents increasing energy intensity at high production rates. This is a stylized educational process model, not a calibrated plant energy model.

## Baselines

### Constant-rate policy

Produces at approximately mean demand. This is intentionally simple and serves as a sanity check.

### Three-step deterministic MPC-like grid search

At every decision period the baseline searches a discrete production-rate grid over a short future horizon using expected demand and expected future electricity prices.

This is not a continuous global optimum, but it is a stronger IE baseline than a fixed rule because it explicitly trades:

- current energy price;
- future price pattern;
- inventory;
- backlog;
- ramping.

## SAC

The Soft Actor-Critic implementation contains:

- tanh-Gaussian stochastic actor;
- twin Q critics;
- twin target critics;
- replay buffer;
- soft target updates;
- entropy regularization;
- reparameterized actor gradients;
- gradient clipping.

The entropy coefficient is fixed in this compact implementation. Automatic entropy tuning is a possible extension.

## Evaluation

Policies are frozen before held-out evaluation. Reported metrics are:

- mean total cost;
- p90 total cost;
- mean energy use;
- final backlog;
- total production-rate movement.

A lower energy-use figure is not automatically better if it is achieved by creating backlog. Cost and service-related state metrics must be interpreted together.

## Scope

The environment represents energy-aware production planning, not low-level physical process control. It is appropriate for production-rate/capacity decisions over operational periods, not actuator-level temperature or pressure control.
