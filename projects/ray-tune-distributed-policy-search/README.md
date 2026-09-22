# Ray Tune Distributed Policy Search

A compact distributed hyperparameter-search example that tunes a stochastic inventory policy with **Ray Tune 2.58.0**.

The policy has two integer decisions:

- `reorder_point`: inventory level that triggers replenishment.
- `order_quantity`: replenishment quantity.

The simulator uses Poisson demand and evaluates every trial on the same seed schedule. The trainable reports four progressively stronger estimates; `ASHAScheduler` can stop weak trials early using Ray Tune's `training_iteration` metric.

## Why Ray Tune

This repository demonstrates the parts that distinguish Tune from a local optimizer:

- `tune.Tuner` orchestration;
- typed search spaces with `tune.randint`;
- per-trial resource declarations;
- `ASHAScheduler` early stopping;
- incremental `tune.report` metrics;
- a result grid and independent validation of the selected policy.

The example runs locally by default, but the same Tune program can run on a Ray cluster without changing the objective function.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
ray-tune-policy-demo
```

or:

```bash
python -m ray_policy.cli
```

## Test

```bash
pytest
```

GitHub Actions exercises Python 3.10–3.13 and enforces at least 90% coverage.

## Structure

- `src/ray_policy/simulation.py`: stochastic inventory simulator.
- `src/ray_policy/tuning.py`: Ray Tune trainable, ASHA scheduler, and search driver.
- `src/ray_policy/cli.py`: command-line demonstration.
- `tests/`: deterministic simulation, reporting contract, and real Tune integration tests.
