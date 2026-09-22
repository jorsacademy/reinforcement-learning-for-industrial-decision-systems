from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from .simulation import evaluate_policy

RUNG_SEEDS = (
    (101,),
    (101, 202),
    (101, 202, 303),
    (101, 202, 303, 404),
)


@dataclass(frozen=True)
class PolicySearchResult:
    reorder_point: int
    order_quantity: int
    cost: float
    trials: int


def policy_trainable(config: dict[str, int]) -> None:
    """Iteratively evaluates one policy so ASHA can prune weak trials."""
    r = int(config["reorder_point"])
    q = int(config["order_quantity"])
    for seeds in RUNG_SEEDS:
        cost = evaluate_policy(r, q, seeds=seeds)
        tune.report({"cost": cost})


def build_scheduler() -> ASHAScheduler:
    return ASHAScheduler(
        time_attr="training_iteration",
        max_t=len(RUNG_SEEDS),
        grace_period=1,
        reduction_factor=2,
    )


def run_tune_search(
    *,
    num_samples: int = 8,
    seed: int = 7,
    local_dir: str | None = None,
) -> PolicySearchResult:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    storage = local_dir or tempfile.mkdtemp(prefix="ray-policy-")
    tuner = tune.Tuner(
        tune.with_resources(policy_trainable, {"cpu": 1}),
        param_space={
            "reorder_point": tune.randint(0, 16),
            "order_quantity": tune.randint(2, 21),
        },
        tune_config=tune.TuneConfig(
            metric="cost",
            mode="min",
            scheduler=build_scheduler(),
            search_alg=BasicVariantGenerator(random_state=seed),
            num_samples=num_samples,
            max_concurrent_trials=2,
        ),
        run_config=tune.RunConfig(
            name=f"inventory-policy-{seed}",
            storage_path=str(Path(storage).resolve()),
            verbose=0,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="cost", mode="min")
    r = int(best.config["reorder_point"])
    q = int(best.config["order_quantity"])
    validation_cost = evaluate_policy(r, q, seeds=(1001, 1002, 1003, 1004))
    return PolicySearchResult(r, q, validation_cost, len(results))
