from ray.tune.schedulers import ASHAScheduler

from ray_policy.tuning import RUNG_SEEDS, build_scheduler, policy_trainable, run_tune_search


def test_scheduler_and_rungs():
    scheduler = build_scheduler()
    assert isinstance(scheduler, ASHAScheduler)
    assert len(RUNG_SEEDS) == 4
    assert [len(x) for x in RUNG_SEEDS] == [1, 2, 3, 4]


def test_policy_trainable_reports_all_rungs(monkeypatch):
    reports = []
    monkeypatch.setattr("ray_policy.tuning.tune.report", lambda metrics: reports.append(metrics))
    policy_trainable({"reorder_point": 4, "order_quantity": 9})
    assert len(reports) == 4
    assert all(report["cost"] > 0 for report in reports)


def test_ray_tune_runs_end_to_end(tmp_path):
    result = run_tune_search(num_samples=2, local_dir=str(tmp_path))
    assert 0 <= result.reorder_point <= 15
    assert 2 <= result.order_quantity <= 20
    assert result.cost > 0
    assert result.trials >= 1


def test_invalid_sample_budget():
    try:
        run_tune_search(num_samples=0)
    except ValueError as exc:
        assert "num_samples" in str(exc)
    else:
        raise AssertionError("expected ValueError")
