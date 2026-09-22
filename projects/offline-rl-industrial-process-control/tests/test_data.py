import numpy as np

from offline_process_control.data import dataset_summary, generate_dataset


def test_dataset_shapes_and_terminals():
    data = generate_dataset(episodes=3, seed=5)
    n = len(data["observations"])
    assert n == 3 * 120
    assert data["observations"].shape == (n, 5)
    assert data["actions"].shape == (n, 1)
    assert data["next_observations"].shape == (n, 5)
    assert data["rewards"].shape == (n,)
    assert int(data["terminals"].sum()) == 3
    assert np.all(data["actions"] <= 0.8 + 1e-6)
    assert np.all(data["actions"] >= -0.8 - 1e-6)


def test_dataset_summary_is_finite():
    summary = dataset_summary(generate_dataset(episodes=2, seed=1))
    assert summary["transitions"] == 240.0
    assert np.isfinite(list(summary.values())).all()
