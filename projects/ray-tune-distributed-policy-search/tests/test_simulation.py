import pytest

from ray_policy.simulation import InventoryConfig, evaluate_policy


def test_policy_evaluation_is_deterministic():
    a = evaluate_policy(5, 10, seeds=(1, 2))
    b = evaluate_policy(5, 10, seeds=(1, 2))
    assert a == pytest.approx(b)
    assert a > 0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        evaluate_policy(-1, 5)
    with pytest.raises(ValueError):
        evaluate_policy(1, 0)
    with pytest.raises(ValueError):
        evaluate_policy(1, 5, seeds=())
    with pytest.raises(ValueError):
        evaluate_policy(1, 5, config=InventoryConfig(horizon=0))
