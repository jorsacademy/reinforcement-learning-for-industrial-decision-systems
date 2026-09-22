import numpy as np

from safe_production_control.controllers import AggressiveController, SafetyRuleController


def test_safe_controller_throttles_high_temperature():
    controller = SafetyRuleController()
    cool = np.array([0.4, 0.6, 0.4, 0.7, 0.3, 0.6], dtype=np.float32)
    hot = np.array([0.4, 0.6, 0.9, 0.7, 0.3, 0.6], dtype=np.float32)
    assert controller.act(hot)[0] < controller.act(cool)[0]


def test_safe_controller_respects_utilization_guard():
    action = SafetyRuleController().act(np.array([0.3, 1.0, 0.4, 1.0, 0.2, 1.0], dtype=np.float32))[0]
    assert 0.0 <= action <= 0.93


def test_aggressive_controller_is_more_forceful_in_backlog_case():
    obs = np.array([0.3, 0.8, 0.5, 0.8, 0.3, 0.5], dtype=np.float32)
    assert AggressiveController().act(obs)[0] > SafetyRuleController().act(obs)[0]
