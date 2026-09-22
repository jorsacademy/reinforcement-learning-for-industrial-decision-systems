from production_control.controllers import MPCController, RuleBasedController
from production_control.environment import ProductionControlEnv


def _rollout(controller, seed=3):
    env = ProductionControlEnv()
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = controller.act(obs)
        assert env.action_space.contains(action)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def test_rule_based_controller_completes_episode():
    info = _rollout(RuleBasedController())
    assert info["throughput"] > 0
    assert info["total_cost"] >= 0


def test_mpc_controller_completes_episode():
    info = _rollout(MPCController())
    assert info["throughput"] > 0
    assert info["total_cost"] >= 0
