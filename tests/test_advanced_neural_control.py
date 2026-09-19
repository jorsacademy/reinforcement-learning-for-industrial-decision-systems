import numpy as np

from industrial_rl.latent_state_control import AutoEncoder, ObservationMap, PolicyClassifier
from industrial_rl.model_based_control import CEMConfig, DynamicsTrainingConfig, LearnedDynamicsEnsemble
from industrial_rl.sac_energy import EnergyProductionConfig
from industrial_rl.transfer_control import ProductionDQN, TransferDQNConfig, domain_randomization_sampler


def test_observation_map_and_autoencoder_shapes():
    mapper = ObservationMap(5, 64, seed=1)
    high = mapper.transform(np.zeros(5, dtype=np.float32))
    assert high.shape == (64,)
    ae = AutoEncoder(64, 8)
    assert ae.encoder.__call__ is not None
    policy = PolicyClassifier(8, 9)
    assert policy.net[-1].out_features == 9


def test_domain_randomization_keeps_environment_valid():
    base = EnergyProductionConfig()
    sample = domain_randomization_sampler(base)(np.random.default_rng(2))
    sample.validate()
    assert 5.0 <= sample.base_demand <= 7.5


def test_model_and_transfer_defaults_are_small_positive_objects():
    assert DynamicsTrainingConfig().ensemble_size >= 2
    assert CEMConfig().horizon >= 1
    agent = ProductionDQN(EnergyProductionConfig(), TransferDQNConfig(episodes=1, warmup=1, batch_size=1))
    assert len(agent.actions) == 9
    model = LearnedDynamicsEnsemble(5, DynamicsTrainingConfig(ensemble_size=2, epochs=1, transitions=10))
    assert len(model.members) == 2
