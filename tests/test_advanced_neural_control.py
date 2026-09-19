import unittest

import numpy as np

from industrial_rl.latent_state_control import AutoEncoder, ObservationMap, PolicyClassifier
from industrial_rl.model_based_control import CEMConfig, DynamicsTrainingConfig, LearnedDynamicsEnsemble
from industrial_rl.sac_energy import EnergyProductionConfig
from industrial_rl.transfer_control import ProductionDQN, TransferDQNConfig, domain_randomization_sampler


class AdvancedNeuralControlTests(unittest.TestCase):
    def test_observation_map_and_autoencoder_shapes(self):
        mapper = ObservationMap(5, 64, seed=1)
        high = mapper.transform(np.zeros(5, dtype=np.float32))
        self.assertEqual(high.shape, (64,))
        ae = AutoEncoder(64, 8)
        self.assertTrue(callable(ae.encoder))
        policy = PolicyClassifier(8, 9)
        self.assertEqual(policy.net[-1].out_features, 9)

    def test_domain_randomization_keeps_environment_valid(self):
        base = EnergyProductionConfig()
        sample = domain_randomization_sampler(base)(np.random.default_rng(2))
        sample.validate()
        self.assertLessEqual(5.0, sample.base_demand)
        self.assertLessEqual(sample.base_demand, 7.5)

    def test_model_and_transfer_defaults(self):
        self.assertGreaterEqual(DynamicsTrainingConfig().ensemble_size, 2)
        self.assertGreaterEqual(CEMConfig().horizon, 1)
        agent = ProductionDQN(
            EnergyProductionConfig(),
            TransferDQNConfig(episodes=1, warmup=1, batch_size=1),
        )
        self.assertEqual(len(agent.actions), 9)
        model = LearnedDynamicsEnsemble(
            5,
            DynamicsTrainingConfig(ensemble_size=2, epochs=1, transitions=10),
        )
        self.assertEqual(len(model.members), 2)


if __name__ == "__main__":
    unittest.main()
