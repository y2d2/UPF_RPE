import unittest

import numpy as np

from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent
from Code.ParticleFilter.TargetTrackingParticle import UKFLOSTargetTrackingParticle


class TestConnectedAgentInitialization(unittest.TestCase):
    def test_split_sphere_creates_wrapped_particles_that_run(self):
        agent = UPFConnectedAgent([], x_ha_0=np.zeros(4), sigma_uwb=0.1)
        agent.split_sphere_in_equal_areas(
            r=1.0,
            sigma_uwb=0.1,
            n_altitude=1,
            n_azimuth=1,
            n_heading=1,
        )

        self.assertEqual(len(agent.particles), 1)
        self.assertIsInstance(agent.particles[0], UKFLOSTargetTrackingParticle)

        agent.run_model(
            dt_j=np.zeros(4),
            q_j=np.eye(4) * 1e-6,
            dt_i=np.zeros(4),
            q_i=np.zeros((4, 4)),
            d_ij=1.0,
        )

        self.assertEqual(len(agent.particles), 1)
        self.assertIs(agent.best_particle, agent.particles[0])
        self.assertEqual(len(agent.weights), len(agent.particles))

    def test_create_single_particle_creates_initialized_wrapper(self):
        agent = UPFConnectedAgent([], x_ha_0=np.zeros(4))

        agent.create_single_particle(np.array([1.0, 0.0, 0.0, 0.0]), sigma_uwb=0.1)

        self.assertEqual(len(agent.particles), 1)
        particle = agent.particles[0]
        self.assertIsInstance(particle, UKFLOSTargetTrackingParticle)
        np.testing.assert_allclose(particle.t_si_sj, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
        self.assertTrue(np.all(np.isfinite(np.diag(particle.P_t_si_sj))))

    def test_empty_particles_raise_clear_error(self):
        agent = UPFConnectedAgent([])

        with self.assertRaisesRegex(ValueError, "No particles left"):
            agent.run_model(
                dt_j=np.zeros(4),
                q_j=np.eye(4),
                dt_i=np.zeros(4),
                q_i=np.eye(4),
                d_ij=1.0,
            )

    def test_zero_weights_raise_clear_error(self):
        agent = UPFConnectedAgent([])
        agent.create_single_particle(np.array([1.0, 0.0, 0.0, 0.0]), sigma_uwb=0.1)
        agent.particles[0].weight = 0.0

        with self.assertRaisesRegex(ValueError, "positive finite sum"):
            agent.pruning_resampling()


if __name__ == "__main__":
    unittest.main()
