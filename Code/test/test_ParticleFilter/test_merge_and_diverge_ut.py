import unittest

import numpy as np

from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent


class FakeParticle:
    def __init__(self, weight, t_si_sj, covariance=None, los_state=1):
        self.weight = weight
        self.t_si_sj = np.array(t_si_sj, dtype=float)
        self.P_t_si_sj = np.eye(4) if covariance is None else np.array(covariance, dtype=float)
        self.los_state = los_state
        self.rpea = None


class TestMergeAndDiverge(unittest.TestCase):
    def create_agent(self, particles):
        agent = UPFConnectedAgent(list(particles))
        agent.weights = [particle.weight for particle in particles]
        return agent

    def test_merge_keeps_highest_weight_particle_object(self):
        low_weight_particle = FakeParticle(
            weight=0.2,
            t_si_sj=np.array([1.0, 2.0, 3.0, 0.1]),
            covariance=np.eye(4) * 0.1,
        )
        high_weight_particle = FakeParticle(
            weight=0.8,
            t_si_sj=np.array([1.1, 2.0, 3.0, 0.1]),
            covariance=np.eye(4) * 0.1,
        )
        agent = self.create_agent([low_weight_particle, high_weight_particle])

        agent.merge_similar_particles()

        self.assertEqual(len(agent.particles), 1)
        self.assertIs(agent.particles[0], high_weight_particle)
        self.assertIs(agent.best_particle, high_weight_particle)
        self.assertAlmostEqual(high_weight_particle.weight, 1.0)
        np.testing.assert_allclose(agent.weights, [1.0])

    def test_similar_particle_states_are_merged_by_weighted_moments(self):
        particle_1 = FakeParticle(
            weight=0.75,
            t_si_sj=np.array([0.0, 0.0, 0.0, np.pi - 0.1]),
            covariance=np.eye(4) * 0.1,
        )
        particle_2 = FakeParticle(
            weight=0.25,
            t_si_sj=np.array([0.4, 0.0, 0.0, -np.pi + 0.1]),
            covariance=np.eye(4) * 0.1,
        )
        agent = self.create_agent([particle_1, particle_2])

        agent.merge_similar_particles()

        self.assertEqual(len(agent.particles), 1)
        self.assertIs(agent.particles[0], particle_1)
        self.assertAlmostEqual(particle_1.weight, 1.0)
        np.testing.assert_allclose(particle_1.t_si_sj[:3], np.array([0.1, 0.0, 0.0]))
        self.assertTrue(np.isclose(abs(particle_1.t_si_sj[3]), np.pi, atol=0.11))
        self.assertTrue(np.allclose(particle_1.P_t_si_sj, particle_1.P_t_si_sj.T))

    def test_dissimilar_particles_are_not_merged(self):
        particle_1 = FakeParticle(weight=0.7, t_si_sj=np.array([0.0, 0.0, 0.0, 0.0]))
        particle_2 = FakeParticle(weight=0.3, t_si_sj=np.array([10.0, 0.0, 0.0, 0.0]))
        agent = self.create_agent([particle_1, particle_2])

        agent.merge_similar_particles()

        self.assertEqual(agent.particles, [particle_1, particle_2])
        np.testing.assert_allclose(agent.weights, [0.7, 0.3])
        self.assertIs(agent.best_particle, particle_1)

    def test_particles_with_different_los_state_are_not_merged(self):
        particle_1 = FakeParticle(weight=0.4, t_si_sj=np.zeros(4), los_state=1)
        particle_2 = FakeParticle(weight=0.6, t_si_sj=np.zeros(4), los_state=0)
        agent = self.create_agent([particle_1, particle_2])

        agent.merge_similar_particles()

        self.assertEqual(agent.particles, [particle_2, particle_1])
        np.testing.assert_allclose(agent.weights, [0.6, 0.4])
        self.assertIs(agent.best_particle, particle_2)


if __name__ == "__main__":
    unittest.main()
