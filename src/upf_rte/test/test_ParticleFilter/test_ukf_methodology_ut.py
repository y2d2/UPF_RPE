import unittest

import numpy as np

from upf_rte.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent
from upf_rte.ParticleFilter.TargetTrackingUKF import TargetTrackingUKF, subtract


class FakeParticle:
    def __init__(self, weight):
        self.weight = weight
        self.rpea = None


class TestUKFMethodology(unittest.TestCase):
    def create_ukf(self, sigma_state=None):
        if sigma_state is None:
            sigma_state = np.array([0.1, 1e-6, 1e-6, 1e-6])
        ukf = TargetTrackingUKF(x_ha_0=np.zeros(4), drift_correction_bool=True)
        ukf.set_ukf_properties()
        ukf.set_initial_state(np.array([1.0, 0.0, 0.0, 0.0]), sigma_state)
        return ukf

    def test_calculate_r_projects_host_covariance_onto_range(self):
        ukf = self.create_ukf()

        ukf.calculate_r(0.1, np.diag([0.25, 0.0, 0.0, 0.0]))
        range_aligned_variance = ukf.kf.R[0, 0]

        ukf.calculate_r(0.1, np.diag([0.0, 0.25, 0.0, 0.0]))
        range_perpendicular_variance = ukf.kf.R[0, 0]

        self.assertAlmostEqual(range_aligned_variance, 0.26, places=6)
        self.assertAlmostEqual(range_perpendicular_variance, 0.01, places=6)

    def test_calculate_r_rejects_bad_host_covariance_shape(self):
        ukf = self.create_ukf()

        with self.assertRaisesRegex(ValueError, "P_x_ha must be a 4x4"):
            ukf.calculate_r(0.1, np.eye(3))

    def test_relative_pose_covariance_uses_state_sigma_points(self):
        tight_ukf = self.create_ukf(np.array([0.1, 1e-6, 1e-6, 1e-6]))
        loose_ukf = self.create_ukf(np.array([0.1, 0.2, 0.2, 1e-6]))

        self.assertTrue(np.allclose(tight_ukf.P_t_si_sj, tight_ukf.P_t_si_sj.T))
        self.assertTrue(np.allclose(loose_ukf.P_t_si_sj, loose_ukf.P_t_si_sj.T))
        self.assertGreater(loose_ukf.P_t_si_sj[1, 1], tight_ukf.P_t_si_sj[1, 1])
        self.assertGreater(loose_ukf.P_t_si_sj[2, 2], tight_ukf.P_t_si_sj[2, 2])

    def test_angle_residual_wraps_state_angles(self):
        x = np.zeros(9)
        y = np.zeros(9)
        x[1] = -np.pi + 0.05
        y[1] = np.pi - 0.05
        x[3] = np.pi - 0.02
        y[3] = -np.pi + 0.02

        residual = subtract(x, y)

        self.assertAlmostEqual(residual[1], 0.1, places=6)
        self.assertAlmostEqual(residual[3], -0.04, places=6)

    def test_predict_rejects_legacy_displacement_shapes(self):
        ukf = self.create_ukf()

        with self.assertRaisesRegex(ValueError, "dx_ca must be a 4D"):
            ukf.predict(np.zeros(3), np.zeros((4, 4)))

        with self.assertRaisesRegex(ValueError, "q must be a 4x4"):
            ukf.predict(np.zeros(4), 0.0)

    def test_pruning_preserves_normalized_posterior_weights(self):
        agent = UPFConnectedAgent([FakeParticle(0.8), FakeParticle(0.2)], resample_factor=0.01)

        agent.pruning_resampling()

        self.assertEqual(len(agent.particles), 2)
        np.testing.assert_allclose(agent.weights, [0.8, 0.2])
        self.assertAlmostEqual(sum(agent.weights), 1.0)
        self.assertIs(agent.best_particle, agent.particles[0])


if __name__ == "__main__":
    unittest.main()
