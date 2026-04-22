import unittest

import numpy as np

try:
    from Code.Exp_data.Sensors import IMU, InterRobotDistanceSensor, MeasuredTrajectory, VIO
    import Code.UtilityCode.SE23 as SE23
    HAS_EXP_DATA_DEPS = True
except ImportError:
    HAS_EXP_DATA_DEPS = False


@unittest.skipUnless(HAS_EXP_DATA_DEPS, "This test requires the UPF_RPE optional math dependencies.")
class TestMeasuredSensors(unittest.TestCase):
    def test_measured_trajectory_interpolates_state(self):
        T0 = np.eye(4)
        T1 = SE23.SE3_from_rot_vec_and_trans(np.array([0.0, 0.0, 0.2]), np.array([1.0, 0.0, 0.0]))
        trajectory = MeasuredTrajectory(
            T_global=[T0, T1],
            time=[0.0, 1.0],
            v_body=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            w_body=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        )

        position, velocity, rotation, _ = trajectory.get_local_state_at_time(0.5)
        self.assertTrue(np.allclose(position, np.array([0.5, 0.0, 0.0])))
        self.assertTrue(np.allclose(velocity, np.array([0.5, 0.0, 0.0])))
        self.assertAlmostEqual(rotation[2], 0.1, places=6)

    def test_vio_sensor_returns_velocity_measurement(self):
        T0 = np.eye(4)
        T1 = SE23.SE3_from_rot_vec_and_trans(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        trajectory = MeasuredTrajectory(
            T_global=[T0, T1],
            time=[0.0, 1.0],
            v_body=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            w_body=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
        )
        sensor = VIO(trajectory=trajectory)

        a, v, w = sensor.get_new_measurement(time=0.5)
        self.assertIsNone(a)
        self.assertTrue(np.allclose(v, np.array([0.5, 0.0, 0.0])))
        self.assertTrue(np.allclose(w, np.array([0.0, 0.0, 0.05])))

    def test_imu_sensor_returns_acceleration_measurement(self):
        T0 = np.eye(4)
        T1 = SE23.SE3_from_rot_vec_and_trans(np.zeros(3), np.array([0.5, 0.0, 0.0]))
        trajectory = MeasuredTrajectory(
            T_global=[T0, T1],
            time=[0.0, 1.0],
            v_body=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            w_body=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
            a_body=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        sensor = IMU(trajectory=trajectory)

        a, v, w = sensor.get_new_measurement(time=0.5)
        self.assertTrue(np.allclose(a, np.array([1.0, 0.0, 0.0])))
        self.assertIsNone(v)
        self.assertTrue(np.allclose(w, np.array([0.0, 0.0, 0.1])))

    def test_range_sensor_interpolates_distances(self):
        sensor = InterRobotDistanceSensor(t=[0.0, 1.0], d=[1.0, 3.0])
        self.assertAlmostEqual(sensor.get_new_measurement(0.5), 2.0)


if __name__ == "__main__":
    unittest.main()
