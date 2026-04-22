import unittest

import numpy as np

try:
    from Code.Exp_data.Sensors import ESP_IMU, EspPacketParser, IMU, InterRobotDistanceSensor, MeasuredTrajectory, VIO
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

    def test_esp_packet_parser_parses_multi_sample_packet(self):
        packet = (
            "5;7155308;714825;0;7155474;2;"
            "7155464;38.72;-45.65;177.37;1003.54;0.46;0.59;0.00;2.25;-23.25;34.05;1.00;0.00;0.00;0.00;"
            "7155474;38.73;-45.66;177.38;1003.55;0.47;0.58;0.01;2.26;-23.24;34.06;0.99;0.01;0.00;0.00;"
            "1;4;4.72;-85;-78;"
        )
        parsed = EspPacketParser.parse_packet_string(packet)

        self.assertEqual(parsed.packet_id, 5)
        self.assertEqual(parsed.imu_point_count, 2)
        self.assertEqual(len(parsed.imu_samples), 2)
        self.assertEqual(parsed.valid_ranges, 1)
        self.assertEqual(parsed.ranges[0].rid, 4)
        self.assertAlmostEqual(parsed.ranges[0].dist_m, 4.72)

    def test_esp_imu_ros_times_are_packet_anchored(self):
        imu = ESP_IMU(
            t=[10.99, 11.0],
            raw_time_ms=[10990, 11000],
            a_body=[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            w_body=[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
            m_body=[[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]],
            temperature=[20.0, 21.0],
            q=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        )

        a, v, w = imu.get_new_measurement(10.995)
        self.assertIsNone(v)
        self.assertTrue(np.allclose(a, np.array([2.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(w, np.array([0.0, 0.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
