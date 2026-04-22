import unittest
from pathlib import Path

import numpy as np

try:
    from Code.Exp_data.Sensors import ESP_IMU, EspPacketParser, IMU, InterRobotDistanceSensor, MeasuredTrajectory, VIO
    import Code.UtilityCode.SE23 as SE23
    HAS_EXP_DATA_DEPS = True
except ImportError:
    HAS_EXP_DATA_DEPS = False

try:
    import rosbags  # noqa: F401
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False


TB2_BAG = Path("/workspace/Ptyhon/https-github.com-y2d2-PRP_ARP_M/test_cases/exp_data/tb2_exp_2026_04_21-11_21_04")
IMU_BAG = Path("/workspace/Ptyhon/https-github.com-y2d2-PRP_ARP_M/test_cases/exp_data/imu_esp_uwb_2026_04_16-07_28_17")


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


@unittest.skipUnless(HAS_EXP_DATA_DEPS, "This test requires the UPF_RPE optional math dependencies.")
@unittest.skipUnless(HAS_ROSBAGS, "This test requires the 'rosbags' package.")
class TestMeasuredSensorPlots(unittest.TestCase):
    def test_plot_vio_trajectory_from_bag(self):
        import matplotlib.pyplot as plt

        vio = VIO.from_rosbag(TB2_BAG, odom_topic="/tb2/odom")

        figure, axis = plt.subplots()
        vio.odom_trajectory.plot_trajectory(axis, label="TB2 VIO", color="tab:blue")
        axis.set_title("TB2 VIO trajectory")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True)
        axis.axis("equal")
        axis.legend()
        plt.show()

    def test_plot_imu_trajectory_from_bag(self):
        import matplotlib.pyplot as plt

        imu = IMU.from_rosbag(IMU_BAG, imu_topic="/a200_0957/sensors/imu_0/data")

        figure, axis = plt.subplots()
        imu.odom_trajectory.plot_trajectory(axis, label="Jazzy IMU", color="tab:orange")
        axis.set_title("Jazzy IMU integrated trajectory")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True)
        axis.axis("equal")
        axis.legend()
        plt.show()

    def test_plot_esp_imu_trajectory_from_bag(self):
        import matplotlib.pyplot as plt

        esp_imu = ESP_IMU.from_rosbag(TB2_BAG, packet_topic="/tb2/uwb")

        figure, axis = plt.subplots()
        esp_imu.plot_trajectory(axis, label="TB2 ESP IMU", color="tab:green")
        axis.set_title("TB2 ESP IMU integrated trajectory")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True)
        axis.axis("equal")
        axis.legend()
        plt.show()

    def test_plot_interrobot_distance_from_bag(self):
        import matplotlib.pyplot as plt

        uwb = InterRobotDistanceSensor.from_rosbag(TB2_BAG, uwb_topic="/tb2/uwb")

        figure, axis = plt.subplots()
        uwb.plot(axis, label="TB2 measured UWB distance")
        axis.set_title("TB2 UWB distance")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("distance [m]")
        axis.grid(True)
        axis.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
