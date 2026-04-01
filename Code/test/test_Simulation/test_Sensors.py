import unittest
from Code.Simulation.Trajectories import Trajectory
from Code.Simulation.Sensors import IMU, VIO_2D, IMU_2D, InterRobotDistanceSensor
from Code.test.test_Simulation.test_Trajectories import analytical_schuine_worp
from Code.Simulation.Trajectories_Generator import create_ERB, create_ECB, Velocity_Control_2D, trajectory_generator, create_still_trajectory
import numpy as np
import matplotlib.pyplot as plt
import Code.UtilityCode.SE23 as SE23
import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_schuine_worp_imu(self):
        angle = np.pi/4
        g = 9.81
        v0 = 10
        dt = 0.01

        t_flight, x_range, v_final, ts, x_an, y_an = analytical_schuine_worp(angle, v0, g,dt)

        # numerical solution using trajectory class
        T_OR = np.eye(4)
        V_RR0 = np.array([np.cos(angle)*v0, np.sin(angle)*v0 ,0])

        imu = IMU(T_OR, V_RR0, time=0,sig_w = 0.001, sig_a = 0.0001, sig_ba = 0.0001, sig_bw = 0.0001)
        # imu.b_acc = [np.array([0.0, 0.0, 0.0])]
        # imu.b_gyro = [np.array([0.0, 0.0, 0.0])]

        x_true = []
        y_true = []
        x_imu = []
        y_imu = []

        for t in np.arange(0, t_flight, dt):
            a = np.array([0,-g, 0])
            w = np.array([0,0,0])
            imu.get_new_measurement(a=a, v=None, w= w, time=t)
            x_true.append(imu.true_trajectory.X_RRi[-1][0,4])
            y_true.append(imu.true_trajectory.X_RRi[-1][1,4])
            x_imu.append(imu.odom_trajectory.X_RRi[-1][0,4])
            y_imu.append(imu.odom_trajectory.X_RRi[-1][1,4])

        plt.plot(x_true, y_true, label="True trajectory")
        plt.plot(x_imu, y_imu, label="IMU trajectory")
        plt.plot(x_an, y_an, ":", label="Analytical trajectory")
        plt.legend()

        plt.figure()
        plt.plot(np.array(imu.b_acc))
        plt.figure()
        plt.plot(np.array(imu.b_gyro))

        plt.show()

    def test_2D_control_room(self):
        T_OR = np.eye(4)
        v_R0 = np.array([0,0,0])
        vio_sen = VIO_2D(T_OR, v_R0, time=0, sig_w=0.1, sig_v=0.1)
        imu_sen = IMU_2D(T_OR, v_R0, time=0, sig_w=0.1, sig_a=0.1, sig_ba=0.01, sig_bw=0.01)

        # calibrated IMU
        imu_sen.b_acc = [np.zeros(3)]
        imu_sen.b_gyro = [np.zeros(3)]

        sensor_dict = {"VIO": {"sensor": vio_sen, "color": "red", "linestyle": "--", "label": "VIO"},
                       "IMU": {"sensor": imu_sen, "color": "blue", "linestyle": "--", "label": "IMU"}}
        traj = trajectory_generator(dict_of_sensors=sensor_dict, plt_bool=True)

    def test_interrobot_distance_sensor(self):
        ax = plt.figure().add_subplot(111, projection='3d')
        traj_1 = trajectory_generator( T_OR=np.eye(4), v_R0=np.array([0,0,0]), dict_of_sensors={})
        traj_2 = trajectory_generator( T_OR=SE23.SE3_from_rot_vec_and_trans(np.zeros(3), np.array([5,0,0])), v_R0=np.array([0,0,0]), dict_of_sensors={})
        traj_2 = create_still_trajectory( T_OR=SE23.SE3_from_rot_vec_and_trans(np.zeros(3), np.array([5,0,0])))
        uwb = InterRobotDistanceSensor(traj_1, traj_2, sig_d=0.1)
        for t1 in traj_1.t:
            uwb.get_new_measurement(t1)


        ax = plt.figure().add_subplot(111, projection='3d')
        traj_1.plot_trajectory(ax, label="Robot 1", color="blue")
        traj_2.plot_trajectory(ax, label="Robot 2")



        plt.figure()
        plt.plot(traj_1.t, uwb.d)
        plt.plot(traj_1.t, uwb.d_true, label="True distance")
        plt.show()

if __name__ == '__main__':
    unittest.main()
