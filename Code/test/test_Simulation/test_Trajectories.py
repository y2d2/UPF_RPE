import unittest

from Code.Simulation.Trajectories import Trajectory, IMU
import numpy as np
import matplotlib.pyplot as plt
import Code.UtilityCode.SE23 as SE23
import matplotlib.pyplot as plt

class Trajectory_TestCase(unittest.TestCase):
    def test_trajectory(self):
        X_G0 = SE23.SE23_from_w_v_t(np.zeros(3), np.zeros(3), np.array([0,0,0]))

    def analytical_schuine_worp(self, angle, v0, g, dt):
        t_flight = (2 * v0 * np.sin(angle)) / g
        x_range = v0 * np.cos(angle) * t_flight
        v_final = np.array([v0 * np.cos(angle), -v0 * np.sin(angle), 0])

        ts = [t for t in np.arange(0, t_flight, dt)]
        xs = [v0 * np.cos(angle) * t for t in ts]
        # vys = [v0 * np.sin(angle) - g.t for t in ts ]
        ys = [v0 * np.sin(angle) * t - 0.5 * g * t ** 2 for t in ts]
        return t_flight, x_range, v_final, ts, xs, ys

    def test_schuine_worp(self):
        #analystical solution for range of projectile motion:
        angle = np.pi/4
        g = 9.81
        v0 = 10
        dt = 0.01

        t_flight, x_range, v_final, ts, xs, ys = self.analytical_schuine_worp(angle, v0, g,dt)
        # numerical solution using trajectory class
        T_OR = np.eye(4)
        R = SE23.get_SO3_rotation_matrix(np.array([0,0,angle]))
        T_OR[:3,:3] = R
        V_RR0 = np.array([v0, 0 ,0])
        traj = Trajectory(T_OR, V_RR0, time=0)
        x_nums = []
        y_nums = []
        for t in np.arange(0, t_flight, dt):
            a = np.array([-np.cos(angle)*g,-np.sin(angle)*g, 0])
            w = np.array([0,0,0])
            traj.do_step(a, w, t+dt)
            X_OR = traj.X_OR @ traj.X_RRi[-1]
            x_nums.append(X_OR[0,4])
            y_nums.append(X_OR[1,4])

        plt.plot(x_nums, y_nums)
        plt.plot(xs, ys)
        plt.show()

    def test_schuine_worp_imu(self):
        angle = np.pi/4
        g = 9.81
        v0 = 10
        dt = 0.01

        t_flight, x_range, v_final, ts, x_an, y_an = self.analytical_schuine_worp(angle, v0, g,dt)

        # numerical solution using trajectory class
        T_OR = np.eye(4)
        V_RR0 = np.array([np.cos(angle)*v0, np.sin(angle)*v0 ,0])

        imu = IMU(T_OR, V_RR0, time=0,sig_w = 0.001, sig_a = 0.0001, sig_ba = 0.0001, sig_bw = 0.0001)
        imu.b_acc = [np.array([0.0, 0.0, 0.0])]
        imu.b_gyro = [np.array([0.0, 0.0, 0.0])]

        x_true = []
        y_true = []
        x_imu = []
        y_imu = []

        for t in np.arange(0, t_flight, dt):
            a = np.array([0,-g, 0])
            w = np.array([0,0,0])
            imu.get_new_measurement(a, w, t)
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

if __name__ == '__main__':
    # unittest.main()
    # t = Trajectory_TestCase()
    X_OR0 = SE23.SE23_from_w_v_t(np.array([0,0,np.pi/2]), np.array([1,0,0]), np.array([0,0,0]))
    print(X_OR0)
