import unittest

import matplotlib.pyplot as plt
import numpy as np
from Code.BaseLines.QCQP import QCQP
from Code.DataLoggers.QCQP_DataLogger import QCQP_Log
from Code.Simulation.BiRobotMovement import run_simulation, drone_flight, random_movements_host_random_movements_connected
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix

class MyTestCase(unittest.TestCase):
    def init_test(self, sigma_dv=0.1, sigma_dw=0.1, sigma_uwb=0.1):
        self.uwb_time_steps = 200  # (120 // 0.03)          # Paper simulation time = 120s
        self.odom_time_step = 0.1
        self.uwb_time_step = 0.1 # Paper experiments UWB.py frequency = 37 Hz
        self.factor = int(self.uwb_time_step / self.odom_time_step)
        self.simulation_time_steps = int(self.uwb_time_steps * self.factor)

        self.sigma_uwb = sigma_uwb  # Paper sigma uwb = 0.1
        self.sigma_dv = sigma_dv  # Paper sigma odom = 0.001 m -> not sure how this relates with the heading error.
        self.sigma_dw = sigma_dw

        self.los = []

    def init_drones(self, x_ca_0, h_ca_0, max_range=None):
        self.max_range = max_range
        ha_pose_0 = np.array([0, 0, 0, 0])
        ca_pose_0 = np.concatenate([x_ca_0, np.array([h_ca_0])])
        self.drone = drone_flight(ca_pose_0, start_velocity=np.array([1.,0,0,0]), slowrate_v=0.5, slowrate_w=0.01,
                                  sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw, max_range=self.max_range,
                                  origin_bool=True, simulation_time_step=self.odom_time_step)
        self.host = drone_flight(ha_pose_0, start_velocity=np.array([1.,0,0,0]), slowrate_v=0.5, slowrate_w=0.01,
                                 sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw, max_range=self.max_range,
                                 origin_bool=True, simulation_time_step=self.odom_time_step)

        distance = np.linalg.norm(self.drone.x_start - self.host.x_start)
        self.startMeasurement = distance + np.random.randn(1) * self.sigma_uwb

    def run_test(self, name="Unidentified Test", nlos_function=None):
        dx_ca = np.zeros(4)
        q_ca = np.zeros((4, 4))
        dx_ha = np.zeros(4)
        q_ha = np.zeros((4, 4))
        for j in range(1, self.simulation_time_steps + 1):
            print("Time step: ", j)
            c_ca = get_4d_rot_matrix(dx_ca[-1])
            dx_ca = dx_ca + c_ca @ np.concatenate((self.drone.dx_slam[j], np.array([self.drone.dh_slam[j]])))
            q_ca = q_ca + c_ca @ self.drone.q @ c_ca.T

            c_ha = get_4d_rot_matrix(dx_ha[-1])
            dx_ha = dx_ha + c_ha @ np.concatenate((self.host.dx_slam[j], np.array([self.host.dh_slam[j]])))
            q_ha = q_ha + c_ha @ self.host.q @ c_ha.T

            if j % self.factor == 0:
                dx_i = dx_ha
                dx_j = dx_ca
                d = np.linalg.norm(self.drone.x_real[j] - self.host.x_real[j]) + np.random.randn(1)[0] * self.sigma_uwb
                # print(dx_i, dx_j, d)
                self.qcqp.update(dx_i, dx_j, d)

                self.qcqp_log.log(j)

                dx_ca = np.zeros(4)
                q_ca = np.zeros((4, 4))
                dx_ha = np.zeros(4)
                q_ha = np.zeros((4, 4))


    # ---- TEST CASES ----
    def test_move_randomly_los(self):
        self.init_test(sigma_dv=0.1, sigma_dw=0.01, sigma_uwb=0.1
                       )
        self.init_drones(np.array([5, 5, 0]), np.pi / 4, max_range=20)
        run_simulation(self.simulation_time_steps, self.host, self.drone,
                       random_movements_host_random_movements_connected)

        self.qcqp = QCQP(horizon=100, sigma_uwb=self.sigma_uwb)
        self.qcqp_log = QCQP_Log(self.qcqp, self.host, self.drone)

        self.run_test(name="Move Randomly LOS")



        self.qcqp_log.plot_self()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.host.plot_trajectory(ax= ax, color="green")
        self.drone.plot_trajectory(ax= ax, color="k")
        self.drone.plot_slam_position(ax=ax, color="k", linestyle="--")
        self.host.plot_slam_position(ax=ax, color="green", linestyle="--")
        self.qcqp_log.plot_corrected_estimated_trajectory(ax=ax, color="r", linestyle="", marker="o", label="QCQP")

        plt.show()


if __name__ == '__main__':
    unittest.main()

