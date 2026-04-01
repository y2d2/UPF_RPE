import unittest
import os

from Code.ParticleFilter.TargetTrackingParticle import NLSLOSTargetTrackingParticle

os.environ["OPENBLAS_NUM_THREADS"]= "2"

import matplotlib.pyplot as plt
import numpy as np
from Code.BaseLines.NLS import NLS
from Code.DataLoggers.NLS_DataLogger import NLSDataLogger
from Code.Simulation.BiRobotMovement import run_simulation, drone_flight, random_movements_host_random_movements_connected
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix, inv_transformation_matrix, transform_matrix, \
    get_states_of_transform
from Code.DataLoggers.TargetTrackingParticle_DataLogger import NLSTargetTrackingParticle_DataLogger


class MyTestCase(unittest.TestCase):
    def init_test(self, sigma_dv=0.1, sigma_dw=0.1, sigma_uwb=0.1):
        self.uwb_time_steps = 300  # (120 // 0.03)          # Paper simulation time = 120s
        self.odom_time_step = 0.1
        self.uwb_time_step =1. # Paper experiments UWB.py frequency = 37 Hz
        self.factor = int(self.uwb_time_step / self.odom_time_step)
        self.simulation_time_steps = self.uwb_time_steps * self.factor

        self.sigma_uwb = sigma_uwb  # Paper sigma uwb = 0.1
        self.sigma_dv = sigma_dv  # Paper sigma odom = 0.001 m -> not sure how this relates with the heading error.
        self.sigma_dw = sigma_dw

        self.los = []

    def init_drones(self, x_ca_0, h_ca_0, max_range=None):
        self.max_range = max_range
        ha_pose_0 = np.array([0, 0, 0, 0])
        ca_pose_0 = np.concatenate([x_ca_0, np.array([h_ca_0])])
        self.drone = drone_flight(ca_pose_0, sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw, max_range=self.max_range,
                                  origin_bool=True, simulation_time_step=self.odom_time_step)
        self.host = drone_flight(ha_pose_0, sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw, max_range=self.max_range,
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
                dx = np.vstack([dx_ha.reshape(1, *dx_ha.shape), dx_ca.reshape(1, *dx_ca.shape)])
                q_odom = np.vstack([q_ha.reshape(1, *q_ha.shape), q_ca.reshape(1, *q_ca.shape)])
                d_ij = np.linalg.norm(self.drone.x_real[j] - self.host.x_real[j]) + np.random.randn(1)[0] * self.sigma_uwb
                # if j > 100:
                #     d = np.array([[0, d+2], [0, 0]])
                # else:
                #     d = np.array([[0, d], [0, 0]])
                d = np.array([[0, d_ij], [0, 0]])
                # self.NLS.update(d, dx, q_odom)
                self.NLS_particle.run_model(dt_i=dx_ha, q_i=q_ha, t_i=np.zeros(4), P_i=np.zeros((4, 4)),
                                            dt_j=dx_ca, q_j=q_ca, d_ij=d_ij, sig_uwb=self.sigma_uwb, time_i=j)
                # self.NLS.calculate_mesurement_error(self.NLS.x_origin)

                # self.alg_solver.get_update(d=d, dx_ha=dx_ha, dx_ca=dx_ca, q_ha=q_ha, q_ca=q_ca)

                # self.nls_logger.log_data(j)
                self.nls_particle_logger.log_data(j)

                dx_ca = np.zeros(4)
                q_ca = np.zeros((4, 4))
                dx_ha = np.zeros(4)
                q_ha = np.zeros((4, 4))

        self.nls_particle_logger.plot_self()

    # ---- TEST CASES ----
    def test_move_randomly_los(self):
        self.init_test(sigma_dv=1.e-2, sigma_dw=1e-2, sigma_uwb=1e-1)
        self.init_drones(np.array([5, 5, 0]), np.pi / 4, max_range=20)
        run_simulation(self.simulation_time_steps, self.host, self.drone,
                       random_movements_host_random_movements_connected)

        agents = {"drone_0": self.host, "drone_1": self.drone}
        self.NLS = NLS(agents, 10, self.sigma_uwb)
        self.NLS_particle = NLSLOSTargetTrackingParticle(self.NLS)

        self.nls_particle_logger = NLSTargetTrackingParticle_DataLogger(self.host, self.drone, self.NLS_particle)

        self.run_test(name="Move Randomly LOS")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.NLS.agents_list[0].set_plotting_settings(color="tab:blue")
        self.NLS.agents_list[0].plot_real_position(ax)
        self.NLS.agents_list[1].set_plotting_settings(color="tab:orange")
        self.NLS.agents_list[1].plot_real_position(ax)
        self.nls_particle_logger.rpea_datalogger.plot_corrected_estimated_trajectory(ax, color="tab:blue", linestyle="--")
        self.nls_particle_logger.rpea_datalogger.plot_corrected_estimated_trajectory(ax, agent=1, color="tab:orange", linestyle="--")

        plt.show()

if __name__ == '__main__':
    unittest.main()
