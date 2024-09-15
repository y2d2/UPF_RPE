1  # !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:35:30 2023

@author: yuri
"""
import os
import traceback
from datetime import datetime
import unittest
import numpy as np
import matplotlib.pyplot as plt

# Have to use this since Spyder_WS is a project.
from Code.UtilityCode.utility_fuctions import cartesianToSpherical, limit_angle, get_4d_rot_matrix

from Code.ParticleFilter.TargetTrackingUKF import TargetTrackingUKF
from Code.DataLoggers.TargetTrackingUKF_DataLogger import load_dataFile, UKFDatalogger
from Code.Simulation.BiRobotMovement import moving_gps_tracked_host_fix_connected, \
    moving_sinusoidal_gps_tracked_host_random_connected, random_movements_host_random_movements_connected, \
    fix_host_jumping_y_connected, fix_host_random_movement_connected, \
    drone_flight, run_simulation


class Test_TargetTrackingUKF(unittest.TestCase):
    def initTest(self):
        # Paper = Relative Transformation Estimation Based on Fusion of Odometry and UWB.py Ranging Data

        self.uwb_time_steps = 1000  # (120 // 0.03)          # Paper simulation time = 120s
        self.odom_time_step = 0.1  # Paper experiments UWB.py frequency = 37 Hz
        self.uwb_time_step = 0.1
        self.factor = int( self.uwb_time_step / self.odom_time_step)
        self.simulation_time_steps = self.uwb_time_steps * self.factor

        self.max_range = 20

        self.sigma_uwb = 0.1  # Paper sigma uwb = 0.1
        self.sigma_dv = 0.1  # Paper sigma odom = 0.001 m -> not sure how this relates with the heading error.
        self.sigma_dw = 0.1 #/ 180 * np.pi  # division by 2 since I am not sure what error they took in the paper. I think they assumed the host agent has no drift so no error.

        self.drone = drone_flight(np.array([0,0, 5, np.pi/4]), sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw,
                                  max_range=self.max_range, origin_bool=True, simulation_time_step=self.odom_time_step)
        self.host = drone_flight(np.array([0, 0, 0, 0]), sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw,
                                 max_range=self.max_range, origin_bool=True, simulation_time_step=self.odom_time_step)
        self.init_saves(False)
        self.bool_host_agent_drift = True

    def init_saves(self, save_bool=True):
        if save_bool:
            now = datetime.now()
            self.folder_name = os.path.realpath("/") + "/Results_UKF/"
            self.folder_name += str(now.year) + "_" + str(now.month) + "_" + str(now.day)
            if not os.path.exists(self.folder_name):
                os.mkdir(self.folder_name)
        self.save_bool = save_bool


    # ---- UKF
    def setUKF(self, kappa=-1, alpha: float = 1.0, beta=2,
               n_azimuth=4, n_altitude=3, n_heading_c=4):
        sigma_azimuth = (2 * np.pi / n_azimuth) / np.sqrt(-8 * np.log(0.5))
        sigma_altitude = (np.pi / n_altitude) / np.sqrt(-8 * np.log(0.5))
        ca_sigma_heading = (2 * np.pi / n_heading_c) / np.sqrt(-8 * np.log(0.5))
        sigma_s = [self.sigma_uwb, sigma_azimuth, sigma_altitude, ca_sigma_heading]


        # # Very small start position uncertainty
        # sigma_s = np.ones(3)*1e-4
        # ca_sigma_heading = 1e-4

        x_ha_0 = np.concatenate((self.host.x_start, np.array([self.host.h_start])))
        x_ca_0 = np.concatenate((self.drone.x_start, np.array([self.drone.h_start])))
        x_ca = get_4d_rot_matrix(-x_ha_0[-1]) @ (x_ca_0 - x_ha_0)
        s = cartesianToSpherical(x_ca[:3])
        s = np.array([s[0], s[1], s[2],  x_ca[-1]])
        self.ukf = TargetTrackingUKF(x_ha_0=x_ha_0, drift_correction_bool=self.bool_host_agent_drift)
        self.ukf.set_ukf_properties(kappa, alpha, beta)
        self.ukf.set_initial_state(s, sigma_s)


    def run_test(self, name="Unidentified Testcase", nlos_function= None):
        self.datalogger= UKFDatalogger(self.host, self.drone, self.ukf, name)
        # self.ukf.datalogger.save_graphs(self.folder_name)
        self.ukf.weight = 0.
        # self.ukf.datalogger.log_data(0)


        dx = np.zeros(4)
        q = np.zeros((4, 4))
        q_ha = np.zeros((4,4))
        dx_ha = np.zeros(4)
        x_ha = np.concatenate((self.host.x_start, np.array([self.host.h_start])))
        x_ha_est = x_ha
        self.datalogger.log_data(0)
        for i in range(1,self.simulation_time_steps):
            print("Simulation step: ", i," /",self.simulation_time_steps)
            try:
                f = get_4d_rot_matrix(dx[-1])
                dx = dx + f @ np.concatenate((self.drone.dx_slam[i], np.array([self.drone.dh_slam[i]])))
                q = q + f @ self.drone.q @ f.T

                f_ha = get_4d_rot_matrix(dx_ha[-1])
                dx_ha = dx_ha + f_ha @ np.concatenate((self.host.dx_slam[i], np.array([self.host.dh_slam[i]])))
                # q_ha = q_ha + f_ha @ self.host.q @ f_ha.T
                q_ha = q_ha + self.host.q

                if i % self.factor == 0:
                    x_ha_est = x_ha_est + get_4d_rot_matrix(x_ha_est[-1]) @ dx_ha
                    x_ha = np.concatenate((self.host.x_real[i], [self.host.h_real[i]]))
                    # x_ha = np.concatenate((self.host.x_slam[i], [self.host.h_slam[i]]))
                    x_drift = x_ha - x_ha_est

                    if nlos_function is not None:
                        nlos_function(i)

                    distance = np.linalg.norm(self.host.x_real[i] - self.drone.x_real[i])
                    uwb_measurement = distance + np.random.randn(1)[0] * self.sigma_uwb

                    self.ukf.weight = 1.
                    if self.bool_host_agent_drift:
                        self.ukf.run_filter(dx, q, x_ha_est, q_ha, uwb_measurement, self.sigma_uwb,
                                            bool_drift=self.bool_host_agent_drift)
                    else:
                        self.ukf.run_filter(dx, q, x_ha, q_ha, uwb_measurement, self.sigma_uwb,
                                            bool_drift=self.bool_host_agent_drift)

                    # Reset integration parameters
                    dx = np.zeros(4)
                    q = np.zeros((4, 4))
                    q_ha = np.zeros((4, 4))
                    dx_ha = np.zeros(4)
                    #
                    # self.ukf.predict(dx,q)
                    #
                    #
                    # self.ukf.calculate_r(self.sigma_uwb)
                    #
                    #
                    # self.ukf.update(uwb_measurement, x_ha)

                    self.datalogger.log_data(i)



            except np.linalg.LinAlgError:
                print("Error in ", name)
                print(traceback.format_exc())
                print(self.ukf.kf.P, self.ukf.q_ca)
                break

        self.datalogger.create_3d_plot()
        self.datalogger.plot_error_graph()
        self.datalogger.plot_ukf_states()

    # ---- Test cases : fix host
    def test_TC1_HostAgent_Standing_Still(self):
        TEST_CASE_NAME = "TC1: Fix host agent, Random Connected Agent"
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, fix_host_random_movement_connected)
        self.setUKF()

        self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC1a_JumpingRow_azimuth_pi(self):
        TEST_CASE_NAME = "TC1a: Fix host agent, connected agent jumpingrow on azimuth  = $\pi$"
        self.initTest()
        self.uwb_time_steps = 100
        self.drone.set_start_position(np.array([-10, -1, 0]), 0)
        run_simulation(self.simulation_time_steps, self.host, self.drone, fix_host_jumping_y_connected)
        self.setUKF()

        self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC1b_JumpingRow_altitude_pi_2(self):
        TEST_CASE_NAME = "TC1b: Fix host agent, connected agent jumpingrow on altitude  = $ 0.5 \pi$"
        self.initTest()
        self.uwb_time_steps = 100
        self.drone.set_start_position(np.array([0, -1, 10]), 0)
        run_simulation(self.simulation_time_steps, self.host, self.drone, fix_host_jumping_y_connected)
        self.setUKF()

        self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC1c_HostAgent_Standing_Still_changing_ukf_parameters(self):
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, fix_host_random_movement_connected)
        alphas = [1, 0.1, 1e-3]
        betas = [5]
        kappas = [-1, ]
        for alpha in alphas:
            for beta in betas:
                for kappa in kappas:
                    self.setUKF(kappa=kappa, alpha=alpha, beta=beta)
                    TEST_CASE_NAME = f"TC1c Fix host agent Random Connected Agent alpha ={'c'.join(str(alpha).split('.'))} beta ={beta} kappa = {'c'.join(str(kappa).split('.'))}"
                    self.run_test(TEST_CASE_NAME)
        plt.show()

    # ---- Test case: Moving host, known location, fix agent.
    def test_TC2_MovingHost_KnownMovement_StillAgent(self):
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_gps_tracked_host_fix_connected)

        self.setUKF()
        TEST_CASE_NAME = f"TC2: GPS Tracked Host, fix connected agent. \n alpha = {self.ukf.kf.points_fn.alpha}"
        self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC2a_MovingHost_KnownMovement_StillAgent_ChangingAlpha(self):
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_gps_tracked_host_fix_connected)

        alphas = [1, 0.1, 0.01, 0.001]
        for alpha in alphas:
            self.setUKF(alpha=alpha)
            TEST_CASE_NAME = f"TC2a: GPS Tracked Host, fix connected agent. \n alpha = {alpha} "
            self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC2b_Moving_GPSTracked_Host_StillAgent_WrongEstimate_ChangingAlpha(self):
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_gps_tracked_host_fix_connected)

        # Baseline
        self.setUKF()
        TEST_CASE_NAME = "TC2: GPS Tracked Host, fix connected agent."
        self.run_test(TEST_CASE_NAME)

        alphas = [1, 0.1, 0.01, 0.001]
        for alpha in alphas:
            self.setUKF(alpha=alpha, n_azimuth=8, n_altitude=5)
            self.ukf.kf.x[1] = self.ukf.kf.x[1] + np.pi / 4
            self.ukf.kf.x[2] = self.ukf.kf.x[2] + np.pi / 6

            TEST_CASE_NAME = f"TC2b: GPS Tracked Host, fix connected agent. \n wrong initial estimate: azimuth + $\pi$/2, altitude + $\pi$/4 \n " \
                             f" alpha = {alpha} "

            self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC2c_Moving_GPSTracked_Host_StillAgent_ChangingWrongEstimates(self):
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_gps_tracked_host_fix_connected)

        altitudes = np.array([0, 1 / 12, 1 / 6, 1 / 4])
        azimuths = np.array([0, 1 / 8, 1 / 4, 1])
        for i in range(len(azimuths)):
            self.setUKF(alpha=1)
            self.ukf.kf.x[1] = limit_angle(self.ukf.kf.x[1] + azimuths[i] * np.pi)
            self.ukf.kf.x[2] = limit_angle(self.ukf.kf.x[2] + altitudes[i] * np.pi)
            # self.ukf.calculateRealPose()

            TEST_CASE_NAME = f"TC2c: GPS Tracked Host, fix connected agent. \n wrong initial estimates: azimuth + {azimuths[i]}$\pi$, altitude + {altitudes[i]}$\pi$ \n "
            self.run_test(TEST_CASE_NAME)
        plt.show()

    # --- Test case 3: Sinusoidal moving host, random moving agent.
    def test_tc3_sinusoidal_moving_gps_tracked_host_moving_agent(self):
        TEST_CASE_NAME = "TC3: GPS Tracked Host moving in sinusoidal, random moving connected agent."
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_sinusoidal_gps_tracked_host_random_connected,
                       kwargs={"speed": 1, "w": np.pi / 4, "w_Heading": np.pi / 15})
        self.setUKF()

        self.run_test(TEST_CASE_NAME)

        plt.show()

    def test_tc3a_sinusoidal_moving_gps_tracked_host_moving_agent(self):
        # TEST_CASE_NAME = "TC3: GPS Tracked Host moving in sinusoidal, random moving connected agent."
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_sinusoidal_gps_tracked_host_random_connected,
                       kwargs={"speed": 1, "w": np.pi / 1, "w_Heading": np.pi / 2})
        # self.setUKF()

        alphas = [1, 0.01]
        betas = [0, 2]
        kappas = [-1, 0, 1]
        for alpha in alphas:
            for beta in betas:
                for kappa in kappas:
                    self.setUKF(kappa=kappa, alpha=alpha, beta=beta)
                    TEST_CASE_NAME = f"TC3a Sinusoidal  moving host agent Random Connected Agent alpha ={'c'.join(str(alpha).split('.'))} beta ={beta} kappa = {'c'.join(str(kappa).split('.'))}"
                    self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_tc3a_sinusoidal_moving_gps_tracked_host_moving_agent_wrong_itial_estimation(self):
        TEST_CASE_NAME = "TC3a: GPS Tracked Host moving in sinusoidal, random moving connected agent.\n wrong initial Estimation azimuth += $\pi$/2, altitude += $\pi$/4"
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, moving_sinusoidal_gps_tracked_host_random_connected,
                       {"speed": 1, "w": np.pi / 10, "w_Heading": np.pi / 20})
        self.setUKF(alpha=0.01, n_azimuth=8, n_altitude=5)
        self.ukf.kf.x[1] = self.ukf.kf.x[1] + np.pi / 2
        self.ukf.kf.x[2] = self.ukf.kf.x[2] + np.pi / 4

        self.run_test(TEST_CASE_NAME)
        plt.show()

    # ---- Test cases: moving host , moving connected Agent
    def test_tc4_moving_host_agent_moving_connected_agent(self):
        TEST_CASE_NAME = "TC4: Both agents moving randomly"
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, random_movements_host_random_movements_connected)
        self.bool_host_agent_drift = True

        self.setUKF()
        self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_tc4a_moving_host_agent_moving_connected_agent_wrong_intial_estimate(self):
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, random_movements_host_random_movements_connected)

        altitudes = np.array([0, 1 / 12, 1 / 6, 1 / 4])
        azimuths = np.array([0, 1 / 8, 1 / 4, 1])
        for i in range(len(azimuths)):
            self.setUKF(alpha=0.1, beta=0, kappa=-1)
            self.ukf.kf.x[1] = limit_angle(self.ukf.kf.x[1] + azimuths[i] * np.pi)
            self.ukf.kf.x[2] = limit_angle(self.ukf.kf.x[2] + altitudes[i] * np.pi)
            self.ukf.kf.x[3] = limit_angle(self.ukf.kf.x[3] + azimuths[i] * np.pi)
            # self.ukf.calculateRealPose()

            TEST_CASE_NAME = f"TC4a: Both agents moving randomly. \n wrong initial estimates: azimuth + {azimuths[i]}$\pi$, altitude + {altitudes[i]}$\pi$ \n "
            self.run_test(TEST_CASE_NAME)
        plt.show()

    def test_TC4b_MovingHostAgent_MovingConnectedAgent_WrongIntialEstimate(self):
        TEST_CASE_NAME = "TC4b: Both agents moving randomly, wrong initial estimate \n Azimuth += $\pi$/4 Altitude += $\pi$/8 , heading += $\pi$/4"
        self.initTest()
        run_simulation(self.simulation_time_steps, self.host, self.drone, random_movements_host_random_movements_connected)
        self.setUKF(alpha=0.1, n_azimuth=1, n_altitude=1, n_heading_c=1)
        self.ukf.kf.x[1] = self.ukf.kf.x[1] + np.pi
        self.ukf.kf.x[2] = self.ukf.kf.x[2] + np.pi
        self.ukf.kf.x[3] = self.ukf.kf.x[3] + np.pi

        self.run_test(TEST_CASE_NAME)
        plt.show()

    # ---- Test cases: NLOS  moving host , moving connected Agent

    def random_nlos(self, i):
        if self.ukf.los_state and  np.random.uniform(0, 1) < 0.2:
            self.ukf.switch_los_state()
        elif not self.ukf.los_state and np.random.uniform(0, 1) < 0.2:
            self.ukf.switch_los_state()

    def long_nlos(self,i):
        if i == 400:
            self.ukf.switch_los_state()
        if i == 700:
            self.ukf.switch_los_state()


    def test_TC5_NLOS_random_moving_host_agent_moving_connected_agent(self):
        TEST_CASE_NAME = "TC5: NLOS, Both agents moving randomly"
        self.initTest()
        run_simulation(self.uwb_time_steps, self.host, self.drone, random_movements_host_random_movements_connected)
        self.setUKF()

        self.run_test(TEST_CASE_NAME, nlos_function=self.random_nlos)
        plt.show()

    def test_TC5a_longNLOS_random_moving_host_agent_moving_connected_agent(self):
        TEST_CASE_NAME = "TC5a: long NLOS, Both agents moving randomly"
        self.initTest()
        run_simulation(self.uwb_time_steps, self.host, self.drone, random_movements_host_random_movements_connected)
        self.setUKF()

        self.run_test(TEST_CASE_NAME, nlos_function= self.long_nlos)
        plt.show()
    # Test cases: Copies

    def test_TC6a_Copies(self):
        TEST_CASE_NAME = "TC6a: Copies"
        self.initTest()
        run_simulation(self.uwb_time_steps, self.host, self.drone, random_movements_host_random_movements_connected)
        self.setUKF()

        self.uwb_time_steps = 200
        copiedUKF = None

        # self.ukf.set_datalogger(self.host, self.drone, TEST_CASE_NAME)
        self.ukf.weight = 0.
        # self.ukf.datalogger.log_data(0)
        for i in range(self.uwb_time_steps):
            try:
                if i == int(self.uwb_time_steps//2):
                    copiedUKF  = self.ukf.copy()
                    copiedUKF.switch_los_state()
                self.ukf.weight = 1.

                self.ukf.predict(self.drone.dx_slam[i], self.drone.dh_slam[i])

                distance = np.linalg.norm(self.host.x_real[i] - self.drone.x_real[i])
                uwb_measurement = distance + np.random.randn(1)[0] * self.sigma_uwb
                self.ukf.calculate_r(self.sigma_uwb)
                if np.random.uniform(0, 1) < 0.2:
                    if self.ukf.los_state == 1:
                        self.ukf.los_state = 0
                    else:
                        self.ukf.los_state = 1
                self.ukf.update(uwb_measurement, self.host.x_real[i], self.host.h_real[i])

                self.ukf.datalogger.log_data(i)
                if copiedUKF is not None:
                    copiedUKF.weight = 1.
                    copiedUKF.predict(self.drone.dx_slam[i], self.drone.dh_slam[i])
                    copiedUKF.calculate_r(self.sigma_uwb, self.sigma_dv)
                    copiedUKF.update(uwb_measurement, self.host.x_real[i], self.host.h_real[i])
                    copiedUKF.datalogger.log_data(i)

            except np.linalg.LinAlgError:
                print("Error in ", TEST_CASE_NAME)
                print(traceback.format_exc())
                print(self.ukf.kf.P)
                break

        self.ukf.datalogger.create_3d_plot()
        self.ukf.datalogger.plot_error_graph()
        copiedUKF.datalogger.plot_error_graph()
        copiedUKF.datalogger.create_3d_plot()
        # self.run_test(TEST_CASE_NAME, nlos_function=self.random_nlos)
        plt.show()


    def test_readfile(self):
        fileName = "./Results_UKF/2023_3_15/TC1_dat"
        dl = load_dataFile(fileName)
        dl.create3Dplot()
        dl.plot_ErrorGraph()
        plt.show()


if __name__ == "__main__":
    ut = unittest.main()

