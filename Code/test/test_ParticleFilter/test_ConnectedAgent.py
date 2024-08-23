#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:49:09 2023

@author: yuri
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt

from Code.DataLoggers.TargetTrackingParticle_DataLogger import UKFTargetTrackingParticle_DataLogger
# Have to use this since Spyder_WS is a project.
from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent
from Code.ParticleFilter.TargetTrackingParticle import ListOfUKFLOSTargetTrackingParticles
from Code.DataLoggers.ConnectedAgent_DataLogger import UPFConnectedAgentDataLogger

from Code.Simulation.BiRobotMovement import random_movements_host_random_movements_connected, \
    drone_flight, run_simulation, fix_host_fix_connected
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix
from Code.Simulation.MultiRobotClass import MultiRobotSingleSimulation
from Code.Simulation.RobotClass import NewRobot
from NLOS_RPE_Code.Simulations.NLOS_Manager import NLOS_Manager



class TestConnectedAgent(unittest.TestCase):
    def init_test(self, sigma_v=0.1, sigma_w=0.1, sigma_uwb=0.1, drifting_host=False):
        # Paper = Relative Transformation Estimation Based on Fusion of Odometry and UWB.py Ranging Data

        self.uwb_time_steps = 300  # (120 // 0.03)          # Paper simulation time = 120s
        self.odom_time_step = 0.1
        self.uwb_time_step = 0.1 # Paper experiments UWB.py frequency = 37 Hz
        self.factor = int(self.uwb_time_step / self.odom_time_step)
        self.simulation_time_steps = self.uwb_time_steps * self.factor

        self.sigma_uwb = sigma_uwb  # Paper sigma uwb = 0.1
        self.sigma_v = sigma_v  # Paper sigma odom = 0.001 m -> not sure how this relates with the heading error.
        self.sigma_w = sigma_w  # / 180 * np.pi  # In the paper they use degrees.
        self.nlos_man = NLOS_Manager(nlos_bias=2.)
        self.los = []
        self.drifting_host = drifting_host
        self.debug = False

    def init_drones(self, x_ca_0, h_ca_0, max_range=None):
        self.max_range = max_range
        ha_pose_0 = np.array([0, 0, 0, 0])
        ca_pose_0 = np.concatenate([x_ca_0, np.array([h_ca_0])])
        self.drone = drone_flight(ca_pose_0, sigma_dv=self.sigma_v, sigma_dw=self.sigma_w, max_range=self.max_range,
                                  origin_bool=True, simulation_time_step=self.odom_time_step)
        self.host = drone_flight(ha_pose_0, sigma_dv=self.sigma_v, sigma_dw=self.sigma_w, max_range=self.max_range,
                                 origin_bool=True, simulation_time_step=self.odom_time_step)

        distance = np.linalg.norm(self.drone.x_start - self.host.x_start)
        self.startMeasurement = distance + np.random.randn(1) * self.sigma_uwb

    def run_test(self, nlos_function, name="Unidentified Test"):
        # self.ca.set_logging(self.dl)
        # self.dl.log_data(0)
        dx_ca = np.zeros(4)
        q = np.zeros((4, 4))
        q_ha = np.zeros((4, 4))
        for i in range(1, self.simulation_time_steps):
            print("Simulation step: ", i, " /", self.simulation_time_steps)
            # print(len(self.ca.particles))
            # try:
            d_dx_ca = np.concatenate((self.drone.dx_slam[i], np.array([self.drone.dh_slam[i]])))
            f = get_4d_rot_matrix(dx_ca[-1])
            dx_ca = dx_ca + f @ d_dx_ca
            q = q + f @ self.drone.q @ f.T
            q_ha = q_ha + self.host.q

            # if self.drifting_host:
            d_dx_ha = np.concatenate((self.host.dx_slam[i], np.array([self.host.dh_slam[i]])))


            self.ca.ha.predict(d_dx_ha, self.host.q)

            if i % self.factor == 0:
                distance = np.linalg.norm(self.drone.x_real[i] - self.host.x_real[i])
                uwb_measurement = distance + np.random.randn(1)[0] * self.sigma_uwb
                uwb_measurement, los_state = nlos_function(int(i / self.factor), uwb_measurement)
                self.los.append(los_state)

                if self.drifting_host:
                    x_ha = self.host.x_slam[i]
                    h_ha = self.host.h_slam[i]
                    x_ha = np.concatenate([x_ha, np.array([h_ha])])
                else:
                    x_ha = self.host.x_real[i]
                    h_ha = self.host.h_real[i]
                    x_ha = np.concatenate([x_ha, np.array([h_ha])])

                self.ca.ha.update(x_ha, q_ha)
                self.ca.run_model(dt_j = dx_ca,  q_j=q, dt_i = np.zeros(4), q_i= np.zeros((4,4)), d_ij= uwb_measurement)

                self.dl.log_data(i)

                dx_ca = np.zeros(4)
                q = np.zeros((4, 4))
                self.ca.ha.reset_integration()
                # if not self.drifting_host:
                q_ha = np.zeros((4, 4))
            # except Exception as e:
            #     print(e)
            #     break

    def test_tc1(self):
        # Length of NLOS  is proportional to error on odom?
        # TODO: There is an interplay between sigma_v and sigma_uwb:
        #  More specifically if sigma_dx = sigma_dv* dt > sigma_uwb/10 it seems the UPF becomes over confident.
        self.init_test(sigma_v=0.001, sigma_w=0.001, sigma_uwb=0.1,
                       drifting_host=True)
        self.init_drones(np.array([15., 15., 0]), 0, max_range=25)
        # self.init_drones(np.array([22,0,0]), 0, max_range=40)
        run_simulation(self.simulation_time_steps, self.host, self.drone,
                       random_movements_host_random_movements_connected)
        # run_simulation(self.simulation_time_steps, self.host, self.drone,
        #                fix_host_fix_connected)
        lop = ListOfUKFLOSTargetTrackingParticles()
        lop.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        t_i = np.concatenate((self.host.x_start, [self.host.h_start]))
        lop.split_sphere_in_equal_areas(t_i= t_i, d_ij= self.startMeasurement[0], sigma_uwb= self.sigma_uwb,
                                        n_altitude=3, n_azimuth=4, n_heading=4 )

        self.ca = UPFConnectedAgent(lop.particles, x_ha_0=np.concatenate((self.host.x_start, [self.host.h_start])),
                                    sigma_uwb=self.sigma_uwb, sigma_uwb_factor=1., resample_factor=0.1, id="0x000")

        # self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        # self.ca.set_normal_resampling(resample_factor=0.1, uwb_sigma_factor=2.)
        # self.ca.set_branch_kill_resampling(resample_factor=0.5, sigma_uwb_factor=2.)
        # self.ca.split_sphere_in_equal_areas(self.startMeasurement[0], self.sigma_uwb,
        #                                     n_altitude=3, n_azimuth=4, n_heading=4)


        # self.ca.set_regeneration_parameters()
        self.dl = UPFConnectedAgentDataLogger(self.host, self.drone, self.ca, particle_type=UKFTargetTrackingParticle_DataLogger)
        self.dl.log_data(0)

        self.run_test(nlos_function=self.nlos_man.los)

        self.dl.plot_self(self.los)
        self.dl.get_best_particle_log().create_3d_plot()
        self.dl.get_best_particle_log().plot_error_graph()
        self.dl.get_best_particle_log().plot_ukf_states()
        plt.show()

    def test_tc1_known_start_pose(self):
        self.init_test(sigma_v=0.001, sigma_w=0.001, sigma_uwb=0.1,
                       drifting_host=True)
        self.init_drones(np.array([5, 5, 0]), 0, max_range=20)
        run_simulation(self.simulation_time_steps, self.host, self.drone,
                       random_movements_host_random_movements_connected)
        self.ca = UPFConnectedAgent([], x_ha_0=np.concatenate((self.host.x_start, [self.host.h_start])),
                                    sigma_uwb_factor=1., resample_factor=0.1, id="0x000")
        self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        # x_ca_0 = np.array([5, 5, 0, 0])
        x_ha_0 = np.concatenate((self.host.x_start, np.array([self.host.h_start])))
        x_ca_0 = np.concatenate((self.drone.x_start, np.array([self.drone.h_start])))
        x_ca = get_4d_rot_matrix(-x_ha_0[-1]) @ (x_ca_0 - x_ha_0)
        self.ca.add_particle_with_know_start_pose(x_ca_0=x_ca, azimuth_n=100, altitude_n=100, heading_n=100,
                                                  sigma_uwb=self.sigma_uwb)

        # sigma2_dx_ha = (self.sigma_dv * self.odom_time_step) ** 2
        # sigma2_dh_ha = (self.sigma_dw * self.odom_time_step) ** 2
        # self.ca.ha.kf.P = np.diag([sigma2_dx_ha, sigma2_dx_ha, sigma2_dx_ha, sigma2_dh_ha])

        self.run_test(nlos_function=self.nlos_man.los)

        self.dl.plot_self(self.los)
        # self.dl.plot_start_poses()
        # self.dl.plot_best_particle_variance_graph()
        plt.show()

    def test_tc2(self):
        # Length of NLOS  is proportional to error on odom?
        self.init_test(sigma_v=0.1, sigma_w=0.001, sigma_uwb=0.1,
                       drifting_host=True)
        self.init_drones(np.array([2, 0, 0]), 0, max_range=3)
        run_simulation(self.simulation_time_steps, self.host, self.drone,
                       fix_host_fix_connected)
        self.ca = UPFConnectedAgent([], x_ha_0=np.concatenate((self.host.x_start, [self.host.h_start])), id="0x000")
        self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        self.ca.split_sphere_in_equal_areas(self.startMeasurement[0], 2*self.sigma_uwb,
                                            n_altitude=3, n_azimuth=4, n_heading=4)

        self.run_test(nlos_function=self.nlos_man.los)

        self.dl.plot_self(self.los)
        self.dl.get_best_particle_log().create_3d_plot()
        self.dl.get_best_particle_log().plot_error_graph()
        self.dl.get_best_particle_log().plot_ukf_states()
        plt.show()

    # -----------------------
    # Precalulated trajectories
    # -----------------------
    def load_drones(self, folder_name, sigma_v=0.1, sigma_w=0.1, sigma_uwb=0.1):
        self.uwb_time_step = 1
        self.sigma_uwb = sigma_uwb
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w

        MRS = MultiRobotSingleSimulation(folder_name)
        MRS.load_simulation(self.sigma_v, self.sigma_w, self.sigma_uwb)
        self.agents= {}
        for drone in MRS.drones:
            self.agents[drone]={"drone": MRS.drones[drone]}
        self.distances = MRS.get_uwb_measurements("drone_0", "drone_1")
        self.odom_time_step = MRS.parameters["simulation_time_step"]
        self.uwb_time_steps = MRS.parameters["simulation_time"]
        self.simulation_time_steps = int(self.uwb_time_steps / self.odom_time_step)
        self.factor = int(self.uwb_time_step / self.odom_time_step)

        self.nlos_man = NLOS_Manager(nlos_bias=2.)
        self.los = []

    def run_preloaded_simulation_test(self, nlos_function, name="Unidentified Test", simulation_time_steps=None):
        if simulation_time_steps is None:
            simulation_time_steps = self.simulation_time_steps

        for i in range(0, simulation_time_steps):
            print("Simulation step: ", i, " /", simulation_time_steps)
            for agent in self.agents:
                drone = self.agents[agent]["drone"]
                dx = np.concatenate((drone.dx_slam[i], np.array([drone.dh_slam[i]])))
                self.agents[agent]["upf"].ha.predict(dx, drone.q)

            if i % self.factor == 0 and i > 0:
                drone_0: NewRobot = self.agents["drone_0"]["drone"]
                drone_1: NewRobot = self.agents["drone_1"]["drone"]

                x_0 = np.concatenate([drone_0.x_slam[i], np.array([drone_0.h_slam[i]])])
                dx_0, q_0 = self.agents["drone_0"]["upf"].ha.reset_integration()
                self.agents["drone_0"]["upf"].ha.update(x_0, q_0)

                x_1 = np.concatenate([drone_1.x_slam[i], np.array([drone_1.h_slam[i]])])
                dx_1, q_1 = self.agents["drone_1"]["upf"].ha.reset_integration(x_1)
                self.agents["drone_1"]["upf"].ha.update(x_1, q_1)

                uwb_measurement = self.distances[i]
                uwb_measurement, los_state = nlos_function(int(i / self.factor), uwb_measurement)
                self.los.append(los_state)

                self.agents["drone_0"]["upf"].run_model(dx_1, uwb_measurement, q_i=q_1, time_i=i)
                self.agents["drone_1"]["upf"].run_model(dx_0, uwb_measurement, q_i=q_0, time_i=i)

                self.agents["drone_0"]["dl"].log_data(i)
                self.agents["drone_1"]["dl"].log_data(i)
                # self.agents["drone_0"]["dl"].plot_self(self.los)
                # self.agents["drone_1"]["dl"].plot_self(self.los)
                # plt.show()
                if self.debug:
                    self.agents["drone_0"]["dl"].plot_self(self.los)
                    self.agents["drone_1"]["dl"].plot_self(self.los)
                    plt.show()

    def test_tc2_prerecorded(self):
        n_altitude = 3
        n_azimuth = 4
        n_heading = 4
        NLOS_bool = True
        sim_folder = "../test_cases/RPE_2_agents_LOS/simulations/robot_trajectories/test/sim_3"
        self.load_drones(sim_folder, sigma_v=0.15, sigma_w=0.05, sigma_uwb=0.15)
        dis = self.agents["drone_0"]["drone"].x_real - self.agents["drone_1"]["drone"].x_real
        distances = np.linalg.norm(dis.astype(np.float64), axis=1)
        self.debug = True
        # fig, ax = plt.subplots(2, 2)
        # labels = ["x", "y", "z"]
        # for j, agent in enumerate(self.agents):
        #     drone: NewRobot = self.agents[agent]["drone"]
        #     for i in range(3):
        #         ax[0, j].plot(drone.dx_slam[:, i] - drone.dx_slam_real[:, i], label=labels[i])
        #
        #     ax[0, j].plot(drone.dh_slam - drone.dh_slam_real, label="h")
        #     ax[0, j].legend()
        # ax[1,1].plot(self.distances - distances)
        # plt.show()

        for agent in self.agents:
            drone : NewRobot = self.agents[agent]["drone"]
            for agent1 in self.agents:
                if agent1 != agent:
                    other_drone : NewRobot = self.agents[agent1]["drone"]
            x_0 = np.concatenate((drone.x_start, np.array([drone.h_start])))
            upf = UPFConnectedAgent([], x_ha_0=x_0, id=agent)
            upf.set_ukf_parameters(kappa=-1, alpha=1, beta=2, drift_correction_bool=True)
            upf.set_nlos_detection_parameters(min_likelihood=0.1, degeneration_factor=0.9, min_likelihood_factor=0.2)
            upf.split_sphere_in_equal_areas(self.distances[0], self.sigma_uwb,
                                              n_altitude=n_altitude, n_azimuth=n_azimuth, n_heading=n_heading)
            dl = UPFConnectedAgentDataLogger(drone, other_drone, upf)
            upf.set_logging(dl)
            self.agents[agent]["upf"] = upf
            self.agents[agent]["dl"] = dl

        self.run_preloaded_simulation_test(nlos_function=self.nlos_man.los, simulation_time_steps=500)

        for agent in self.agents:
            self.agents[agent]["dl"].plot_self(self.los)
            # self.ca_0.best_particle.datalogger.create_3d_plot()
            self.agents[agent]["upf"].best_particle.datalogger.plot_error_graph()
            self.agents[agent]["upf"].best_particle.datalogger.plot_ukf_states()


        plt.show()

        """
        self.ca = UPFConnectedAgent("0x000", x_ha_0=np.concatenate((self.host.x_start, [self.host.h_start])))
        self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        # x_ca_0 = np.array([5, 5, 0, 0])
        x_ha_0 = np.concatenate((self.host.x_start, np.array([self.host.h_start])))
        x_ca_0 = np.concatenate((self.drone.x_start, np.array([self.drone.h_start])))
        x_ca = get_transformation_matrix(-x_ha_0[-1]) @ (x_ca_0 - x_ha_0)
        self.ca.add_particle_with_know_start_pose(x_ca_0=x_ca, P_x_ca_0=np.eye(4)*1e-8, sigma_uwb=self.sigma_uwb)



        sigma2_dx_ha = (self.sigma_v * self.odom_time_step) ** 2
        sigma2_dh_ha = (self.sigma_w * self.odom_time_step) ** 2
        self.ca.ha.kf.P = np.diag([sigma2_dx_ha, sigma2_dx_ha, sigma2_dx_ha, sigma2_dh_ha])

        self.run_test(nlos_function=None)

        self.dl.plot_self(self.los)
        # self.dl.plot_start_poses()
        # self.dl.plot_best_particle_variance_graph()
        plt.show()
        """



if __name__ == "__main__":
    unittest.main()
