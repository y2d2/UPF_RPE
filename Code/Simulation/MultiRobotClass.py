#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:21:41 2023

@author: yuri
"""

import numpy as np
import pickle as pkl
import os
import shutil
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

from Code.DataLoggers.TargetTrackingParticle_DataLogger import UKFTargetTrackingParticle_DataLogger, TargetTrackingParticle_DataLogger
# import Experiments.RealRobot
from Code.UtilityCode.utility_fuctions import sphericalToCartesian, limit_angle, transform_matrix, \
    inv_transformation_matrix, get_states_of_transform, cartesianToSpherical
from Code.Simulation.BiRobotMovement import run_multi_drone_simulation, drone_flight, random_moving_drones
from Code.Simulation.RobotClass import load_trajectory_from_dict, NewRobot
# from NLOS_RPE_Code.Simulations.NLOS_Manager import NLOS_Manager
from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent
from Code.ParticleFilter.TargetTrackingParticle import ListOfUKFLOSTargetTrackingParticles, UKFLOSTargetTrackingParticle
from Code.DataLoggers.ConnectedAgent_DataLogger import UPFConnectedAgentDataLogger
from Code.DataLoggers.TargetTrackingUKF_DataLogger import UKFDatalogger
from Code.BaseLines.AlgebraicMethod4DoF import AlgebraicMethod4DoF, Algebraic4DoF_Logger
from Code.BaseLines.NLS import NLS
from Code.DataLoggers.NLS_DataLogger import NLSDataLogger
from Code.BaseLines.QCQP import QCQP
from Code.DataLoggers.QCQP_DataLogger import QCQP_Log


class MultiRobotSingleSimulation:
    """
    This class will group all the data from a single simulation into 1 file.
    1 single simulation is defined as 1 set of trajectories.
    Each set of trajectories can have multiple simulation parameters which will be kept in this file.
    THe structure of the file is as follows:
    sim = { "parameters: { "simulation_time_step" : simulation_time_step, "simulation_time" : simulation_time,},
            "trajectories": { "drone_id": drone_trajectory , ...)} ,
            "sim": [ { "parameters" : { "sigma_v" : sigma_v, "sigma_w" : sigma_w, "sigma_d": sigma_d},
                     "drone_id" : { "dx_slam" : dx_slam , "dh_slam" : dh_slam},
                     "distances" : [{ "drone_id1=s" : [drone_id1, drone_id2], "d" : d}, ...] }, ...]
            }
    """

    def __init__(self, folder):
        self.folder = folder
        self.simulation_name = folder.split("/")[-1]

        # simulation variables:
        self.sim = None
        self.current_sim = None
        self.drones = {}
        self.distances = None
        self.parameters = None
        # plotting variables:
        self.colors = None
        self.ax = None
        self.interactive_mode = False
        self.plt_pause = 0.00001

    # ------------------
    # Trajectory functions:
    # ------------------
    def check_trajectory(self):
        # check if the trajectory has been loaded:
        if self.sim is None:
            self.load_file()
        # Check existence of trajectories: if they exist, return False
        # if needed to remake, remove the folder.
        if self.sim["trajectories"] is not None:
            return True
        return False

    def load_trajectory(self):
        if not self.check_trajectory():
            return False
        self.parameters = self.sim["parameters"]
        for drone_name in self.sim["trajectories"]:
            self.drones[drone_name] = load_trajectory_from_dict(self.sim["trajectories"][drone_name])
        return True

    def create_trajectory(self, max_v=1, max_w=0.05, slowrate_v=0.02, slowrate_w=0.001,
                          simulation_time_step=0.2, simulation_time=1000, max_range=20, range_origin_bool=True,
                          start_poses=None):
        if self.check_trajectory():
            return False

        self.sim["parameters"] = {"simulation_time_step": simulation_time_step, "simulation_time": simulation_time,
                                  "max_range": max_range, "range_origin_bool": range_origin_bool,
                                  "max_v": max_v, "max_w": max_w, "slowrate_v": slowrate_v, "slowrate_w": slowrate_w, }

        simulation_time_steps = int(simulation_time / simulation_time_step)
        self.sim["parameters"]["simulation_time_steps"] = simulation_time_steps
        self.parameters = self.sim["parameters"]

        self.sim["trajectories"] = {}
        drones = []
        # for j in range(self.number_of_drones):
        for start_pose in start_poses:
            drones.append(drone_flight(start_pose, start_velocity=np.array([max_v, 0, 0, 0]), sigma_dv=0, sigma_dw=0,
                                       max_range=max_range,
                                       origin_bool=range_origin_bool, simulation_time_step=simulation_time_step,
                                       slowrate_v=slowrate_v, slowrate_w=slowrate_w, max_v=max_v, max_w=max_w))

        run_multi_drone_simulation(simulation_time_steps, drones, random_moving_drones)
        for j, drone in enumerate(drones):
            self.drones["drone_" + str(j)] = drone
            self.sim["trajectories"]["drone_" + str(j)] = drone.get_trajectory_dict()

        self.save_file()
        self.create_trajectory_image()
        return True

    # ------------------
    # Simulation functions:
    # ------------------
    def check_simulation(self, sigma_v, sigma_w, simga_d):
        if not self.load_trajectory():
            raise Exception("Trajectories have not been created. Please create trajectories first.")

        for sim in self.sim["sim"]:
            if sim["parameters"]["sigma_v"] == sigma_v and sim["parameters"]["sigma_w"] == sigma_w and \
                    sim["parameters"]["sigma_d"] == simga_d:
                return sim
        return None

    def run_simulation(self, sigma_v, sigma_w, sigma_d):
        self.current_sim = self.check_simulation(sigma_v, sigma_w, sigma_d)
        if self.current_sim is None:
            self.current_sim = {"parameters": {"sigma_v": sigma_v, "sigma_w": sigma_w, "sigma_d": sigma_d}}

            # calculate slam trajectories.
            for drone_name in self.drones:
                dx_slam, dh_slam = self.drones[drone_name].calculate_d_slam(sigma_v, sigma_w)
                self.drones[drone_name].calculate_d_slam_drift()
                self.current_sim[drone_name] = {"dx_slam": dx_slam, "dh_slam": dh_slam}

            # calculate distances
            self.current_sim["distances"] = []
            for drone_name_0 in self.drones:
                for drone_name_1 in self.drones:
                    if drone_name_0 != drone_name_1:
                        existing = False
                        for d in self.current_sim["distances"]:
                            if drone_name_0 in d["drone_ids"] and drone_name_1 in d["drone_ids"]:
                                existing = True
                                break
                        if not existing:
                            distances = self.drones[drone_name_0].x_real - self.drones[drone_name_1].x_real
                            distances = np.linalg.norm(distances.astype(np.float64), axis=1)
                            distances = distances + np.random.randn(*distances.shape) * sigma_d
                            self.current_sim["distances"].append(
                                {"drone_ids": [drone_name_0, drone_name_1], "d": distances})
            self.sim["sim"].append(self.current_sim)
            self.save_file()
        else:
            for drone_name in self.drones:
                self.drones[drone_name].dx_slam = self.current_sim[drone_name]["dx_slam"]
                self.drones[drone_name].dh_slam = self.current_sim[drone_name]["dh_slam"]
                self.drones[drone_name].set_uncertainties(sigma_v, sigma_w)
                self.drones[drone_name].calculate_d_slam_drift()

    def load_simulation(self, sigma_v, sigma_w, sigma_d):
        self.current_sim = self.check_simulation(sigma_v, sigma_w, sigma_d)
        if self.current_sim is None:
            self.run_simulation(sigma_v, sigma_w, sigma_d)
            # raise Exception("Simulation does not exist. Please run simulation first.")

        for drone_name in self.drones:
            self.drones[drone_name].dx_slam = self.current_sim[drone_name]["dx_slam"]
            self.drones[drone_name].dh_slam = self.current_sim[drone_name]["dh_slam"]
            self.drones[drone_name].set_uncertainties(sigma_v, sigma_w)
            self.drones[drone_name].calculate_d_slam_drift()

    # ------------------
    #  distance functions:
    # ------------------
    def get_uwb_measurements(self, id_1, id_2):
        for distance in self.current_sim["distances"]:
            if id_1 in distance["drone_ids"] and id_2 in distance["drone_ids"]:
                return distance["d"]

    # ------------------
    # Plotting functions:
    # ------------------
    def set_colors(self):
        if self.colors is None:
            colormap = matplotlib.colormaps["nipy_spectral"]  # nipy_spectral, Set1,Paired
            colors = [colormap(i) for i in np.linspace(0, 1, len(self.drones) + 2)]
            self.colors = {}
            for i, drone_name in enumerate(self.drones):
                self.colors[drone_name] = colors[i + 1]

    def create_3D_trajectory_ax(self):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        self.ax = plt.axes(projection="3d")

    def set_ax_range(self):
        if self.parameters["max_range"] is not None:
            self.ax.set_xticks(range(-self.parameters["max_range"], self.parameters["max_range"], 5))
            self.ax.set_yticks(range(-self.parameters["max_range"], self.parameters["max_range"], 5))
            self.ax.set_zticks(range(-self.parameters["max_range"], self.parameters["max_range"], 5))

            ax_range = self.parameters["max_range"] - 0.2
            self.ax.axes.set_xlim3d(left=-ax_range, right=ax_range)
            self.ax.axes.set_ylim3d(bottom=-ax_range, top=ax_range)
            self.ax.axes.set_zlim3d(bottom=-ax_range, top=ax_range)

    def init_plot(self, interactive=False):
        self.set_colors()
        self.create_3D_trajectory_ax()
        if interactive:
            self.interactive_mode = True
            plt.ion()
        self.set_ax_range()

    def create_trajectory_image(self):
        self.init_plot()

        for drone_name in self.drones:
            drone = self.drones[drone_name]
            drone.set_plotting_settings(color=self.colors[drone_name], linestyle="--")
            drone.plot_real_position(self.ax, annotation=drone_name)

        plt.legend()
        plt.savefig(self.folder + "/trajectories.png")
        plt.close()

    def plot_simulation(self):
        self.init_plot()

        for drone_name in self.drones:
            drone = self.drones[drone_name]
            drone.set_plotting_settings(color=self.colors[drone_name], linestyle="--")
            drone.plot_real_position(self.ax, annotation=drone_name)
            drone.plot_slam_position(self.ax, annotation=drone_name)
        plt.legend()

    def plot_trajectories(self):
        self.load_trajectory()
        self.init_plot()
        self.plot_trajectories_evolution(-1)
        plt.show()

    def plot_trajectories_evolution(self, t, history=None):
        if history is None or history > t:
            start_time = 0
        else:
            start_time = t - history

        self.ax.clear()
        self.set_ax_range()

        self.ax.set_title(
            self.simulation_name + ": Trajectories from time step " + str(start_time) + " to " + str(t))
        for drone_name in self.drones:
            drone = self.drones[drone_name]
            self.ax.plot3D(drone.x_real[start_time:t, 0], drone.x_real[start_time:t, 1],
                           drone.x_real[start_time:t, 2],
                           color=self.colors[drone_name], linestyle="--", label=drone_name)
        plt.legend()
        if self.interactive_mode:
            plt.pause(self.plt_pause)

    # ------------------
    # File manipulation functions:
    # ------------------

    def delete_sim(self, sigma_v, sigma_w, sigma_uwb):
        self.load_file()
        sim = self.check_simulation(sigma_v, sigma_w, sigma_uwb)
        if sim is not None:
            self.sim["sim"].remove(sim)
            self.save_file()

    def delete(self):
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)

    def load_file(self):
        if os.path.exists(self.folder):
            with open(self.folder + "/trajectories.pkl", "rb") as f:
                self.sim = pkl.load(f)
            f.close()
        else:
            os.mkdir(self.folder)
            self.sim = {"trajectories": None,
                        "sim": []}

    def save_file(self):
        with open(self.folder + "/trajectories.pkl", "wb") as f:
            pkl.dump(self.sim, f)
        f.close()


class MultiRobotSimulation:
    def __init__(self, trajectory_folder_name="./robot_trajectories", reset_trajectories=False):
        # ---- Drones
        self.drones = {}
        self.simulation_name = ""

        # ---- Experiment parameters:
        self.sigma_dx_r = 1.
        self.sigma_dh_r = 1.
        self.sigma_dv = 0.01
        self.sigma_dw = 0.01
        self.sigma_uwb = 0.1
        self.kappa = -1
        self.alpha = 1
        self.beta = 2

        # ---- Simulation parameters
        self.max_v = 1
        self.max_w = 0.05
        self.slowrate_v = 0.02
        self.slowrate_w = 0.001
        self.simulation_time_step = 0.2
        # self.uwb_time_step = 1
        self.simulation_time = 1000
        # self.simulation_time_steps = 1000
        self.simulation_time_steps = int(self.simulation_time / self.simulation_time_step)
        self.number_of_drones = 5
        self.max_range = None
        self.range_origin_bool = True

        # Trajectory folder:
        self.trajectory_folder_name = trajectory_folder_name
        self.reset_trajectories = reset_trajectories

        # Plotting parameters:
        self.colors = {"drone_0": "dimgrey", "drone_1": "steelblue", "drone_2": "crimson", "drone_3": "darkgreen",
                       "drone_4": "olive", "drone_5": "darkorange"}
        self.ax = None
        self.interactive_mode = False
        self.plt_pause = 0.00001

    # ---- Simulation methods
    def set_simulation_parameters(self, max_v=1, max_w=0.05, slowrate_v=0.02, slowrate_w=0.001,
                                  simulation_time_step=0.2, simulation_time=1000,
                                  number_of_drones=5, max_range=20, range_origin_bool=True,
                                  trajectory_folder_name="./robot_trajectories", reset_trajectories=False):
        self.max_v = max_v
        self.max_w = max_w
        self.slowrate_v = slowrate_v
        self.slowrate_w = slowrate_w
        self.simulation_time_step = simulation_time_step
        self.simulation_time = simulation_time
        self.simulation_time_steps = int(self.simulation_time / self.simulation_time_step)
        self.number_of_drones = number_of_drones
        self.range_origin_bool = range_origin_bool
        if self.range_origin_bool:
            self.max_range = max_range

        # Trajectory folder:
        self.trajectory_folder_name = trajectory_folder_name
        self.reset_trajectories = reset_trajectories

    def create_unobservable_trajectories(self, start_range=10):
        # TODO make this function.
        if not os.path.isdir(self.trajectory_folder_name):
            os.mkdir(self.trajectory_folder_name)
        elif self.reset_trajectories:
            shutil.rmtree(self.trajectory_folder_name)
            os.mkdir(self.trajectory_folder_name)
        trajectories = ["parallel_flying", "Standing_still_1", "Standing_still_2", "Horizontal_plane"]
        for trajectory in trajectories:
            self.simulation_name = "/sim_" + trajectory

    def create_hard_trajectories(self, number, start_range):
        if not os.path.isdir(self.trajectory_folder_name):
            os.mkdir(self.trajectory_folder_name)
        elif self.reset_trajectories:
            shutil.rmtree(self.trajectory_folder_name)
            os.mkdir(self.trajectory_folder_name)
        for i in range(number):
            self.simulation_name = "/sim_" + str(i)
            if not os.path.isdir(self.trajectory_folder_name + self.simulation_name):
                os.mkdir(self.trajectory_folder_name + self.simulation_name)
                drones = []
                sim = {}
                # for j in range(self.number_of_drones):
                startpose1 = sphericalToCartesian([0,
                                                   np.random.uniform(-np.pi, np.pi),
                                                   np.random.uniform(-np.pi / 2, np.pi / 2)])
                startpose1 = np.concatenate([startpose1, np.array([np.random.uniform(-np.pi, np.pi)])])
                drones.append(
                    drone_flight(startpose1, sigma_dv=0, sigma_dw=0, max_range=self.max_range,
                                 origin_bool=self.range_origin_bool,
                                 simulation_time_step=self.simulation_time_step, slowrate_v=self.slowrate_v,
                                 slowrate_w=self.slowrate_w, max_v=self.max_v, max_w=self.max_w))

                startpose2 = sphericalToCartesian([start_range,
                                                   np.random.uniform(-np.pi, np.pi),
                                                   np.random.uniform(-np.pi / 2, np.pi / 2)])
                startpose2 = np.concatenate([startpose2, np.array([np.random.uniform(-np.pi, np.pi)])])
                drones.append(
                    drone_flight(startpose2, sigma_dv=0, sigma_dw=0, max_range=self.max_range,
                                 origin_bool=self.range_origin_bool,
                                 simulation_time_step=self.simulation_time_step, slowrate_v=self.slowrate_v,
                                 slowrate_w=self.slowrate_w, max_v=self.max_v, max_w=self.max_w))

                run_multi_drone_simulation(self.simulation_time_steps, drones, random_moving_drones)

                for j, drone in enumerate(drones):
                    self.drones["drone_" + str(j)] = {"drone": drone}
                    sim["drone_" + str(j)] = drone.get_trajectory_dict()
                with open(self.trajectory_folder_name + self.simulation_name + "/trajectories.pkl", "wb") as f:
                    pkl.dump(sim, f)
                f.close()
                # pkl.dump(sim, open(self.trajectory_folder_name + "/sim_" + str(i) + "/trajectories.pkl", "wb"))
                self.create_trajectory_image(drones, self.trajectory_folder_name + "/sim_" + str(i))

    def create_trajectories(self, number):
        if not os.path.isdir(self.trajectory_folder_name):
            os.mkdir(self.trajectory_folder_name)
        elif self.reset_trajectories:
            shutil.rmtree(self.trajectory_folder_name)
            os.mkdir(self.trajectory_folder_name)
        for i in range(number):
            print("Generating trajectory: " + str(i) + " of " + str(number) + " trajectories.")
            self.simulation_name = "/sim_" + str(i)
            startpose1 = sphericalToCartesian([np.random.uniform(0, self.max_range),
                                               np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi / 2, np.pi / 2)])
            startpose1 = np.concatenate([startpose1, np.array([np.random.uniform(-np.pi, np.pi)])])
            startpose2 = sphericalToCartesian([np.random.uniform(0, self.max_range),
                                               np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi / 2, np.pi / 2)])
            startpose2 = np.concatenate([startpose2, np.array([np.random.uniform(-np.pi, np.pi)])])

            traj = MultiRobotSingleSimulation(self.trajectory_folder_name + "/" + self.simulation_name)
            traj.create_trajectory(max_v=self.max_v, max_w=self.max_w,
                                   slowrate_v=self.slowrate_v, slowrate_w=self.slowrate_w,
                                   simulation_time_step=self.simulation_time_step, simulation_time=self.simulation_time,
                                   max_range=self.max_range, range_origin_bool=self.range_origin_bool,
                                   start_poses=[startpose1, startpose2])

    # ---- Recreate Trajectories ----
    def recreate_trajectories(self, simulation_name):
        sim_folder = self.trajectory_folder_name + "/" + simulation_name
        traj = MultiRobotSingleSimulation(sim_folder)
        traj.run_simulation(self.sigma_dv, self.sigma_dw, self.sigma_uwb)

    def set_uncertainties(self, sigma_dv=1., sigma_dw=1., sigma_uwb=0.1):
        self.sigma_dv = sigma_dv
        self.sigma_dw = sigma_dw
        self.sigma_uwb = sigma_uwb

    def run_simulations(self, sigma_vs, sigma_ws, sigma_ds):
        total = len(sigma_vs)
        if len(sigma_ws) != total or len(sigma_ds) != total:
            print("Error: sigma lists are not of equal length")
            return

        for sim_folder in os.listdir(self.trajectory_folder_name):
            for i in range(total):
                self.set_uncertainties(sigma_dv=sigma_vs[i], sigma_dw=sigma_ws[i], sigma_uwb=sigma_ds[i])
                traj = MultiRobotSingleSimulation(self.trajectory_folder_name + "/" + sim_folder)
                traj.run_simulation(self.sigma_dv, self.sigma_dw, self.sigma_uwb)
                traj.plot_simulation()

    # ---- Plotting methods ----
    def create_trajectory_image(self, list_of_drone, folder):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        ax = plt.axes(projection="3d")
        for drone_name in self.drones:
            drone = self.drones[drone_name]["drone"]
            drone.set_plotting_settings(color=self.colors[drone_name], linestyle="--")
            drone.plot_real_position(ax, annotation=drone_name)

        plt.legend()
        plt.savefig(self.trajectory_folder_name + "/" + self.simulation_name + "/trajectories.png")
        plt.close()

    def init_plot(self, interactive=False):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        self.ax = plt.axes(projection="3d")
        if interactive:
            self.interactive_mode = True
            plt.ion()
        self.set_ax_range()

    def set_ax_range(self):
        if self.max_range is not None:
            self.ax.set_xticks(range(-self.max_range, self.max_range, 5))
            self.ax.set_yticks(range(-self.max_range, self.max_range, 5))
            self.ax.set_zticks(range(-self.max_range, self.max_range, 5))

            ax_range = self.max_range - 0.2
            self.ax.axes.set_xlim3d(left=-ax_range, right=ax_range)
            self.ax.axes.set_ylim3d(bottom=-ax_range, top=ax_range)
            self.ax.axes.set_zlim3d(bottom=-ax_range, top=ax_range)

    def plot_trajectories(self, simulation):
        self.init_plot()
        self.recreate_trajectories(simulation)
        self.plot_trajectories_evolution(-1)
        plt.show()

    def plot_trajectories_evolution(self, time, history=None):
        if history is None or history > time:
            start_time = 0
        else:
            start_time = time - history
        self.ax.clear()
        self.set_ax_range()

        self.ax.set_title(
            self.simulation_name + ": Trajectories from time step " + str(start_time) + " to " + str(time))
        for drone_name in self.sim_drones:
            drone = self.sim_drones[drone_name]["drone"]
            self.ax.plot3D(drone.x_real[start_time:time, 0], drone.x_real[start_time:time, 1],
                           drone.x_real[start_time:time, 2],
                           color=self.colors[drone_name], linestyle="--", label=drone_name)
        plt.legend()
        if self.interactive_mode:
            plt.pause(self.plt_pause)


class TwoAgentSystem():
    def __init__(self, trajectory_folder, result_folder):
        # Agents variables:
        self.number_of_agents = 2

        # data variables:
        self.data = None

        # Simulation variables:
        self.agents = None
        self.sim: MultiRobotSingleSimulation = None
        # self.uwb_rate = 1. # second
        self.frequency = None
        self.factor = None
        # self.nlos_man = NLOS_Manager(nlos_bias=2.)
        self.los_state = []
        self.uwb_error = []
        self.d0 = None

        # Experiment variables:
        self.experiment_data = None

        #Parameters
        self.parameters = {}
        self.type = ""
        self.prefix = ""

        # QCQP and Algebraic parameters:
        self.horizon = 100

        # UPF Parameters:
        self.kappa = None
        self.alpha = None
        self.beta = None
        self.n_azimuth = None
        self.n_altitude = None
        self.n_heading = None
        self.sigma_uwb_factor = None
        self.resample_factor = None

        # uncertainties:
        self.sigma_dv = None
        self.sigma_dw = None
        self.sigma_uwb = None

        # Folder variables:
        self.trajectory_folder = trajectory_folder
        self.result_folder = result_folder
        self.result_name = ""
        self.result_file = None

        # Plot and debug variables:
        self.plot_bool = False
        self.debug_bool = False
        self.debug_plot_bool = False


        # Save for analyis:
        self.save_bool = False
        self.save_folder = None

        # Sims
        self.sims = 0
        self.total_sims = 0
        self.current_sim_name = ""

    def set_uncertainties(self, sigma_dv=1., sigma_dw=1., sigma_uwb=0.1):
        self.sigma_dv = sigma_dv
        self.sigma_dw = sigma_dw
        self.sigma_uwb = sigma_uwb

    def set_ukf_properties(self, kappa=-1, alpha=1, beta=2, n_azimuth=4, n_altitude=3, n_heading=4, resample_factor=0.1,
                           sigma_uwb_factor=1.0):
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.n_azimuth = n_azimuth
        self.n_altitude = n_altitude
        self.n_heading = n_heading
        self.resample_factor = resample_factor
        self.sigma_uwb_factor = sigma_uwb_factor

    def generate_results_file(self, redo_bool):
        if self.result_folder is None:
            self.result_folder = "./Results"
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)

        i = 0
        exists = True
        name = self.prefix + self.test_name + "|" + self.current_sim_name + "|s_uwb=" + str(
            self.sigma_uwb) + "|s_dv=" + str(self.sigma_dv) + "|s_dw=" + str(self.sigma_dw)
        name = "c".join(name.split("."))
        name = "n".join(name.split("-"))
        self.result_file = os.path.join(self.result_folder, name + ".pkl")
        if not os.path.isfile(self.result_file) or redo_bool:
            self.data = {}
            self.data[self.current_sim_name] = {}
            self.data["parameters"] = self.parameters
            self.data["parameters"]["type"] = self.type
            with open(self.result_file, "wb") as f:
                pkl.dump(self.data, f)
            f.close()
            return True
        else:
            try:
                print("Result file exists:", self.result_file)
                with open(self.result_file, "rb") as f:
                    self.data = pkl.load(f)
                f.close()
                self.data = {}
            except EOFError:
                print("Error opening:", self.result_file)
                self.data = {}
                self.data[self.current_sim_name] = {}
                self.data["parameters"] = self.parameters
                return True
        return False
        # while exists:
        #     file_name = name + str(i) + ".pkl"
        #     file = os.path.join(sim_folder, file_name)
        #     if os.path.exists(file):
        #         i += 1
        #     else:
        #         self.result_file = file
        #         exists = False

    @DeprecationWarning
    def get_results_file(self):
        if self.result_file is None:
            # result_name = "number_of_agents_" + str(self.number_of_agents)
            result_name = "_sigma_dv_" + str(np.round(self.sigma_dv, 3)) + "_sigma_dw_" + str(
                np.round(self.sigma_dw, 3)) \
                          + "_sigma_uwb_" + str(np.round(self.sigma_uwb, 3))
            result_name += "_freq_" + str(self.uwb_rate)
            # result_name += "_alpha_" + str(self.alpha) + "_kappa_" + str(self.kappa) + "_beta_" + str(self.beta)
            result_name = "c".join(result_name.split("."))
            result_name = "neg".join(result_name.split("-"))
            self.result_name = result_name
            self.result_file = os.path.join(self.result_folder, result_name + ".pkl")

    @DeprecationWarning
    def get_data(self):
        if self.result_file is None:
            self.get_results_file()
        print("Getting data from file: ", self.result_file)
        if self.data is None:
            if os.path.isfile(self.result_file):
                with open(self.result_file, "rb") as f:
                    try:
                        self.data = pkl.load(f)
                    except EOFError:
                        print(self.result_file)
                f.close()
            else:
                self.data = {}

            if not "parameters" in self.data:
                self.data["parameters"] = {"sigma_dv": self.sigma_dv, "sigma_dw": self.sigma_dw,
                                           "sigma_uwb": self.sigma_uwb, "uwb_rate": self.uwb_rate,
                                           "alpha": self.alpha, "kappa": self.kappa, "beta": self.beta,
                                           "number_of_agents": self.number_of_agents}

    @DeprecationWarning
    def get_data_from_file(self, file):
        self.result_file = file
        self.get_data()
        return self.data

    def get_sim(self, sim):
        sim_folder = self.trajectory_folder + "/" + sim
        self.sim = MultiRobotSingleSimulation(sim_folder)
        self.sim.run_simulation(self.sigma_dv, self.sigma_dw, self.sigma_uwb)
        # self.factor = int(1.0/self.frequency / self.sim.parameters["simulation_time_step"])
        if self.plot_bool:
            self.sim.init_plot(interactive=True)

    #-----------------------------
    # ---- Experiments Functions
    #-----------------------------

    def run_experiment(self, methods=[], redo_bool=False, experiment_data=None, prefix="exp_", res_type="experiment"):
        self.type = res_type
        self.prefix = prefix
        if experiment_data is None:
            return
        # check if experiment_data is a list
        if type(experiment_data) is not list:
            experiment_data = [experiment_data]

        for exp_data in experiment_data:
            if self.save_bool and not os.path.exists(self.save_folder + "/" + exp_data["name"]):
                os.mkdir(self.save_folder + "/" + exp_data["name"])
            self.experiment_data = exp_data
            self.current_sim_name = self.experiment_data["name"]
            # self.get_data()
            # if self.current_sim_name not in self.data:
            #     self.data[self.current_sim_name] = {}
            for method in methods:
                print(datetime.now(), " ", method + " of ", methods)
                self.reset_agents_w_exp_data()
                self.parse_test_name(method)
                if self.generate_results_file(redo_bool):
                    self.run_exp(method)
                    if os.path.isfile(self.result_file):
                        os.remove(self.result_file)
                    with open(self.result_file, "wb") as f:
                        pkl.dump(self.data, f)
            print("-----------------------------------")

    def reset_agents_w_exp_data(self):
        self.parameters = {}
        self.agents = {}
        for drone_name in self.experiment_data["drones"]:
            T_vicon = self.experiment_data["drones"][drone_name]["T_real"]
            DT_vio = self.experiment_data["drones"][drone_name]["DT_slam"]
            Q_vio = self.experiment_data["drones"][drone_name]["Q_slam"]

            drone = NewRobot()
            drone.from_experimental_data(T_vicon, DT_vio, Q_vio, self.experiment_data["sample_freq"])
            self.agents[drone_name] = {"drone": drone}
        self.d0 = self.experiment_data["uwb"][0]

    def run_exp(self, test_name):
        self.los_state = []
        self.uwb_error =[]
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]

        # drone0.form_experimental_data(self.experiment_data["drone_0"], self.experiment_data["Q_vio"])
        # drone1.set_vio_slam(self.experiment_data["drone_1"], self.experiment_data["Q_vio"])
        distances = self.experiment_data["uwb"]
        exp_len = len(self.experiment_data["uwb"])
        self.factor = int(self.experiment_data["sample_freq"] / self.frequency)
        for i in range(0, exp_len - 1):
            if self.debug_bool:
                print(datetime.now(), " Experiment step: ", i, " /", exp_len)
            if self.plot_bool:
                pass
                # self.sim.plot_trajectories_evolution(i, 50)

            # Integrate the odometry:
            for drone in self.agents:
                self.agents[drone]["drone"].integrate_odometry(i)

            if i % self.factor == 0:
                dx_0, q_0 = drone0.reset_integration()
                dx_1, q_1 = drone1.reset_integration()

                uwb_measurement = distances[i]
                self.los_state.append(self.experiment_data["los_state"][i])
                self.uwb_error.append(self.experiment_data["uwb_error"][i])

                eval("self.run_" + self.method + "_simulation" + "(dx_0, q_0, dx_1, q_1, uwb_measurement, i)")

        if self.plot_bool:
            plt.close()
        eval("self.end_" + self.method + "_test()")

    # ---- Test functions.
    def run_simulations(self, methods=[], nlos_function=None, redo_bool=False, sim_list=None, prefix="sim_",
                        res_type="simulation"):
        self.prefix = prefix
        self.type = res_type
        # if nlos_function == None:
        #     nlos_function = self.nlos_man.los

        self.total_sims = len(os.listdir(self.trajectory_folder))
        self.sims = 1
        for sim in os.listdir(self.trajectory_folder):
            if sim_list is None or sim in sim_list:
                self.current_sim_name = sim
                self.get_sim(self.current_sim_name)
                print(datetime.now(), " ", self.sims, " of ", self.total_sims, " simulations.")

                if self.save_bool and not os.path.exists(self.save_folder + "/" + sim):
                    os.mkdir(self.save_folder + "/" + sim)

                for method in methods:
                    print(datetime.now(), " ", method + " of ", methods)
                    self.reset_agents()
                    self.parse_test_name(method)
                    if self.generate_results_file(redo_bool):
                        self.run_simulation(method, nlos_function)
                        if os.path.isfile(self.result_file):
                            os.remove(self.result_file)
                        with open(self.result_file, "wb") as f:
                            pkl.dump(self.data, f)
                        f.close()

                print("-----------------------------------")
            self.sims += 1

    def reset_agents(self):
        self.parameters = {}
        self.agents = {}
        for drone_name in self.sim.drones:
            self.agents[drone_name] = {"drone": self.sim.drones[drone_name]}

    def generate_general_parameters(self, parameters):
        for par in ["sigma_dv", "sigma_dw", "sigma_uwb", "frequency", "horizon"]:
            if par in parameters:
                setattr(self, par, parameters[par])
            else:
                parameters[par] = getattr(self, par)
        # #
        # if "sigma_uwb" in parameters:
        #     self.sigma_uwb = parameters["sigma_uwb"]
        # else:
        #     parameters["sigma_uwb"] = self.sigma_uwb
        # if "frequency" in parameters:
        #     self.frequency = parameters["frequency"]
        # else:
        #     parameters["frequency"] = self.frequency
        # if "horizon" in parameters:
        #     self.horizon = parameters["horizon"]
        # else:
        #     parameters["horizon"] = self.horizon
        # if "sigma_dv" in parameters:
        #     self.sigma_dv = parameters["sigma_dv"]
        # else:
        #     parameters["sigma_dv"] = self.sigma_dv
        # if "sigma_dw" in parameters:
        #     self.sigma_dw = parameters["sigma_dw"]
        # else:
        #     parameters["sigma_dw"] = self.sigma_dw
        return parameters

    def parse_test_name(self, test_name):
        self.test_name = test_name
        parsing_test_name = test_name.split("|")
        parameters = {}
        for i in range(1, len(parsing_test_name)):
            parameter = parsing_test_name[i].split("=")
            parameters[parameter[0]] = float(parameter[1])
        self.parameters = self.generate_general_parameters(parameters)
        eval("self.init_" + parsing_test_name[0] + "_test(self.parameters)")

    def run_simulation(self, test_name, nlos_function=None):
        self.d0 = self.sim.get_uwb_measurements("drone_0", "drone_1")[0]
        self.parse_test_name(test_name)

        self.los_state = []
        self.uwb_error =[]
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]
        distances = self.sim.get_uwb_measurements("drone_0", "drone_1")
        self.factor = int(1.0 / self.frequency / self.sim.parameters["simulation_time_step"])
        for i in range(1, self.sim.parameters["simulation_time_steps"]):
            if self.plot_bool:
                self.sim.plot_trajectories_evolution(i, 50)

            # Integrate the odometry:
            for drone in self.agents:
                self.agents[drone]["drone"].integrate_odometry(i)

            if i % self.factor == 0:
                if self.debug_bool:
                    print(datetime.now(), self.sims, "/", self.total_sims, self.test_name, " Simulation step: ", i,
                          " /", self.sim.parameters["simulation_time_steps"])
                dx_0, q_0 = drone0.reset_integration()
                dx_1, q_1 = drone1.reset_integration()

                uwb_measurement = distances[i]
                los_state = 1
                # uwb_measurement, los_state = nlos_function(int(i / self.factor), uwb_measurement)
                self.los_state.append(los_state)

                eval("self.run_" + self.method + "_simulation" + "(dx_0, q_0, dx_1, q_1, uwb_measurement, i)")

        if self.plot_bool:
            plt.close()
        eval("self.end_" + self.method + "_test()")

    # --- UPF functions

    def init_upf_test(self, parameters={}):
        self.init_gen_upf_test(NLOS_bool=True, drift_bool=True, parameters=parameters)

    def init_losupf_test(self, parameters={}):
        self.init_gen_upf_test(NLOS_bool=False, drift_bool=True, parameters=parameters)

    def init_nodriftupf_test(self, parameters={}):
        self.init_gen_upf_test(NLOS_bool=False, drift_bool=False, parameters=parameters)

    def init_upfnaive_test(self, parameters={}):
        self.init_gen_upf_test(NLOS_bool=True, drift_bool=True, naive_sampling_bool=True, parameters=parameters)

    def generate_upf_parameters(self):
        # for par in ["alpha", "kappa", "beta", "n_azimuth", "n_heading", "n_altitude", "resample_factor",  "sigma_uwb_factor"]:
        #     if par in self.parameters:
        #         eval("self."+par + "=" + str(self.parameters[par]))
        #     else:
        #         eval("self.parameters['"+par+"'] = self."+par)

        if "alpha" not in self.parameters:
            self.parameters["alpha"] = self.alpha
        else:
            self.alpha = self.parameters["alpha"]
        if "kappa" not in self.parameters:
            self.parameters["kappa"] = self.kappa
        else:
            self.kappa = self.parameters["kappa"]
        if "beta" not in self.parameters:
            self.parameters["beta"] = self.beta
        else:
            self.beta = self.parameters["beta"]
        if "n_azimuth" not in self.parameters:
            self.parameters["n_azimuth"] = self.n_azimuth
        else:
            self.n_azimuth = self.parameters["n_azimuth"]
        if "n_heading" not in self.parameters:
            self.parameters["n_heading"] = self.n_heading
        else:
            self.n_heading = self.parameters["n_heading"]
        if "n_altitude" not in self.parameters:
            self.parameters["n_altitude"] = self.n_altitude
        else:
            self.n_altitude = self.parameters["n_altitude"]
        if "resample_factor" not in self.parameters:
            self.parameters["resample_factor"] = self.resample_factor
        else:
            self.resample_factor = self.parameters["resample_factor"]
        if "sigma_uwb_factor" not in self.parameters:
            self.parameters["sigma_uwb_factor"] = self.sigma_uwb_factor
        else:
            self.sigma_uwb_factor = self.parameters["sigma_uwb_factor"]

    def init_gen_upf_test(self, NLOS_bool=True, drift_bool=True, naive_sampling_bool=False, logger=True, parameters={}):
        self.generate_upf_parameters()
        self.method = "upf"
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]
        # init_distance = self.sim.get_uwb_measurements("drone_0", "drone_1")[0]
        # self.d0 = init_distance
        # init drone0
        x_ha_0 = np.concatenate((drone0.x_start, np.array([drone0.h_start])))

        lop = ListOfUKFLOSTargetTrackingParticles(drift_correction_bool=drift_bool)
        lop.set_ukf_parameters(kappa=self.kappa, alpha=self.alpha, beta=self.beta)
        if "multi_particles" in parameters and parameters["multi_particles"] ==0 :
            t_S0_S1 = self.find_initial_t(drone0, drone1)
            s_S0_S1 = np.concatenate((cartesianToSpherical(t_S0_S1[:3]), np.array([t_S0_S1[-1]])))
            particle = lop.create_particle(x_ha_0, s_S0_S1, np.array([self.sigma_uwb, 0.000001,0.000001,0.000001]), s_S0_S1[0],self.sigma_uwb)
            lop.particles.append(particle)
        else:
            lop.split_sphere_in_equal_areas(t_i= x_ha_0, d_ij= self.d0, sigma_uwb = self.sigma_uwb,
                                        n_azimuth=self.n_azimuth, n_altitude=self.n_altitude, n_heading=self.n_heading)

        upf0 = UPFConnectedAgent(lop.particles, x_ha_0=x_ha_0, drift_correction_bool=drift_bool,
                                 sigma_uwb=self.sigma_uwb, id="drone_1")
        upf0.resample_factor = self.resample_factor
        upf0.sigma_uwb_factor = self.sigma_uwb_factor
        # upf0.set_ukf_parameters(kappa=self.kappa, alpha=self.alpha, beta=self.beta)
        # upf0.split_sphere_in_equal_areas(r=self.d0, sigma_uwb=self.sigma_uwb,
        #                                  n_azimuth=self.n_azimuth, n_altitude=self.n_altitude, n_heading=self.n_heading)

        dl0 = UPFConnectedAgentDataLogger(drone0, drone1, upf0, UKFTargetTrackingParticle_DataLogger)
        dl0.log_data(0)

        # upf0.set_logging(dl0)
        self.agents["drone_0"][self.test_name] = upf0
        self.agents["drone_0"]["log"] = dl0

        # init drone1
        x_ha_1 = np.concatenate((drone1.x_start, np.array([drone1.h_start])))

        lop = ListOfUKFLOSTargetTrackingParticles(drift_correction_bool=drift_bool)
        lop.set_ukf_parameters(kappa=self.kappa, alpha=self.alpha, beta=self.beta)
        if "multi_particles" in parameters and parameters["multi_particles"] ==0 :
            t_S1_S0 = get_states_of_transform(inv_transformation_matrix(t_S0_S1))
            s_S1_S0 = np.concatenate((cartesianToSpherical(t_S1_S0[:3]), np.array([t_S1_S0[-1]])))
            particle = lop.create_particle(x_ha_1, s_S1_S0, np.array([self.sigma_uwb, 0.00001,0.00001,0.000001 ]), s_S1_S0[0],self.sigma_uwb)
            lop.particles.append(particle)
        else:
            lop.split_sphere_in_equal_areas(t_i= x_ha_1, d_ij= self.d0, sigma_uwb = self.sigma_uwb,
                                        n_azimuth=self.n_azimuth, n_altitude=self.n_altitude, n_heading=self.n_heading)

        upf1 = UPFConnectedAgent(lop.particles, x_ha_0=x_ha_1, drift_correction_bool=drift_bool,
                                 sigma_uwb=self.sigma_uwb, id="drone_0")
        upf1.resample_factor = self.resample_factor
        upf1.sigma_uwb_factor = self.sigma_uwb_factor

        dl1 = UPFConnectedAgentDataLogger(drone1, drone0, upf1, UKFTargetTrackingParticle_DataLogger)
        dl1.log_data(0)

        # upf1.set_logging(dl1)
        self.agents["drone_1"][self.test_name] = upf1
        self.agents["drone_1"]["log"] = dl1

    def run_upf_simulation(self, dx_0, q_0, dx_1, q_1, uwb_measurement, i):
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]

        upf0: UPFConnectedAgent = self.agents["drone_0"][self.test_name]
        upf1: UPFConnectedAgent = self.agents["drone_1"][self.test_name]
        upf0log: UPFConnectedAgentDataLogger = self.agents["drone_0"]["log"]
        upf1log: UPFConnectedAgentDataLogger = self.agents["drone_1"]["log"]

        # # Not sure if this is needed.
        # upf0.ha.predict(dx_ha=dx_0, Q_ha=q_0)
        # upf1.ha.predict(dx_ha=dx_1, Q_ha=q_1)
        # dx_0, q_0 = upf0.ha.reset_integration()
        # dx_1, q_0 = upf1.ha.reset_integration()

        # Drone 0
        x_ha = drone0.x_slam[i]
        h_ha = drone0.h_slam[i]
        x_ha_0 = np.concatenate([x_ha, np.array([h_ha])])
        upf0.ha.update(x_ha_0, q_0)
        # Timing the execution of the algorihtm
        t1 = time.time()
        upf0.run_model(dt_j = dx_1, q_j=q_1, dt_i = dx_0, q_i = q_0, d_ij = uwb_measurement, time_i=i)
        t2 = time.time()
        # Logging
        upf0log.log_data(i, t2 - t1)

        # Drone 1
        x_ha = drone1.x_slam[i]
        h_ha = drone1.h_slam[i]
        x_ha_1 = np.concatenate([x_ha, np.array([h_ha])])
        upf1.ha.update(x_ha_1, q_1)
        # Timing the execution of the algorihtm
        t3 = time.time()
        upf1.run_model(dt_j = dx_0, q_j=q_0, dt_i =  dx_1, q_i = q_1, d_ij= uwb_measurement, time_i=i)
        t4 = time.time()
        # Logging
        upf1log.log_data(i, t4 - t3)

        if self.debug_bool:
            print("UPF time 1: ", t4 - t3)
            print("UPF time 0: ", t2 - t1)
        # try:
        if self.plot_bool and self.debug_plot_bool:
            for agent in self.agents:
                try:
                    # self.agents[agent]["upf"].upf_connected_agent_logger.plot_self(self.los_state)
                    self.agents[agent]["log"].plot_self(self.los_state)
                    plt.pause(2)
                    plt.close()
                except KeyError:
                    print("keyerror")
            plt.show()
        # except Exception as e:
        #     print(e)

    def end_upf_test(self):
        self.data[self.current_sim_name][self.test_name] = {}
        for drone_id in self.agents:
            # UPF Results
            upf: UPFConnectedAgent = self.agents[drone_id][self.test_name]
            dl_ca: UPFConnectedAgentDataLogger = self.agents[drone_id]["log"]
            # dl_ca: UPFConnectedAgentDataLogger = upf.upf_connected_agent_logger
            dl_bp: TargetTrackingParticle_DataLogger = dl_ca.get_best_particle_log()
            dl_bp_rpea: UKFDatalogger = dl_bp.rpea_datalogger

            upf_result = {"number_of_particles": dl_ca.number_of_particles,
                          "calculation_time": dl_ca.calulation_time,
                          # "los_state": dl_bp.los_state,
                          # "los_error": np.abs(np.array(self.los_state) - np.array(dl_bp.los_state)),
                          "error_x_relative": dl_bp_rpea.error_relative_transformation_est,
                          "error_h_relative": dl_bp_rpea.error_relative_heading_est,
                          "sigma_h_relative": dl_bp_rpea.sigma_h_ca,
                          "sigma_x_relative": dl_bp_rpea.sigma_x_ca,
                          "LOS_state" : dl_bp.los_state,
                          "d_error": self.uwb_error,
                          "True_los_state": self.los_state, # Can be removed, depends on sigma_uwb
                          "NIS": dl_bp_rpea.NIS}
            self.data[self.current_sim_name][self.test_name][drone_id] = upf_result
            print(upf_result["d_error"])
            if self.save_bool:
                with open(
                        self.save_folder + "/" + self.current_sim_name + "/" + drone_id + "_" + self.test_name + ".pkl",
                        'wb') as file:
                    pkl.dump(dl_ca, file)

        # if "slam" not in self.data[self.current_sim_name]:aw
        self.data[self.current_sim_name]["slam"] = {}
        for drone_id in self.agents:
            dl_bp_rpea: UKFDatalogger = self.agents[drone_id]["log"].get_best_particle_log().rpea_datalogger
            slam_result = {"error_x_relative": dl_bp_rpea.error_relative_transformation_slam,
                           "error_h_relative": dl_bp_rpea.error_relative_heading_slam}
            self.data[self.current_sim_name]["slam"][drone_id] = slam_result

            # A new file will be created
            # self.data[self.current_sim_name]["slam"][drone_id]["error_x_relative"] = dl_bp.error_relative_transformation_est
            #     self.data[self.current_sim_name]["WLS"] = {}
        #     print(upf_result["los_error"])
        # print(self.los_state)
        if os.path.isfile(self.result_file):
            os.remove(self.result_file)
        with open(self.result_file, "wb") as f:
            pkl.dump(self.data, f)

        try:
            if self.plot_bool:
                for agent in self.agents:
                    self.agents[agent]["log"].plot_self(self.los_state)
                    # plt.show()
                    # plt.pause(0.1)
                    # plt.close()
        except Exception as e:
            print(e)

    # ----------------------------
    # ---- Benchmark test_na_5_na_8_nh_8 functions ----
    # ----------------------------

    # ---- Algebraic functions ----
    def init_algebraic_test(self, parameters={}):
        self.method = "algebraic"
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]
        start_distance = self.d0
        # distances = self.sim.get_uwb_measurements("drone_0", "drone_1")
        x_ha = np.concatenate((drone0.x_start, [drone0.h_start]))
        AM0 = AlgebraicMethod4DoF(start_distance, sigma_uwb=self.sigma_uwb, x_ha=x_ha)
        alg_log = Algebraic4DoF_Logger(alg_solver=AM0, host=drone0, connect=drone1)
        # alg_log.
        AM0.init_logging(alg_log)
        if "horizon" in parameters:
            AM0.horizon = int(parameters["horizon"])
        self.agents["drone_0"][self.test_name] = AM0

        x_ha = np.concatenate((drone1.x_start, [drone1.h_start]))
        AM1 = AlgebraicMethod4DoF(start_distance, sigma_uwb=self.sigma_uwb, x_ha=x_ha)
        alg_log = Algebraic4DoF_Logger(alg_solver=AM1, host=drone1, connect=drone0)
        # alg_log.
        AM1.init_logging(alg_log)
        if "horizon" in parameters:
            AM1.horizon = int(parameters["horizon"])
        self.agents["drone_1"][self.test_name] = AM1

    def run_algebraic_simulation(self, dx_0, q_0, dx_1, q_1, uwb_measurement, i):
        # to make it fair the algebraic method will use the latest 10 measurement.
        # Alternative use the 10 measurements eqaully spaced in time. ?

        AM0: AlgebraicMethod4DoF = self.agents["drone_0"][self.test_name]
        AM1: AlgebraicMethod4DoF = self.agents["drone_1"][self.test_name]

        t1 = time.time()
        AM0.get_update(uwb_measurement, dx_0, dx_1, q_ha=q_0, q_ca=q_1)
        t2 = time.time()
        # AM0.WLS_on_position()
        t3 = time.time()
        AM0.logger.log(i, t2 - t1, t3 - t1)

        t4 = time.time()
        AM1.get_update(uwb_measurement, dx_1, dx_0, q_ha=q_1, q_ca=q_0)
        t5 = time.time()
        # AM1.WLS_on_position()
        t6 = time.time()
        AM1.logger.log(i, t5 - t4, t6 - t4)

        if self.debug_bool:
            print("ALG time 0: ", t2 - t1, " WLS: ", t3 - t1)
            print("ALG time 1: ", t5 - t4, " WLS: ", t6 - t4)

    def end_algebraic_test(self):
        self.data[self.current_sim_name][self.test_name] = {}
        # self.data[self.current_sim_name][self.test_name+"_WLS"] = {}
        for drone_id in self.agents:
            am_log: Algebraic4DoF_Logger = self.agents[drone_id][self.test_name].logger
            alg_results = {"calculation_time": am_log.calculation_time_alg,
                           "error_x_relative": am_log.x_ca_r_alg_error,
                           "error_h_relative": am_log.x_ca_r_alg_heading_error, }
            self.data[self.current_sim_name][self.test_name][drone_id] = alg_results

            # wls_results = {"calculation_time": am_log.calculation_time_wls,
            #                "error_x_relative": am_log.x_ca_r_WLS_error,
            #                "error_h_relative": am_log.x_ca_r_WLS_heading_error}
            # self.data[self.current_sim_name][self.test_name+"_WLS"][drone_id] = wls_results

            if self.save_bool:
                with open(
                        self.save_folder + "/" + self.current_sim_name + "/" + drone_id + "_" + self.test_name + ".pkl",
                        'wb') as file:
                    pkl.dump(self.agents[drone_id][self.test_name], file)

    # ---- NLS test_na_5_na_8_nh_8 functions ----
    def init_NLS_test(self, parameters={}):
        self.method = "NLS"
        agents = {"drone_0": self.agents["drone_0"]["drone"], "drone_1": self.agents["drone_1"]["drone"]}
        self.agents["drone_0"][self.test_name] = NLS(agents, horizon=int(self.horizon), sigma_uwb=self.sigma_uwb)
        if "perfect_guess" in parameters:
            if parameters["perfect_guess"] == 0:
                best_guess = self.find_best_initial_guess()
                self.agents["drone_0"][self.test_name].set_best_guess({"drone_1": best_guess})
            # self.agents["drone_0"][self.test_name].set_best_guess(parameters["perfect_guess"])
        #
        # initial_t = self.find_initial_t()
        # self.agents["drone_0"][self.test_name].set_best_guess({"drone_1": initial_t})
        self.agents["drone_0"][self.test_name].init_logging(NLSDataLogger(self.agents["drone_0"][self.test_name]))

    def init_NLS_p_test(self):
        self.method = "NLS"
        agents = {"drone_0": self.agents["drone_0"]["drone"], "drone_1": self.agents["drone_1"]["drone"]}
        self.agents["drone_0"][self.test_name] = NLS(agents, horizon=int(self.horizon), sigma_uwb=self.sigma_uwb)
        best_guess = self.find_best_initial_guess()
        #initial_t = self.find_initial_t()
        # self.agents["drone_0"]["NLS"].set_best_guess({"drone_1": best_guess})
        self.agents["drone_0"][self.test_name].init_logging(NLSDataLogger(self.agents["drone_0"][self.test_name]))

    def find_initial_t(self, drone0, drone1):
        # drone0: NewRobot = self.agents["drone_0"]["drone"]
        # drone1: NewRobot = self.agents["drone_1"]["drone"]
        t_O_S0 = np.concatenate((drone0.x_start, np.array([drone0.h_start])))
        t_O_S1 = np.concatenate((drone1.x_start, np.array([drone1.h_start])))

        T_S0_O = inv_transformation_matrix(t_O_S0)
        T_O_S1 = transform_matrix(t_O_S1)

        T_S0_S1 = T_S0_O @ T_O_S1
        t_S0_S1 = get_states_of_transform(T_S0_S1)
        return t_S0_S1

    def find_best_initial_guess(self):
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        # start_distance = self.sim.get_uwb_measurements("drone_0", "drone_1")[0]
        start_distance = self.d0
        distances = 1e6
        dis_angle = np.pi
        x_start = self.agents["drone_1"]["drone"].x_start
        h_start = self.agents["drone_1"]["drone"].h_start
        x_start_guess = np.zeros(3)
        h_start_guess = 0

        x_ha_0 = np.concatenate((drone0.x_start, np.array([drone0.h_start])))
        upf0 = UPFConnectedAgent([], x_ha_0=x_ha_0, id="drone_0")
        upf0.set_ukf_parameters(kappa=self.kappa, alpha=self.alpha, beta=self.beta)
        upf0.split_sphere_in_equal_areas(r=start_distance, sigma_uwb=self.sigma_uwb,
                                         n_azimuth=self.n_azimuth, n_altitude=self.n_altitude, n_heading=self.n_heading)

        for particle in upf0.particles:
            dis = np.linalg.norm(particle.t_oi_sj[:3] - x_start)
            dis_h = np.abs(limit_angle(particle.t_oi_sj[-1] - h_start))
            if dis <= distances and dis_h <= dis_angle:
                distances = dis
                x_start_guess = particle.t_oi_sj[:3]
                dis_angle = dis_h
                h_start_guess = particle.t_oi_sj[-1]
        if self.debug_bool:
            print(x_start_guess, h_start_guess)
            print(x_start, h_start)

        return np.concatenate([x_start_guess, np.array([h_start_guess])])

    def run_NLS_simulation(self, dx_0, q_0, dx_1, q_1, uwb_measurement, i):
        dx = np.vstack([dx_0.reshape(1, *dx_0.shape), dx_1.reshape(1, *dx_1.shape)])
        q_odom = np.vstack([q_0.reshape(1, *q_0.shape), q_1.reshape(1, *q_1.shape)])
        d = np.array([[0, uwb_measurement], [0, 0]])
        t1 = time.time()
        self.agents["drone_0"][self.test_name].update(d, dx, q_odom)
        t2 = time.time()
        self.agents["drone_0"][self.test_name].nls_logger.log_data(i, t2 - t1)

        if self.debug_bool:
            print("NLS time 0 + 1: ", t2 - t1)

    def end_NLS_test(self):
        self.data[self.current_sim_name][self.test_name] = {}
        for drone_id in self.agents:
            self.data[self.current_sim_name][self.test_name][drone_id] = \
                self.agents["drone_0"][self.test_name].nls_logger.results[drone_id]["NLS"]
        if os.path.isfile(self.result_file):
            os.remove(self.result_file)
        with open(self.result_file, "wb") as f:
            pkl.dump(self.data, f)
        if self.save_bool:
            with open(
                    self.save_folder + "/" + self.current_sim_name + "/" + self.test_name + ".pkl",
                    'wb') as file:
                pkl.dump(self.agents["drone_0"][self.test_name], file)
            file.close()

    # ---- QCQP ----
    def init_QCQP_test(self, parameters={}):
        self.method = "QCQP"

        # agents = {"drone_0": self.agents["drone_0"]["drone"], "drone_1": self.agents["drone_1"]["drone"]}
        # ---- Drone 0
        self.agents["drone_0"][self.test_name] = QCQP(horizon=int(self.horizon), sigma_uwb=self.sigma_uwb)
        self.agents["drone_0"][self.test_name + "_log"] = QCQP_Log(self.agents["drone_0"][self.test_name],
                                                                   self.agents["drone_0"]["drone"],
                                                                   self.agents["drone_1"]["drone"])
        # self.agents["drone_0"][self.test_name].horizon = int(self.horizon)
        # ---- Drone 1
        self.agents["drone_1"][self.test_name] = QCQP(horizon=int(self.horizon), sigma_uwb=self.sigma_uwb)
        self.agents["drone_1"][self.test_name + "_log"] = QCQP_Log(self.agents["drone_1"][self.test_name],
                                                                   self.agents["drone_1"]["drone"],
                                                                   self.agents["drone_0"]["drone"])
        # self.agents["drone_1"][self.test_name].horizon = int(self.horizon)

    def run_QCQP_simulation(self, dx_0, q_0, dx_1, q_1, uwb_measurement, i):

        t1 = time.time()
        self.agents["drone_0"][self.test_name].update(dx_0, dx_1, uwb_measurement)
        t2 = time.time()
        self.agents["drone_0"][self.test_name + "_log"].log(i, t2 - t1)
        t3 = time.time()
        self.agents["drone_1"][self.test_name].update(dx_1, dx_0, uwb_measurement)
        t4 = time.time()
        self.agents["drone_1"][self.test_name + "_log"].log(i, t4 - t3)

        if self.debug_bool:
            print("QCQP time 0 + 1: ", t2 - t1 + t4 - t3)

    def end_QCQP_test(self):
        self.data[self.current_sim_name][self.test_name] = {}
        for drone_id in self.agents:
            self.data[self.current_sim_name][self.test_name][drone_id] = \
                self.agents[drone_id][self.test_name + "_log"].results["QCQP"]
        if os.path.isfile(self.result_file):
            os.remove(self.result_file)
        with open(self.result_file, "wb") as f:
            pkl.dump(self.data, f)
            if self.save_bool:
                with open(
                        self.save_folder + "/" + self.current_sim_name + "/" + drone_id + "_" + self.test_name + ".pkl",
                        'wb') as file:
                    qcqp_log = self.agents[drone_id][self.test_name + "_log"]
                    qcqp_log.qcqp = None
                    pkl.dump(qcqp_log, file)
                file.close()


class EvaluationMethod():
    def __init__(self, test_name, parameters={}):
        self.test_name = test_name
        self.parameters = parameters

        # def generate_general_parameters(self, parameters):
        for par in self.parameters:
            if par in parameters:
                setattr(self, par, parameters[par])
            else:
                parameters[par] = getattr(self, par)
        self.agents = {}

    def run_evaluation(self, dx_0, q_0, dx_1, q_1, uwb_measurement, i):
       pass

    def end_evaluation(self):
        pass


class QCPQMethod(EvaluationMethod):
    def __init__(self, parameters={}):
        self.horizon = 100
        super().__init__("QCQP", parameters)
        # ---- Drone 0
        self.agents["drone_0"][self.test_name] = QCQP(horizon=int(self.horizon), sigma_uwb=self.sigma_uwb)
        self.agents["drone_0"][self.test_name + "_log"] = QCQP_Log(self.agents["drone_0"][self.test_name],
                                                                   self.agents["drone_0"]["drone"],
                                                                   self.agents["drone_1"]["drone"])

        # ---- Drone 1
        self.agents["drone_1"][self.test_name] = QCQP(horizon=int(self.horizon), sigma_uwb=self.sigma_uwb)
        self.agents["drone_1"][self.test_name + "_log"] = QCQP_Log(self.agents["drone_1"][self.test_name],
                                                                   self.agents["drone_1"]["drone"],
                                                                   self.agents["drone_0"]["drone"])
    def run_evaluation(self, dx_0, q_0, dx_1, q_1, uwb_measurement, i):
        pass
        # self.method = "QCQP"

        # agents = {"drone_0": self.agents["drone_0"]["drone"], "drone_1": self.agents["drone_1"]["drone"]}
        # ---- Drone 0

        # self.agents["drone_1"][self.test_name].horizon = int(self.horizon)
