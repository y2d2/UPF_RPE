#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:24:51 2023

@author: yuri
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import scipy.spatial.distance
from filterpy.kalman import KalmanFilter

from matplotlib.gridspec import GridSpec

from Code.ParticleFilter.TargetTrackingUKF import TargetTrackingUKF
from Code.DataLoggers.TargetTrackingUKF_DataLogger import UKFDatalogger
from Code.Simulation.RobotClass import NewRobot
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix, cartesianToSpherical
from Code.ParticleFilter.TargetTrackingParticle import TargetTrackingParticle

class KFHostAgent:
    """
    TODO: Keep track of the growth in uncertainty from the previous update to now.
        The idea is that the uncertainty of the previous step is added to the uncertainty of the x_ca_odom,
        while the current uncertainty from the previous x_ha until now is added to the uncertainty of the measurement.
    """

    def __init__(self, x_ha_0=np.zeros(4)):
        self.x_ha_0 = x_ha_0
        self.x_ha = x_ha_0

        # Drift related variables.
        self.x_ha_vio = x_ha_0
        self.dx_ha_vio = np.zeros(4)
        # self.p_dx_ha_vio = np.zeros((4, 4))
        self.p_x_ha = np.zeros((4, 4))
        self.dx_drift = np.zeros(4)

        # Odom-communication related variables.
        self.dx_ha_odom = np.zeros(4)
        self.p_dx_ha = np.zeros((4, 4))

        # KF variables
        self.kf = KalmanFilter(dim_x=4, dim_z=4)
        self.kf.x = x_ha_0  # = x_ha_odom
        self.kf.F = np.eye(4)
        self.kf.H = np.eye(4)
        self.kf.P = np.zeros((4, 4))
        self.kf.R = np.zeros((4, 4))

    def set_x_ha(self, x_ha):
        " Only for when "
        self.x_ha = x_ha

    def predict(self, dx_ha, Q_ha):
        """
        Predict the evolution of the host agent without correction from other agents.
        """
        B = get_4d_rot_matrix(self.kf.x[3])
        self.kf.predict(u=dx_ha, B=B, Q=Q_ha)

        # for now, I will keep those to separate. This is the odom to communicate with the other agents.
        b = get_4d_rot_matrix(self.dx_ha_odom[3])
        self.dx_ha_odom = self.dx_ha_odom + b @ dx_ha
        self.p_dx_ha = self.p_dx_ha + b @ Q_ha @ b.T
        return None

    def update(self, x_ha, q_ha):
        """
        Update the host agent with the correction from the other agents.
        """
        self.dx_drift = x_ha - self.kf.x
        self.kf.R = q_ha
        self.kf.update(x_ha)

        self.x_ha = x_ha
        self.kf.x = x_ha
        self.p_x_ha = q_ha
        return None

    def get_x_ha_odom_sigma(self):
        # return np.linalg.norm(np.sqrt(np.diag(self.p_dx_ha_prev[:3, :3])))
        # return np.eye(4)*1e-6
        return self.p_x_ha.copy()

    def reset_integration(self, x_ha=np.zeros(4)):
        # Uncorrected movement. (seems not the best thing.)
        dx = copy.deepcopy(self.dx_ha_odom)
        q = copy.deepcopy(self.p_dx_ha)
        self.dx_ha_odom = np.zeros(4)
        self.p_dx_ha = np.zeros((4, 4))

        return dx, q


class UPFConnectedAgent:
    """
    Class that represents the individual particle filters per connected agent
    on the estimation of the trajectory of the connected agents.
    """

    def __init__(self, list_of_particles = [], x_ha_0=np.zeros(4), drift_correction_bool=True,
                 sigma_uwb = 0.1, sigma_uwb_factor=1.0, resample_factor=0.1, id="0x000"):

        self.id = id
        # State variables:
        self.best_particle = None

        # Internal variables:
        self.dh_ca = 0
        self.dx_ca = np.zeros(3)
        # self.iterations = 0

        # Particle Filter variables:
        self.particles: List[TargetTrackingParticle] = list_of_particles
        self.particle_type = TargetTrackingUKF
        self.weights = []
        self.totalWeight = 1.
        self.best_particle: TargetTrackingParticle | None = None

        # Host agent variables:
        self.ha = KFHostAgent(x_ha_0)
        self.x_ha = x_ha_0
        self.x_ha_prev = x_ha_0
        self.dh_ha = 0
        self.p_dx_ha = np.zeros((4, 4))

        # UWB Extrinsicty variables:
        self.t_si_uwb = np.zeros(4)
        self.t_sj_uwb = np.zeros(4)

        # Measurement variables:
        self.measurement = 0

        # Uncertainty variables:
        self.sigma_uwb = sigma_uwb
        self.sigma_x_ha = 0
        self.sigma_dh_ca = 0
        self.sigma_dx_ca = 0
        self.sigma_dx_ha = 0
        self.sigma_dh_ha = 0
        self.P_ha = 0

        # Area variables:
        self.n_altitude = 0
        self.n_azimuth = 0
        self.n_heading = 0

        # UKF Variables:
        self.kappa = 0
        self.alpha = 0
        self.beta = 0
        self.drift_correction_bool = drift_correction_bool
        self.set_ukf_parameters()

        # Regeneration variables:
        self.regeneration_bool = False
        self.max_number_of_particles = 0

        # Logging variables:
        # self.logging = False
        # self.upf_connected_agent_logger = None
        self.time_i = None

        self.resample = self.branch_kill_resampling
        self.resample_factor = resample_factor
        self.sigma_uwb_factor = sigma_uwb_factor

        self.max_dis = 0.1
        # def set_logging(self, ca_logger):

    #     self.logging = True
    #     self.upf_connected_agent_logger = ca_logger

    def set_uwb_extrinsicity(self, t_si_uwb, t_sj_uwb):
        self.t_si_uwb = t_si_uwb
        self.t_sj_uwb = t_sj_uwb

    # @deprecated()
    def set_ukf_parameters(self, kappa=-1, alpha=1, beta=2):
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        # self.drift_correction_bool = drift_correction_bool

    # @deprecated()
    def set_regeneration_parameters(self, max_number_of_particles=10, regeneration=True):
        self.regeneration_bool = regeneration
        self.max_number_of_particles = max_number_of_particles
        # self.regenaration_sigmas = regeneration_sigmas

    # @deprecated()
    def generate_new_particles(self):
        if self.regeneration_bool:
            while len(self.particles) < self.max_number_of_particles:
                self.generate_new_particle()

    # @deprecated()
    def generate_new_particle(self):
        azimuth = np.random.uniform(-np.pi, np.pi)
        altitude = np.random.uniform(-np.pi / 2, np.pi / 2)
        heading = np.random.uniform(-np.pi, np.pi)
        #TODO should make something that is parameterized.
        sigma_azimuth = (2 * np.pi / 8) / np.sqrt(-8 * np.log(0.5))
        sigma_altitude = (np.pi / 5) / np.sqrt(-8 * np.log(0.5))
        sigma_heading = (2 * np.pi / 8) / np.sqrt(-8 * np.log(0.5))
        particle = self.create_particle()
        particle.weight = 0.1
        s = np.array([self.uwb_measurement, azimuth, altitude], dtype=float)
        sigma_s = [2 * self.sigma_uwb, sigma_azimuth, sigma_altitude]
        particle.set_initial_state(s, sigma_s)
        self.particles.append(particle)

    # @deprecated()
    def create_particle(self):
        weight = 1. / self.n_azimuth / self.n_altitude / self.n_heading
        particle = self.particle_type(x_ha_0=self.ha.x_ha_0, weight=weight,
                                      drift_correction_bool=self.drift_correction_bool)
        particle.set_uwb_extrinsicity(self.t_si_uwb, self.t_sj_uwb)
        particle.set_ukf_properties(self.kappa, self.alpha, self.beta)
        return particle

    # @deprecated()
    def add_particle_with_know_start_pose(self, x_ca_0, azimuth_n, altitude_n, heading_n, sigma_uwb):
        self.n_altitude = 1
        self.n_azimuth = 1
        self.n_heading = 1
        self.sigma_uwb = self.sigma_uwb_factor * sigma_uwb

        sigma_azimuth = (2 * np.pi / azimuth_n) / np.sqrt(-8 * np.log(0.5))
        sigma_altitude = (np.pi / altitude_n) / np.sqrt(-8 * np.log(0.5))
        sigma_heading = (2 * np.pi / heading_n) / np.sqrt(-8 * np.log(0.5))
        sigma_s = [2 * sigma_uwb, sigma_azimuth, sigma_altitude]

        particle = self.create_particle()
        s = cartesianToSpherical(x_ca_0[:3]).tolist()
        # sigma_heading = np.sqrt(P_x_ca_0[3, 3])
        # d = np.linalg.norm(x_ca_0[:3])
        # sigma_x_ca = np.linalg.norm(np.sqrt(np.diag(P_x_ca_0[:3, :3])))
        # sigma_angle = sigma_x_ca / d
        # sigma_s = [sigma_x_ca, sigma_angle, sigma_angle]
        particle.set_initial_state(s, sigma_s)
        self.particles.append(particle)
        self.set_best_particle(self.particles[0])

    def create_single_particle(self, t, sigma_uwb):
        self.n_altitude = 1
        self.n_azimuth = 1
        self.n_heading = 1
        self.sigma_uwb = self.sigma_uwb_factor * sigma_uwb
        s_S0_S1 = cartesianToSpherical(t[:3])
        particle = self.create_particle()
        particle.set_initial_state(s_S0_S1, np.array([self.sigma_uwb, 0.000001, 0.000001]),  t[-1], 0.000001,
                                   self.sigma_uwb)
        self.particles.append(particle)
        self.set_best_particle(self.particles[0])

    # @deprecated()
    def split_sphere_in_equal_areas(self, r: float, sigma_uwb: float, n_altitude: int, n_azimuth: int, n_heading: int):
        """
        Function to split the area of a sphere in almost equal areas (= weights)
        Starting from the n_altitude that has to be uneven (will be made uneven).
        For each altitude
        """
        self.n_altitude = n_altitude
        self.n_azimuth = n_azimuth
        self.n_heading = n_heading
        self.sigma_uwb = self.sigma_uwb_factor * sigma_uwb

        if n_altitude % 2 == 0:
            n_altitude += 1
        # n_altitude = n_altitude+2
        altitude_delta = np.pi / (n_altitude)
        # altitudes = np.round([-np.pi / 2  + i * altitude_delta for i in range(1,n_altitude-1)], 4)
        altitudes = np.round([-np.pi / 2 + altitude_delta / 2 + i * altitude_delta for i in range(n_altitude)], 4)
        # calculate the areas to calculate the number of azimuth angles is needed.
        altitude_surfaces = [(np.sin(altitude + altitude_delta / 2) - np.sin(altitude - altitude_delta / 2)) / 2 for
                             altitude in altitudes]
        # print(altitude_surfaces)
        area = altitude_surfaces[int((len(altitude_surfaces) - 1) / 2)] / n_azimuth
        # print(area)
        # Calculate azimuth bins such that the area is per particle is equal.
        azimuth_bins = [math.ceil(altitude_surface / area) for altitude_surface in altitude_surfaces]
        # print(azimuth_bins)
        # Calculate the weights per particle, should  be almost equal to each other, though slight difference may occur due to round off.
        # weights = [altitude_surface / azimuthBin for altitude_surface, azimuthBin in
        #            zip(altitude_surfaces, azimuth_bins)]

        sigma_azimuths = [(2 * np.pi / azimuthBin) / np.sqrt(-8 * np.log(0.5)) for azimuthBin in azimuth_bins]
        sigma_altitude = (np.pi / n_altitude) / np.sqrt(-8 * np.log(0.5))

        headings = [-np.pi + (2 * np.pi / n_heading) * j for j in range(n_heading)]
        sigma_heading = (2 * np.pi / n_heading) / np.sqrt(-8 * np.log(0.5))

        for i, altitude in enumerate(altitudes):
            azimuths = [-np.pi + (2 * np.pi / azimuth_bins[i]) * j for j in range(azimuth_bins[i])]
            sigma_s = [2 * sigma_uwb, sigma_azimuths[i], sigma_altitude]
            for azimuth in azimuths:
                s = np.array([r, azimuth, altitude], dtype=float)
                for heading in headings:
                    particle = self.create_particle()
                    particle.set_initial_state(s, sigma_s)
                    self.particles.append(particle)
        # for alt in [-np.pi/2, np.pi/2]:
        #     for az in [-np.pi, np.pi]:
        #         for heading in headings:
        #             particle = TargetTrackingUKF(x_ha_0=self.ha.x_ha_0, weight=1)
        #             particle.set_ukf_properties(self.kappa, self.alpha, self.beta)
        #             particle.set_initial_state(np.array([r, az, alt]), np.array([sigma_uwb, 2*np.pi/3, sigma_altitude]),
        #                                        heading, sigma_heading, sigma_uwb)
        #             self.particles.append(particle)

        self.set_best_particle(self.particles[0])

    def run_model(self, dt_j, q_j,  dt_i, q_i, d_ij,  time_i=None):
        # self.iterations += 1
        self.uwb_measurement = d_ij
        self.time_i = time_i
        # self.check_validity(dx_ca, q_ca)
        # self.ha.predict(dx_ha=dt_i, Q_ha=q_i)
        # self.ha.reset_integration()


        self.run_predict_update_los(dt_j, q_j,  dt_i, q_i, d_ij)
        self.resample()
        # self.calculate_average_particle()
        if len(self.particles) > 5000:
            raise Exception("Too many particles")

    def check_validity(self, dx_ca, q_ca):
        if q_ca is not None or q_ca == np.zeros((4, 4)):
            dis = scipy.spatial.distance.mahalanobis(np.zeros(4), dx_ca, np.linalg.inv(q_ca))
            print(dis)
        else:
            print("No distance can be calculated")

    def run_predict_update_los(self, dt_j, q_j,  dt_i, q_i, d_ij):
        keep = []
        self.totalWeight = 0
        self.weights = []
        P_x_ha = self.ha.get_x_ha_odom_sigma()
        self.sigma_x_ha = np.linalg.norm(np.sqrt(np.diag(P_x_ha[:3, :3])))

        for particle in self.particles:
            try:
                particle.run_model(dt_i=dt_i, q_i=q_i, t_i=self.ha.x_ha, P_i=P_x_ha,
                                   dt_j=dt_j, q_j=q_j, d_ij=d_ij, sig_uwb=self.sigma_uwb, time_i=self.time_i)
                keep.append(particle)
                self.weights.append(particle.weight)
                self.totalWeight += particle.weight
            except np.linalg.LinAlgError as e:
                print(e)
                pass
        self.particles = keep

    # def resample(self):
    #     """
    #     Branch kill resampling for UPF.
    #     """
    #     new_particles = []
    #     new_weight = 0
    #     new_weights = []
    #
    #     # Lowerd the average_weight such that depletion is less fast.
    #     factor = 10
    #     average_weight = 1/ factor/ len(self.particles)
    #     # average_weight = self.totalWeight / len(self.particles)
    #
    #     # First let's do best particle.
    #     # best_particle.weight = best_particle.weight / self.totalWeight
    #     # new_particles.append(best_particle)
    #     # new_weight += best_particle.weight
    #
    #     for particle in self.particles:
    #         particle.weight = particle.weight / self.totalWeight
    #         size = int(particle.weight / average_weight)
    #         weight = int(particle.weight / average_weight)
    #
    #
    #         if weight > 0:
    #             weight = int(size / factor) + 1.
    #             merged = False
    #             for i, kept_particle in enumerate(new_particles):
    #                 if self.compare_particle(kept_particle, particle):
    #                     kept_particle.weight += 1.
    #                     new_weight += 1.
    #                     new_weights[i] += particle.kf.likelihood
    #                     merged = True
    #                     break
    #             if not merged:
    #                 particle.weight = 1.
    #                 new_particles.append(particle)
    #                 new_weights.append(particle.kf.likelihood)
    #                 new_weight += particle.weight
    #                 # particle.weight = weight*average_weight
    #                 # new_particles.append(particle)
    #                 # new_weight += weight
    #
    #                 # else:
    #                 #     best_particle.weight += particle.weight
    #                 #     new_weight += particle.weight
    #
    #     self.particles = new_particles
    #     best_particle = self.particles[np.where(new_weights == np.max(new_weights))[0][0]]
    #     self.set_best_particle(best_particle)
    #
    #     self.generate_new_particles()
    #     if not self.particles:
    #         raise Exception("No particles left")

    def set_normal_resampling(self, resample_factor=0.1, uwb_sigma_factor=1.5):
        self.sigma_uwb_factor = uwb_sigma_factor
        self.resample_factor = resample_factor
        self.resample = self.normal_resampling

    def normal_resampling(self):
        new_particles = []
        new_weight = 0
        new_weights = []
        # Lowerd the average_weight such that depletion is less fast.
        average_weight = self.resample_factor / len(self.particles)

        for particle in self.particles:
            particle.weight = particle.weight / self.totalWeight
            size = int(particle.weight / average_weight)
            weight = int(particle.weight / average_weight)

            if weight > 0:
                # weight = int(size / factor) + 1.
                merged = False
                for i, kept_particle in enumerate(new_particles):
                    if self.compare_particle(kept_particle, particle):
                        kept_particle.weight += particle.weight
                        new_weight += particle.weight
                        new_weights[i] += particle.weight
                        merged = True
                        break
                if not merged:
                    new_particles.append(particle)
                    new_weights.append(particle.weight)
                    new_weight += particle.weight

        self.particles = new_particles
        best_particle = self.particles[np.where(new_weights == np.max(new_weights))[0][0]]
        self.set_best_particle(best_particle)

        self.generate_new_particles()
        if not self.particles:
            raise Exception("No particles left")

    def set_branch_kill_resampling(self, resample_factor=0.1, sigma_uwb_factor=1.5):
        self.sigma_uwb_factor = sigma_uwb_factor
        self.resample_factor = resample_factor
        self.resample = self.branch_kill_resampling

    def branch_kill_resampling(self):
        new_particles = []
        new_weight = 0
        new_weights = []

        # Lowerd the average_weight such that depletion is less fast.
        # average_weight = 1. / factor / len(self.particles)
        average_weight = self.resample_factor / len(self.particles)

        # First let's do best particle.
        # best_particle.weight = best_particle.weight / self.totalWeight
        # new_particles.append(best_particle)
        # new_weight += best_particle.weight

        for particle in self.particles:
            particle.weight = particle.weight / self.totalWeight
            size = int(particle.weight / average_weight)
            weight = int(particle.weight / average_weight)

            if weight > 0:
                if weight > 1 / self.resample_factor:
                    weight = 2.
                else:
                    weight = 1.
                # weight = int(size / factor) + 1.
                merged = False
                for i, kept_particle in enumerate(new_particles):
                    if self.compare_particle(kept_particle, particle) is not None:
                        kept_particle.weight += weight
                        new_weight += weight
                        new_weights[i] += particle.weight
                        merged = True
                        break
                if not merged:
                    particle.weight = weight
                    new_particles.append(particle)
                    new_weights.append(particle.weight)
                    new_weight += particle.weight
                    # particle.weight = weight*average_weight
                    # new_particles.append(particle)
                    # new_weight += weight

                    # else:
                    #     best_particle.weight += particle.weight
                    #     new_weight += particle.weight

        self.particles = new_particles
        best_particle = self.particles[np.where(new_weights == np.max(new_weights))[0][0]]
        self.set_best_particle(best_particle)

        # self.generate_new_particles()
        if not self.particles:
            raise Exception("No particles left")

    def calculate_average_particle(self):
        #TODO calculate avarega particle + uncertainty.
        self.average_t_si_sj = np.zeros(4)
        self.t_si_sj = np.empty((0, 4))
        for particle in self.particles:
            self.average_t_si_sj = particle.t_si_sj * particle.weight + self.average_t_si_sj
            self.t_si_sj = np.concatenate((self.t_si_sj, particle.t_si_sj.reshape(1, 4)), axis=0)

        self.t_si_sj_sig = np.max(self.t_si_sj, axis=0) - np.min(self.t_si_sj, axis=0)

    def set_best_particle(self, best_particle):
        self.best_particle = best_particle

    def compare_particle(self, particle_1: TargetTrackingParticle, particle_2: TargetTrackingParticle):
        # TODO: Improve this, (does not take into account uncertainty.)
        """
        Compare the particle with the best particle.
        :param particle_2:
        :return:
        """
        if particle_1 is not particle_2:
            value = particle_1.compare(particle_2)
            print(value, self.max_dis)
            if value is not None and value < self.max_dis:
                if particle_1.weight > particle_2.weight:
                    return particle_1
                else:
                    return particle_2
        return None



class UPFConnectedAgentDataLogger:
    def __init__(self, host_agent: NewRobot, connected_agent: NewRobot, upf_connected_agent: UPFConnectedAgent):
        self.host_agent = host_agent
        self.connected_agent = connected_agent
        self.upf_connected_agent = upf_connected_agent
        self.particle_count = 0
        self.particle_logs = []
        self.init_variables()
        self.i = 0

        # Host agent variables:
        self.number_of_particles = []
        self.ha_pose_stds = np.zeros((0, 4))
        self.sigma_x_ha = []
        self.dx_ha_stds = np.zeros((0, 4))
        self.dx_drift = np.zeros((0, 4))

        # Timing variables:
        self.calulation_time = []

    def find_particle_log(self, particle) -> UKFDatalogger:
        for particle_log in self.particle_logs:
            if particle_log.ukf == particle:
                return particle_log
        return None

    def get_best_particle_log(self):
        return self.find_particle_log(self.upf_connected_agent.best_particle)

    def add_particle(self, particle):
        # particle.set_datalogger(self.host_agent, self.connected_agent, name="Particle " + str(self.particle_count))
        particle_log = UKFDatalogger(self.host_agent, self.connected_agent, particle,
                                     name="Particle " + str(self.particle_count))
        self.particle_count += 1
        self.particle_logs.append(particle_log)

    def init_variables(self):
        for particle in self.upf_connected_agent.particles:
            self.add_particle(particle)
            # self.ukfs_logger.append(Datalogger(self.host_agent, self.connected_agent, particle, name="Particle "+ str(i)))

    def log_data(self, i, calculation_time=0):
        self.calulation_time.append(calculation_time)
        if self.upf_connected_agent.time_i is None:
            self.i = i
        else:
            self.i = self.upf_connected_agent.time_i
        self.log_ha_data()
        for particle in self.upf_connected_agent.particles:
            particle_log: UKFDatalogger = self.find_particle_log(particle)
            if particle_log is not None:
                particle_log.log_data(i)
            else:
                self.add_particle(particle)

    def log_ha_data(self):
        self.number_of_particles.append(len(self.upf_connected_agent.particles))

        ha_stds = np.sqrt(np.diag(self.upf_connected_agent.ha.p_x_ha))
        self.ha_pose_stds = np.concatenate((self.ha_pose_stds, ha_stds.reshape(1, 4)), axis=0)
        self.sigma_x_ha.append(self.upf_connected_agent.sigma_x_ha)
        dx_ha_stds = np.diag(
            self.upf_connected_agent.ha.p_dx_ha)  # np.sqrt(np.diag(self.upf_connected_agent.ha.p_dx_ha))
        self.dx_ha_stds = np.concatenate((self.dx_ha_stds, dx_ha_stds.reshape(1, 4)), axis=0)
        dx_ha = self.upf_connected_agent.ha.dx_drift
        self.dx_drift = np.concatenate((self.dx_drift, dx_ha.reshape(1, 4)), axis=0)

    # ---- Plot functions
    def plot_poses(self, ax, color_ha, color_ca, name_ha, name_ca):
        self.plot_estimated_trajectory(ax, color=color_ca, name=name_ca)
        self.host_agent.set_plotting_settings(color=color_ha)
        self.host_agent.plot_real_position(ax, annotation=None)

    def plot_start_poses(self, ax):
        # plt.figure()
        # ax = plt.axes(projection="3d")
        self.plot_estimated_trajectory(ax, color="r")
        # self.plot_connected_agent_trajectory(ax)
        self.plot_best_particle(ax, color="gold", alpha=1)
        self.plot_connected_agent_trajectories(ax, color="k", i=self.i)
        self.plot_host_agent_trajectory(ax, color="darkgreen", i=self.i)

        ax.legend()

    def plot_host_agent_trajectory(self, ax, color="darkgreen", name="Host Agent", i=-1, history=None):
        self.host_agent.set_plotting_settings(color=color)
        self.host_agent.plot_real_position(ax, annotation=name, i=i, history=history)

    def plot_connected_agent_trajectories(self, ax, color="k", i=-1):
        # self.plot_estimated_trajectory(ax, color=color, alpha=0.1)
        self.plot_connected_agent_trajectory(ax, color=color, alpha=1, i=i)
        self.plot_best_particle(ax, color=color, alpha=0.5)

    def plot_connected_agent_trajectory(self, ax, color="black", alpha=1., i=-1, history=None):
        self.connected_agent.set_plotting_settings(color=color)
        self.connected_agent.plot_real_position(ax, annotation="Connected Agent", alpha=alpha, i=i, history=history)
        # self.connected_agent.plot_slam_position(ax, annotation="Connected Agent SLAM", alpha=alpha)

    def plot_estimated_trajectory(self, ax, color="k", alpha=0.1, name="connected agent"):
        for particle_log in self.particle_logs:
            particle_log.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=0.1, linestyle=":", label=None)
        for particle in self.upf_connected_agent.particles:
            particle_log = self.find_particle_log(particle)
            particle_log.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=1, label=None)

    def plot_self(self, los=None, host_id="No host id"):
        bp_dl: UKFDatalogger = self.find_particle_log(self.upf_connected_agent.best_particle)
        fig = plt.figure(figsize=(18, 10))  # , layout="constrained")
        fig.suptitle("Host Agent: " + host_id + "; Connected agent: " + self.upf_connected_agent.id)
        ax = []
        gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        ax_3d = fig.add_subplot(gs[:3, :3], projection="3d")
        self.plot_start_poses(ax_3d)

        # ---- Best Particle Axis
        ax_best_particle = [fig.add_subplot(gs[i, -1]) for i in range(2)]
        # ax_best_particle[0].set_title("Best Particle")
        bp_dl.plot_ukf_drift(ax_best_particle)
        ax_best_particle[0].legend(loc="upper left")

        # ---- Host agent Axis
        ha_ax = fig.add_subplot(gs[2, -1])
        ha_ax.plot(self.sigma_x_ha, label="std of the host agent trajectory ha [m]")
        label = ["x [m]", "y [m]", "z [m]", "h [(rad)]"]
        ha_ax.set_ylabel("std of dx of the host agent")
        for i in range(4):
            ha_ax.plot(self.ha_pose_stds[:, i], label=label[i])
        ha_ax.grid(True)
        ha_ax.legend()

        dx_ha_ax = fig.add_subplot(gs[3, -1])
        label = ["x [m]", "y [m]", "z [m]", "h [(rad)]"]
        dx_ha_ax.set_xlabel("std of dx of the host agent")
        for i in range(4):
            dx_ha_ax.plot(self.dx_ha_stds[:, i], label=label[i])
        dx_ha_ax.legend()

        dx_ha_ax_1 = fig.add_subplot(gs[3, 2])
        label = ["x [m]", "y [m]", "z [m]", "h [(rad)]"]
        dx_ha_ax_1.set_xlabel("dx drift of the host agent")
        for i in range(4):
            dx_ha_ax_1.plot(self.dx_drift[:, i], label=label[i])
        dx_ha_ax_1.legend()

        # ---- Particle Axis
        particle_ax = fig.add_subplot(gs[3, 0])
        particle_ax.plot(self.number_of_particles)
        particle_ax.set_title("Number of particles")
        particle_ax.grid(True)

        likelihood_ax = fig.add_subplot(gs[3, 1])
        likelihood_ax.plot(self.calulation_time, label="Calculation time")
        likelihood_ax.set_title("Calculation time")
        likelihood_ax.legend()

        # ---- Likelihood Axis
        # likelihood_ax = fig.add_subplot(gs[3, 1])
        # plt.figure()
        # likelihood_ax = plt
        likelihood_ax.plot(bp_dl.likelihood, label="Likelihood")
        likelihood_ax.plot(bp_dl.weight, label="Weigth")
        if los is not None:
            likelihood_ax.plot(los, color="k", label="Real LOS State")

        likelihood_ax.legend()
        likelihood_ax.grid(True)
        likelihood_ax.set_title("LOS state and likelihood best particle.")
        # likelihood_ax.suptitle("LOS state and likelihood best particle.")
        # likelihood_ax.set_xlabel("Time [s]")

        # plt.figure()
        # if los is not None:
        #     plt.plot(los, color="k", label="LOS State")
        #     plt.plot(bp_dl.los_state, linestyle="--", color="crimson", label="LOS state estimation")
        #
        # plt.xlabel("Time [s]")
        # plt.legend()
        # plt.grid(True)

        return fig

    def plot_best_particle(self, ax, color="gold", alpha=1., history=None):
        # print(self.upf_connected_agent.best_particle)
        best_particle_log = self.find_particle_log(self.upf_connected_agent.best_particle)
        best_particle_log.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=alpha,
                                                                 label="Best Particle", history=history)

    def plot_best_particle_variance_graph(self):
        best_particle_log = self.find_particle_log(self.upf_connected_agent.best_particle)
        best_particle_log.plot_error_graph()
        for particles in self.upf_connected_agent.particles:
            particle_log = self.find_particle_log(particles)
            particle_log.plot_error_graph()

    def plot_connected_agent(self, ax):
        bp_dl = self.find_particle_log(self.upf_connected_agent.best_particle)
        bp_dl.plot_ukf_drift(ax[:2])

        likelihood_ax = ax[2]
        likelihood_ax.plot(bp_dl.likelihood, color="darkred", label="Likelihood")
        likelihood_ax.plot(bp_dl.weight, color="darkblue", label="Weigth")
        # likelihood_ax.plot(bp_dl.los_state, color="darkgreen", label="LOS state")
        likelihood_ax.plot(0, color="k", label="# particles")
        likelihood_ax.legend()
        likelihood_ax.grid(True)

        tw = likelihood_ax.twinx()
        tw.plot(self.number_of_particles, color="k")

    def plot_ca_best_particle(self, ax, i, color, history):
        bp_dl = self.find_particle_log(self.upf_connected_agent.best_particle)
        bp_dl.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=1, label=None, i=i, history=history)

    def plot_ca_active_particles(self, ax, i, color, history):
        active_particles = []
        for par_log in self.particle_logs:
            if par_log.i > i:
                # active_particles.append(par_log)
                par_log.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=1, label=None, i=i,
                                                               history=history)
                # par_log.datalogger.plot_ca_estimated_trajectory(ax, color="b", alpha=0.3, label=None, i = int(i/10)+1)
        # self.plot_connected_agent_trajectory(ax, i = i)

        # fig = plt.figure(figsize=(18, 10), projection="3d")
