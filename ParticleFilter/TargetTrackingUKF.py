#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:30:22 2023

@author: yuri

###############################################################################

THIS MODULE CONTAINS A UKF FOR TARGET TRACKING ASSUMING THE BEARINGS OF OBJECT
ARE KNOWN.


The idea is given that, working in spherical coordinates, that knowing roughly
the azimuth, altitude and distance to an object and given that we can measure
the relative displacement of said object we can estimate its heading and
trajectory using an UKF.

Why an UKF because its heading is not know to use from the start and sin and cos
are highly non-linear functions.

In this case the trajectory of the connected agent is kept. This includes a lot more information,
but makes it also much more complex.

"""
from datetime import datetime

import numpy as np
import h5py
import copy
import matplotlib.pyplot as plt
from ParticleFilter.ModifiedUKF import ModifiedUnscentedKalmanFilter, ModifiedMerweScaledSigmaPoints
from UtilityCode.utility_fuctions import cartesianToSpherical, sphericalToCartesian, limit_angle, \
    get_4d_rot_matrix, transform_matrix, inv_transformation_matrix, \
    get_states_of_transform, get_covariance_of_transform
from Simulation.RobotClass import NewRobot
from deprecated import deprecated


def subtract(x, y):
    # dx = np.zeros(x.shape)
    dx = x - y
    # dx[0] = x[0] - y[0]
    # dx[1] = limit_Angle(x[1] - y[1])
    # dx[2] = limit_Angle(x[2] - y[2])
    # # if np.abs(dx[2]) > np.pi /2:
    # #     dx[2] = np.sign(dx[2]) * ( np.abs(dx[2]) - 2* (np.abs(dx[2]) - np.pi/2))
    # #     dx[1] = limit_Angle(dx[1] + np.pi)
    # dx[3] = x[3] - y[3]
    return dx


def subtract_spherical(x, y):
    # Does not work because a negative R becomes - pi in azimuth.
    dx = np.zeros(x.shape)
    dx[:3] = cartesianToSpherical(sphericalToCartesian(x[:3]) - sphericalToCartesian(y[:3]))
    dx[3] = limit_angle(x[3] - y[3])

    return dx


class TargetTrackingUKF:
    """
    state vector = [ x, y, phi, theta_CA]
    input vector = [ Dx_HA, Dy_HA, Dz_HA, Dx_CA, Dy_CA, Dz_CA, D_theta_CA]
    measurement vector = [d_uwb]

    # TODO: Incorporate the heading uncertainty of the host agents in the state.
        Use the uncertainty of the host agents heading to generate particles at a distance from the current location of the host agent.
        How do we track during NLOS? -> Keep tack of accumuated uncertainty since this does not affect the Transformation estimation
        Generate more particles one a LOS measurement is received?
        Remove the uncertainty from the UWB.py measurement -> translational drift is incorporated into the connecte agent from both host and connected agent.
        Remove the uncertainty of the host agent heading from the state. -> Covered now by generating particles using this uncertainty. (Highly non linear.)

    # TODO : will include trhe heading uncertainty of the host agent as an additional state. Hmm not sure this is the best thing to do.
        Keeping the heading uncertainty of the host agent at the level of the host agent may help reduce the drift on this heading.

    # TODO: So what I need to do is keep the the higher level EKF to correct for the drift of the host agent using the extimated trajectory.
        and then keep the residual drift and propagate it to the UPF. Where the uncertaitny on the location is added to the location of the connected agent.
        And augmenting the state of the UKF with an additional state that represent the uncertainty of the host heading.
        and the heading drift is used to create new particles that are moved arround the host agent using the uncertainty on the host agent heading.
    """

    def __init__(self, x_ha_0=np.zeros(4), weight=1., los_state=True, data_logging_bool=True,
                 minimun_likelihood_factor=0.1, drift_correction_bool = True):

        # ---- Input variables old:
        # x_ha : Current position (x,y,z,h) of the host agent in the absolute reference frame of the host agent.
        # self.x_ha = x_ha_0
        # self.x_ha_prev = x_ha_0
        # Dt_sj : Delta position odometry of the connected agent expressed in his local odometry frame.
        self.Dt_sj = np.zeros(4)
        # q_ca: noise matrix of the odometry of the connected agent.
        self.q_ca = np.zeros((4, 4))
        #  uwb_measurement : Measurement of the UWB.py sensor.
        self.uwb_measurement = 0
        #
        # # ---- Internal variables:
        # # x_ca_odom : Odometry of the connected agent wrt the position of the connected agent at connection time. (path of connected agent)
        # self.x_ca_odom = np.zeros(3)
        # # x_ha_0 : Position of the host agent in the absolute reference frame of the host agent at connection time.
        # self.x_ha_0 = x_ha_0
        # # x_ca : Position of the connected agent in the absolute reference frame of the host agent.
        # self.x_ca = np.zeros(3)
        # # x_ca_r : Relative position of connected agent wrt the position of the connected agent in the host agent absolute reference frame. (RTE)
        # self.x_ca_r = np.zeros(3)
        # # s_ca_r : Relative position of connected agent wrt the position of the connected agent expressed in the spherical coordinates.
        # self.s_ca_r = np.zeros(3)
        # # x_ca_0 : Position of the connected agent in the absolute reference frame of the host agent at connection time.
        # self.x_ca_0 = np.zeros(3)
        # # h_ca : Heading of the connected agent in the absolute reference frame of the host agent.
        # self.h_ca = 0
        # sigma_x_ca_0 : Uncertainty on the start position of the connected agent
        self.sigma_x_ca_0 = 0
        # sigma_x_ca : Uncertainty on the current position of the connected agent
        self.sigma_x_ca = 0
        # sigma_x_ca_r : Relative uncertainty on the relative transformation of the connected agent.
        self.sigma_x_ca_r = 0
        # simga_h_ca : Uncertainty on the heading of the connected agent.
        self.sigma_h_ca = 0
        # sigma_dx_ha : uncertainty on the odometry of the host agent.
        self.q_ha = np.zeros((4, 4))

        # ---- Input variables New:
        self.drift_correction_bool = drift_correction_bool
        self.t_oi_cij = x_ha_0
        self.t_oi_si_prev = x_ha_0
        self.t_oi_si = x_ha_0

        self.t_si_sj = np.zeros(4)
        self.t_cij_cji = np.zeros(4)
        self.t_oi_sj = np.zeros(4)
        self.t_oi_cji = np.zeros(4)
        self.t_cji_sj = np.zeros(4)
        self.D_t_sj = np.zeros(4)

        # ---- Kalman filter variables:
        # kf_variables = r_0, theta_0, phi_0, h_ca_0,  x_ca_odom, y_ca_odom, z_ca_odom, h_ca_odom
        # r_0, theta_0, phi_0: Relative position of the connected agent wrt the host agent at connection time expressed in spherical coordinates.
        # h_ca_0: Heading of the connected agent at connection time in the reference frame of the host agent at connection time.
        # x_ca_odom, y_ca_odom, z_ca_odom, h_ca_odom : Odometry of the connected agent wrt the position of the connected agent at connection time.
        # h_ha_drift : Drift of the host agent heading.
        self.kf_variables = 9
        self.kf: ModifiedUnscentedKalmanFilter = None
        self.weight = weight
        self.dt = 0.1
        self.alpha = 1
        self.kappa = -1
        self.beta = 2

        # ---- Kalman filter state:
        self.los_state = los_state
        self.nlos_degradation = 0.9
        self.P_x_ca = np.zeros((4, 4))
        self.sigma_dh_ca = 0
        self.sigma_dx_ca = 0

        # ---- NLOS detection variables:
        self.minimum_likelihood = 0.0
        self.minimum_likelihood_factor = minimun_likelihood_factor

        # ---- Data logging variables:
        self.time_i = None
        self.data_logging_bool = data_logging_bool
        self.datalogger = None

    # -------------------------------------------------------------------------------------- #
    # --- Initialization functions:
    # -------------------------------------------------------------------------------------- #
    def set_datalogger(self, host_agent, connected_agent, name):
        if self.data_logging_bool:
            self.datalogger = Datalogger(host_agent, connected_agent, self, name)

    def set_ukf_properties(self, kappa=-1, alpha=1, beta=2):
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        points = ModifiedMerweScaledSigmaPoints(self.kf_variables, alpha, beta, kappa, subtract=subtract)
        self.kf = ModifiedUnscentedKalmanFilter(dim_x=self.kf_variables, dim_z=1, dt=self.dt, fx=self.fx, hx=self.hx,
                                                points=points, residual_x=subtract)

    def set_initial_state(self, s, sigma_s, ca_heading, ca_sigma_heading, sigma_uwb):
        self.kf.x = np.array([s[0], s[1], s[2], ca_heading, 0, 0, 0, 0, 0])
        self.kf.P = np.diag(
            [(sigma_s[0]) ** 2, sigma_s[1] ** 2, sigma_s[2] ** 2, ca_sigma_heading ** 2, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
        self.calculate_x_ca()
        self.calculate_P_x_ca()

    def set_NLOS_parameters(self, minimum_likelihood_factor, nlos_degradation):
        self.minimum_likelihood_factor = minimum_likelihood_factor
        self.nlos_degradation = nlos_degradation
    # -------------------------------------------------------------------------------------- #
    # --- Filter functions: Main function
    # -------------------------------------------------------------------------------------- #
    def run_filter(self, dx_ca, q_ca, measurement, x_ha, P_x_ha, sigma_uwb, bool_drift = True, time_i=None):
        self.time_i = time_i
        self.t_oi_si = x_ha
        self.set_host_agent_uncertainty(P_x_ha)
        self.predict(dx_ca, q_ca)
        if self.los_state:
            self.update(measurement, x_ha, P_x_ha, sigma_uwb, bool_drift)
        else:
            self.nlos_update()
        self.calculate_x_ca()
        self.calculate_P_x_ca()
        self.weight = self.weight * self.kf.likelihood
        # self.weight = self.kf.likelihood
    # -------------------------------------------------------------------------------------- #
    # --- Prediction functions
    # -------------------------------------------------------------------------------------- #
    def predict(self, dx_ca, q=None):
        if q is None:
            q = np.diag([self.sigma_dx_ca ** 2, self.sigma_dx_ca ** 2, self.sigma_dx_ca ** 2, self.sigma_dh_ca ** 2])
        DT_sj = transform_matrix(self.Dt_sj).astype(np.float64)
        f = get_covariance_of_transform(DT_sj).astype(np.float64)
        self.Dt_sj = self.Dt_sj + f @ dx_ca.astype(np.float64)
        self.q_ca = self.q_ca + f @ q.astype(np.float64) @ f.T
        return None

    def reset_prediction_states(self):
        self.kf.Q = np.zeros((self.kf_variables, self.kf_variables))
        self.Dt_sj = np.zeros(4)
        self.q_ca = np.zeros((4, 4))

    def calculate_q(self):
        self.kf.Q = np.zeros((self.kf_variables, self.kf_variables))
        self.kf.Q[4:-1, 4:-1] = self.q_ca
        # self.kf.Q[1:4,1:4] = np.ones((3,3)) * 0.1 ** 2
        # Adding uncertainty on the heading of the host agent into the start pose of the connected agent.
        # self.kf.Q[-1,-1] = self.q_ha[-1,-1]

    def fx(self, x, dt):
        if self.drift_correction_bool:
            # calculate previous T_si_sj_k
            T_si_oi_k = inv_transformation_matrix(self.t_oi_si_prev)
            T_oi_cij_k = transform_matrix(self.t_oi_cij)
            t_cij_cji_k = np.append(sphericalToCartesian(x[:3]), np.array([x[3]]))
            T_cij_cji_k = transform_matrix(t_cij_cji_k)
            T_cji_sj_k = transform_matrix(x[4:-1])
            T_si_sj_k = T_si_oi_k @ T_oi_cij_k @ T_cij_cji_k @ T_cji_sj_k

            # calculate drifted T_cji_sj_d
            T_D = transform_matrix(np.array([0, 0, 0, x[-1]]))
            # T_D = np.eye(5)
            T_cij_oi = inv_transformation_matrix(self.t_oi_cij)
            T_cji_cij_k = inv_transformation_matrix(t_cij_cji_k)
            T_oi_si_k = transform_matrix(self.t_oi_si_prev)
            T_cji_sj_d = T_cji_cij_k @ T_cij_oi @ T_oi_si_k @ T_D @ T_si_sj_k
        else:
            T_cji_sj_d = transform_matrix(x[4:-1])
        # calculate new T_cji_sj
        DT_sj = transform_matrix(self.Dt_sj)
        T_cji_sj = T_cji_sj_d @ DT_sj
        x[4:-1] = get_states_of_transform(T_cji_sj)

        # x[4:-1] = x_odom[:3]
        return x

    # -------------------------------------------------------------------------------------- #
    # --- Update functions
    # -------------------------------------------------------------------------------------- #
    def calculate_r(self, sigma_uwb):
        self.kf.R = np.diag([np.linalg.norm(self.q_ha[:3, :3]) ** 2 + sigma_uwb ** 2])

    def update(self, z, x_ha, P_x_ha, sigma_uwb, bool_drift = True):
        # Prediction step
        self.calculate_q()
        self.kf.predict()
        # Update step
        self.uwb_measurement = z
        self.calculate_r(sigma_uwb)
        self.kf.update(np.array([z]))
        # Reset
        self.t_oi_si_prev = x_ha
        self.reset_prediction_states()
        self.set_residual_drift(bool_drift)

    def hx(self, x):
        T_oi_cij = transform_matrix(self.t_oi_cij)
        t_cij_cji = np.append(sphericalToCartesian(x[:3]), np.array([x[3]]))
        T_cij_cji = transform_matrix(t_cij_cji)
        T_cji_sj = transform_matrix(x[4:-1])
        T_si_oi = inv_transformation_matrix(self.t_oi_si)
        # T_sj_sj = transform_matrix(self.Dt_sj)
        T_si_sj = T_si_oi @ T_oi_cij @ T_cij_cji @ T_cji_sj
        t_si_sj = get_states_of_transform(T_si_sj)
        r = np.linalg.norm(t_si_sj[:3])

        # # Calculate the cartesian start position in the absolute reference frame of the host agent.
        # x_ca_0 = self.x_ha_0[:3] + get_rot_matrix(self.x_ha_0[-1]) @ sphericalToCartesian(x[:3])
        # # calculate the end position of the connected agent in the absolute reference frame of the host agent.
        # x_ca = x_ca_0 + get_rot_matrix(self.x_ha_0[-1]) @ get_rot_matrix(x[3]) @ x[4:-2]
        # # Calculate the distance between the two agents.


        return np.array([r])

    # -------------------------------------------------------------------------------------- #
    # --- Postprocessing functions
    # -------------------------------------------------------------------------------------- #
    def calculate_x_ca(self):

        T_oi_cij = transform_matrix(self.t_oi_cij)
        self.t_cij_cji = np.append(sphericalToCartesian(self.kf.x[:3]), np.array([self.kf.x[3]]))
        T_cij_cji = transform_matrix(self.t_cij_cji)
        T_cji_sj = transform_matrix(self.kf.x[4:-1])
        T_si_oi = inv_transformation_matrix(self.t_oi_si)
        T_sj_sj = transform_matrix(self.Dt_sj)
        T_si_sj = T_si_oi @ T_oi_cij @ T_cij_cji @ T_cji_sj @ T_sj_sj
        self.t_si_sj = get_states_of_transform(T_si_sj)
        # x_ca_r = self.t_si_sj[:3]
        # self.x_ca_r = x_ca_r
        # self.s_ca_r = cartesianToSpherical(x_ca_r)
        T_oi_sj = T_oi_cij @ T_cij_cji @ T_cji_sj @ T_sj_sj
        self.t_oi_sj = get_states_of_transform(T_oi_sj)
        # self.x_ca = self.t_oi_sj[:3]
        # self.h_ca = self.t_oi_sj[-1]
        T_oi_cji = T_oi_cij @ T_cij_cji
        self.t_oi_cji = get_states_of_transform(T_oi_cji)
        # self.x_ca_0 = self.t_oi_cji[:3]
        self.t_cji_sj = get_states_of_transform(T_cji_sj)
        # self.x_ca_odom = self.t_cji_sj[:3]

        #
        # else:
        #     # self.x_ca_odom = self.kf.x[4:-2]  # + create_Rotation_Matrix(self.kf.x[4]) @ self.dx_ca
        #     self.x_ca_odom = self.kf.x[4:-2] + get_rot_matrix(self.kf.x[-2]) @ self.Dt_sj[:3]
        #
        #     self.x_ca_0 = self.x_ha_0[:3] + get_rot_matrix(self.x_ha_0[-1]) @ sphericalToCartesian(self.kf.x[:3])
        #     # Calculate the absolute position of the connected agent.
        #     self.x_ca = self.x_ca_0 + get_rot_matrix(self.x_ha_0[-1]) @ get_rot_matrix(self.kf.x[3]) @ self.x_ca_odom
        #     self.x_ca_r = self.x_ca - self.x_ha[:3]  # expressed in the absolute reference frame of the host agent.
        #     self.s_ca_r = cartesianToSpherical(self.x_ca_r)
        #     # Expressed in the local reference frame of the host agent.
        #     self.s_ca_r[1] = limit_angle(self.s_ca_r[1] - self.x_ha[3])
        #     self.x_ca_r = sphericalToCartesian(self.s_ca_r)
        #     self.h_ca = self.x_ha_0[-1] + self.kf.x[3] + self.kf.x[-2] + self.Dt_sj[-1]
        return None

    def calculate_P_x_ca(self):
        f = get_4d_rot_matrix(self.kf.x[3])
        stds = np.sqrt(np.diag(self.kf.P))
        ca_0 = sphericalToCartesian(self.kf.x[:3])  # + self.x_ha_0[:3]
        ca_0_max = sphericalToCartesian(self.kf.x[:3] + stds[:3])  # + self.x_ha_0[:3]
        dis_0 = np.linalg.norm(ca_0_max - ca_0)

        self.sigma_x_ca_0 = dis_0  # / self.kf.x[0]
        self.sigma_x_ca = np.sqrt(dis_0 ** 2
                                  + np.linalg.norm(self.kf.P[4:-2, 4:-2].astype(np.float64))
                                  + np.linalg.norm(self.q_ca[:3, :3].astype(np.float64)))
        self.sigma_h_ca = np.sqrt(self.kf.P[3, 3] + self.kf.P[-2, -2] + self.q_ca[-1, -1])

    # -------------------------------------------------------------------------------------- #
    # --- Residual drift functions
    # -------------------------------------------------------------------------------------- #
    def set_host_agent_uncertainty(self, P_x_ha):
        T_oi_si = transform_matrix(self.t_oi_si_prev)
        f = get_covariance_of_transform(T_oi_si)
        # f = get_4d_rot_matrix(self.x_ha_prev[-1]).astype(np.float64)
        self.q_ha = self.q_ha + f @ P_x_ha.astype(np.float64) @ f.T

    def set_residual_drift(self, bool_drift = True):
        # Should be called at the PF level in case there is residual drift.
        self.kf.P[:, -1] = np.zeros(self.kf_variables)
        self.kf.P[-1, :] = np.zeros(self.kf_variables)
        self.kf.P[-1, -1] = 1e-20
        self.kf.x[-1] = 0
        if bool_drift:
            self.q_ca[:3, :3] = self.q_ca[:3, :3] + self.q_ha[:3, :3]
            self.kf.P[-1, -1] = self.kf.P[-1, -1] + self.q_ha[-1, -1]
            if np.sqrt(self.kf.P[-1, -1]) > np.pi :
                self.kf.P[-1, -1] = np.pi**2
        self.q_ha = np.zeros((4,4))


    # -------------------------------------------------------------------------------------- #
    # --- NLOS functions
    # -------------------------------------------------------------------------------------- #
    def switch_los_state(self):
        self.los_state = not self.los_state
        self.minimum_likelihood = self.kf.likelihood * self.minimum_likelihood_factor

    def nlos_update(self):
        self.kf._likelihood = self.nlos_degradation * self.kf.likelihood
        if self.kf._likelihood < self.minimum_likelihood:
            self.kf._likelihood = self.minimum_likelihood

    # -------------------------------------------------------------------------------------- #
    # --- Copy functions
    # -------------------------------------------------------------------------------------- #

    def copy(self):
        copiedUKF = TargetTrackingUKF(x_ha_0=self.t_oi_cij, weight=self.weight, los_state=self.los_state,
                                      data_logging_bool=self.data_logging_bool)
        copiedUKF.set_ukf_properties(self.kappa, self.alpha, self.beta)

        copiedUKF.kf.x = copy.deepcopy(self.kf.x)
        copiedUKF.kf.P = copy.deepcopy(self.kf.P)
        copiedUKF.kf.R = copy.deepcopy(self.kf.R)
        copiedUKF.kf.Q = copy.deepcopy(self.kf.Q)
        copiedUKF.kf._likelihood = copy.deepcopy(self.kf.likelihood)
        copiedUKF.minimum_likelihood_factor = copy.deepcopy(self.minimum_likelihood_factor)
        copiedUKF.nlos_degradation = copy.deepcopy(self.nlos_degradation)
        copiedUKF.minimum_likelihood = copy.deepcopy(self.minimum_likelihood)

        copiedUKF.Dt_sj = copy.deepcopy(self.Dt_sj)
        # copiedUKF.dh_ca = copy.deepcopy(self.dh_ca)
        copiedUKF.q_ca = copy.deepcopy(self.q_ca)
        copiedUKF.x_ha = copy.deepcopy(self.t_oi_si)
        copiedUKF.x_ha_prev =  copy.deepcopy(self.t_oi_si_prev)
        copiedUKF.q_ha = copy.deepcopy(self.q_ha)

        # copiedUKF.h_ha = copy.deepcopy(self.h_ha)

        copiedUKF.dataLogging = copy.deepcopy(self.data_logging_bool)
        copiedUKF.datalogger = self.datalogger.copy(copiedUKF)
        copiedUKF.weight = copy.deepcopy(self.weight)

        return copiedUKF
    #


class Datalogger():
    def __init__(self, hostAgent: NewRobot, connectedAgent: NewRobot, ukf: TargetTrackingUKF,
                 name="Undefined Testcase"):

        self.name = name
        self.host_agent: NewRobot = hostAgent
        self.connected_agent: NewRobot = connectedAgent
        self.ukf = ukf

        # Color properties for plotting:
        self.estimation_color = "steelblue"
        self.slam_color = "crimson"
        self.absolute_linestyle = "-"
        self.relative_linestyle = "-"
        self.mean_linestyle = "-"
        self.stds_linestyle = "--"

        if self.ukf is not None:
            self.kf_variables = self.ukf.kf_variables
            # ---- Save to file variables:
            self.save_bool = False
            self.now = None
            self.save_folder = None
            self.prior_save_name = ""
            self.posterior_save_name = ""

            # ---- logged variables:
            self.data_logged = False
            self.i = 0

            self.host_agent_trajectory = np.empty((0, 3))
            self.connected_agent_trajectory = np.empty((0, 3))
            self.connected_agent_heading = []

            self.ca_position_estimation = np.empty((0, 3))
            self.error_ca_position = []

            self.ca_heading_estimation = []
            self.error_ca_heading = []
            self.error_ca_position_slam = []
            self.error_ca_heading_slam = []

            self.spherical_estimated_relative_transformation = np.empty((0, 3))
            self.spherical_relative_transformation = np.empty((0, 3))
            self.error_spherical_relative_transformation_estimation = np.empty((0, 3))

            self.error_relative_transformation_est = []
            self.error_relative_heading_est = []
            self.error_relative_heading_slam = []
            self.error_relative_transformation_slam = []

            self.spherical_relative_start_position = np.empty((0, 3))
            self.spherical_estimated_relative_start_position = np.empty((0, 4))

            self.x = np.empty((0, self.kf_variables))
            self.stds = np.empty((0, self.kf_variables))
            self.stds_trajectory = []
            self.sigma_x_ca_0 = []
            self.sigma_x_ca = []
            self.sigma_x_ca_r = []
            self.sigma_h_ca = []
            self.likelihood = []
            self.weight = []
            self.los_state = []

            self.ca_s = []
            self.ha_s = []

            self.estimated_ca_position = np.empty((0, 3))
            # metric variables:
            self.mean_error_relative_transformation_est = []
            self.sigma_error_relative_transformation_est = []
            self.mean_error_relative_heading_est = []
            self.sigma_error_relative_heading_est = []

            self.mean_error_relative_transformation_slam = []
            self.sigma_error_relative_transformation_slam = []
            self.mean_error_relative_heading_slam = []
            self.sigma_error_relative_heading_slam = []

    def save_graphs(self, folder="./"):
        self.now = datetime.now()
        self.save_bool = True
        self.save_folder = folder
        self.prior_save_name = "_".join(self.name.split(":")[0].split(" "))
        self.posterior_save_name = "_".join([str(self.now.hour), str(self.now.minute), str(self.now.second)])
        if self.save_folder[-1] != "/":
            self.save_folder += "/"

    def log_data(self, i):
        self.data_logged = True
        if self.ukf.time_i is None:
            self.i = i
        else:
            self.i = self.ukf.time_i
        self.log_spherical_data(self.i)
        self.log_cartesian_data(self.i)
        self.log_slam_data(self.i)
        self.log_variances()
        # self.log_metric()

    @deprecated
    def log_metric(self):
        self.mean_error_relative_heading_slam.append(np.mean(self.error_relative_heading_slam))
        self.sigma_error_relative_heading_slam.append(np.std(self.error_relative_heading_slam))
        self.mean_error_relative_transformation_slam.append(np.mean(self.error_relative_transformation_slam))
        self.sigma_error_relative_transformation_slam.append(np.std(self.error_relative_transformation_slam))

        self.mean_error_relative_heading_est.append(np.mean(self.error_relative_heading_est))
        self.sigma_error_relative_heading_est.append(np.std(self.error_relative_heading_est))
        self.mean_error_relative_transformation_est.append(np.mean(self.error_relative_transformation_est))
        self.sigma_error_relative_transformation_est.append(np.std(self.error_relative_transformation_est))

    def log_variances(self):
        self.likelihood.append(self.ukf.kf.likelihood)
        self.weight.append(self.ukf.weight)
        self.x = np.append(self.x, np.reshape(self.ukf.kf.x, (1, self.kf_variables)), axis=0)
        std = np.array([np.sqrt(self.ukf.kf.P[i, i]) for i in range(self.ukf.kf.P.shape[0])])
        self.stds = np.append(self.stds, np.reshape(std, (1, self.kf_variables)), axis=0)
        self.stds_trajectory.append(np.linalg.norm(std[5:]))

        self.sigma_x_ca_0.append(self.ukf.sigma_x_ca_0)
        self.sigma_x_ca.append(self.ukf.sigma_x_ca)
        self.sigma_x_ca_r.append(self.ukf.sigma_x_ca_r)
        self.sigma_h_ca.append(self.ukf.sigma_h_ca)

        self.los_state.append(1 if self.ukf.los_state else 0)

    def log_spherical_data(self, i):
        ca_p_real = self.connected_agent.x_real[i]
        ca_h_real = self.connected_agent.h_real[i]
        ha_p_real = self.host_agent.x_real[i]
        ha_h_real = self.host_agent.h_real[i]



        # relative_transformation = (ca_p_real - ha_p_real)
        # spherical_relative_transformation = cartesianToSpherical(relative_transformation)
        # spherical_relative_transformation[1] = limit_angle(spherical_relative_transformation[1] - ha_h_real)
        # relative_transformation_r = sphericalToCartesian(spherical_relative_transformation)
        #
        # error_relative_transformation_est = np.linalg.norm(
        #     relative_transformation_r - sphericalToCartesian(self.ukf.s_ca_r))
        t_G_si = np.append(ha_p_real, np.array([ha_h_real]))
        T_G_si = transform_matrix(t_G_si)
        T_si_G = inv_transformation_matrix(np.append(ha_p_real, np.array([ha_h_real])))
        T_G_sj = transform_matrix(np.append(ca_p_real, np.array([ca_h_real])))
        T_si_sj = T_si_G @ T_G_sj
        t_si_sj = get_states_of_transform(T_si_sj)
        e_t_si_sj = t_si_sj - self.ukf.t_si_sj

        error_relative_transformation_est = np.linalg.norm(e_t_si_sj[:3])
        self.error_relative_transformation_est.append(error_relative_transformation_est)

        spherical_relative_transformation = cartesianToSpherical(t_si_sj[:3])
        self.spherical_relative_transformation = np.append(self.spherical_relative_transformation,
                                                           spherical_relative_transformation.reshape(1, 3), axis=0)

        spherical_relative_transformation_estimation = np.reshape(cartesianToSpherical(self.ukf.t_si_sj[:3]), (1, 3))
        self.spherical_estimated_relative_transformation = np.append(self.spherical_estimated_relative_transformation,
                                                                     spherical_relative_transformation_estimation,
                                                                     axis=0)

        self.spherical_estimated_relative_start_position = np.append(self.spherical_estimated_relative_start_position,
                                                                     np.reshape(self.ukf.kf.x[:4], (1, 4)), axis=0)

        error_s = spherical_relative_transformation - spherical_relative_transformation_estimation
        error_s[0, 1] = limit_angle(error_s[0, 1])
        error_s[0, 2] = limit_angle(error_s[0, 2])
        error_s = np.reshape(np.abs(error_s), (1, 3))
        self.error_spherical_relative_transformation_estimation = np.append(
            self.error_spherical_relative_transformation_estimation, error_s, axis=0)

        est_T_G_sj = T_G_si @ transform_matrix(self.ukf.t_si_sj)
        est_t_G_sj = get_states_of_transform(est_T_G_sj)
        self.estimated_ca_position = np.append(self.estimated_ca_position, est_t_G_sj[:3].reshape(1, 3), axis=0)

    def log_cartesian_data(self, i):
        # self.ca_s.append(self.connected_agent.getLogData("Odom s")[i])
        # self.ha_s.append(self.host_agent.getLogData("Odom s")[i])
        ca_p_real = self.connected_agent.x_real[i]
        ha_p_real = self.host_agent.x_real[i]
        ca_h_real = self.connected_agent.h_real[i]
        ha_h_real = self.host_agent.h_real[i]

        self.host_agent_trajectory = np.append(self.host_agent_trajectory, np.reshape(ha_p_real, (1, 3)), axis=0)
        self.connected_agent_trajectory = np.append(self.connected_agent_trajectory, np.reshape(ca_p_real, (1, 3)),
                                                    axis=0)

        self.connected_agent_heading.append(limit_angle(ca_h_real))

        ca_position_estimation = self.ukf.t_oi_sj[:3]
        self.error_ca_position.append(np.linalg.norm(ca_position_estimation - ca_p_real))
        self.ca_position_estimation = np.append(self.ca_position_estimation, np.reshape(ca_position_estimation, (1, 3)),
                                                axis=0)
        self.ca_heading_estimation.append(limit_angle(self.ukf.t_oi_sj[-1]))
        self.error_ca_heading.append(np.abs(limit_angle(ca_h_real - self.ukf.t_oi_sj[-1])))

        relative_heading_estimation = limit_angle(self.ukf.t_si_sj[3])
        relative_heading = limit_angle(ca_h_real - ha_h_real)
        self.error_relative_heading_est.append(np.abs(limit_angle(relative_heading_estimation - relative_heading)))

    def log_slam_data(self, i):
        """
        This module tries to estimate the relative estimated error of the real RPE vs if the SLAM would be used without correction.
        It does include error due to drift in orientation. Therefore the estimation of the slam error should be the same between 2 agents,
        however the stiamtion of the RPE error should be different, due to different drifts in own orientation.
        """
        ca_p_slam = self.connected_agent.x_slam[i]
        ca_p_real = self.connected_agent.x_real[i]
        ha_p_slam = self.host_agent.x_slam[i]
        ca_h_slam = self.connected_agent.h_slam[i]
        ca_h_real = self.connected_agent.h_real[i]
        ha_h_slam = self.host_agent.h_slam[i]

        relative_transformation_slam = (ca_p_slam - ha_p_slam)
        spherical_relative_transformation = cartesianToSpherical(relative_transformation_slam)
        spherical_relative_transformation[1] = limit_angle(spherical_relative_transformation[1] - ha_h_slam)
        relative_transformation_slam = sphericalToCartesian(spherical_relative_transformation)

        relative_transformation = sphericalToCartesian(self.spherical_relative_transformation[-1])

        self.error_relative_transformation_slam.append(
            np.linalg.norm(relative_transformation_slam - relative_transformation))
        relative_heading_slam = limit_angle(ca_h_slam - ha_h_slam)
        relative_heading = limit_angle(self.connected_agent.h_real[i] - self.host_agent.h_real[i])
        self.error_relative_heading_slam.append(np.abs(limit_angle(relative_heading_slam - relative_heading)))

        self.error_ca_position_slam.append(np.linalg.norm(ca_p_slam - ca_p_real))
        self.error_ca_heading_slam.append(np.abs(limit_angle(ca_h_real - ca_h_slam)))

    def plot_ukf_drift(self, ax):
        # plot drift of the position
        ax[0].set_ylabel("Position [m]")
        # ax[0].plot(self.error_ca_position, linestyle=self.relative_linestyle, color="darkgreen",
        #            label="ca position estimation error")
        ax[0].plot(self.error_relative_transformation_est, linestyle=self.relative_linestyle,
                   color=self.estimation_color, label="Error on position")
        # ax[0].plot(self.error_relative_transformation_slam, linestyle=self.relative_linestyle, color=self.slam_color,
        #            label="relative transformation slam")
        # ax[0].plot(self.sigma_x_ca_0, color=self.estimation_color, linestyle=self.stds_linestyle, label="std on the start position estimation")
        ax[0].plot(self.sigma_x_ca, color="red", linestyle=self.stds_linestyle,
                   label="Estimated std on the position by the UPF.")
        ax[0].grid(True)
        ax[1].legend()

        # plot drift of the heading
        ax[1].set_ylabel("Heading [(rad)]")
        ax[1].plot(self.error_relative_heading_est, linestyle=self.relative_linestyle,
                   color=self.estimation_color, label="Error on orientation")
        # ax[1].plot(self.error_relative_heading_slam, linestyle=self.relative_linestyle, color=self.slam_color)
        ax[1].plot(self.sigma_h_ca, color="red", linestyle=self.stds_linestyle,
                   label="Estimated std on the position by the UPF.")
        ax[1].grid(True)
        ax[1].legend()

    def create_3d_plot(self):
        if self.data_logged:
            fig = plt.figure()
            fig.set_size_inches(18.5, 10.5)
            fig.set_dpi(100)

            ax = plt.axes(projection="3d")
            self.plot_trajectory(self.host_agent_trajectory, ax, color="darkgreen",
                                 label="Host Agent real trajectory [m]")
            self.plot_trajectory(self.connected_agent_trajectory, ax, color="k",
                                 label="Connected Agent real trajectory [m]")
            self.plot_trajectory(self.ca_position_estimation, ax, color="r", linestyle="--",
                                 label="Estimated trajectory of the connected agent [m]")

            plt.title(" 3D trajectories \n" + self.name)
            ax.legend()

            if self.save_bool:
                save_name = self.prior_save_name + "_Trajectories_" + self.posterior_save_name

                plt.savefig(self.save_folder + save_name)

    #TODO: What is difference between self.estimated_ca_position and self.ca_position_estimation?
    def plot_ca_corrected_estimated_trajectory(self, ax, color="k", alpha=1, linestyle="--", marker="", label=None, i=-1, history=None):
        try:
            if history is None or history > i:
                self.plot_trajectory(self.estimated_ca_position[:i,:], ax, color, alpha, linestyle, marker, label)
            else:
                j = i - history
                if j < 0:
                    j = 0
                self.plot_trajectory(self.estimated_ca_position[j:i,:], ax, color, alpha, linestyle, marker, label)
        except IndexError:
            print("index error")
    def plot_ca_estimated_trajectory(self, ax, color="k", alpha=1, linestyle="--", marker="", label=None, i=-1):
        self.plot_trajectory(self.ca_position_estimation[:i,:], ax, color, alpha, linestyle, marker, label)

    def plot_trajectory(self, data, ax, color="k", alpha=1, linestyle="-", marker="", label=None):
        if self.data_logged:
            ax.plot3D(data[:, 0], data[:, 1], data[:, 2],
                      marker=marker, alpha=alpha, linestyle=linestyle, label=label, color=color)
            ax.plot3D(data[0, 0], data[0, 1], data[0, 2],
                      marker="o", alpha=alpha, color=color)
            ax.plot3D(data[-1, 0], data[-1, 1], data[-1, 2],
                      marker="x", alpha=alpha, color=color)

    def plot_relative_transformation(self, ax, color="k", alpha=1, linestyle="-", marker="", label="Real"):
        """
        ax should be a 4x1 array of axes.
        """
        if self.data_logged:
            self.plot_3d_data_on_1d_plots(ax[:3], self.spherical_relative_transformation, color=color,
                                          linestyle=linestyle, alpha=alpha, marker=marker, labels=[label] * 3)
            ax[3].plot(self.connected_agent_heading, color=color, linestyle=linestyle, alpha=alpha, marker=marker,
                       label=label)

    def plot_relative_transformation_estimation(self, ax, color="crimson", alpha=1, linestyle="--", marker="",
                                                label="Estimated"):
        """
        ax should be a 4x1 array of axes.
        """
        if self.data_logged:
            if ax[0].get_ylabel() == "":
                ylabel = ["Relative distance [m]", "Relative azimuth [rad]", "Relative altitude [rad]",
                          "Relative heading [rad]"]
                for i in range(4):
                    ax[i].set_ylabel(ylabel[i], color=color)
                    ax[i].tick_params(axis='y', labelcolor=color)

                self.plot_3d_data_on_1d_plots(ax[:3], self.spherical_estimated_relative_transformation, color=color,
                                              linestyle=linestyle, alpha=alpha, marker=marker, labels=[label] * 3)
                ax[3].plot(self.ca_heading_estimation, color=color, linestyle=linestyle, alpha=alpha, marker=marker,
                           label=label)

                self.plot_3d_data_on_1d_plots(ax, self.spherical_estimated_relative_start_position, color=color,
                                              linestyle=":", alpha=alpha, marker=marker,
                                              labels=[label + " start pose"] * 4)

    def plot_relative_transformation_error(self, ax, color="steelblue", label="Error"):
        if self.data_logged:
            if ax[0].get_ylabel() == "":
                ylabel = ["[m]", "[rad]", "[rad]", "[rad]"]
                for i in range(4):
                    ax[i].set_ylabel("std and error " + ylabel[i], color=color)
                    ax[i].tick_params(axis='y', labelcolor=color)

            self.plot_3d_data_on_1d_plots(ax[:3], self.error_spherical_relative_transformation_estimation, color=color,
                                          linestyle="-", labels=[label] * 3)
            ax[3].plot(self.error_ca_heading, color=color, linestyle="-", label=label)

            self.plot_3d_data_on_1d_plots(ax, self.stds[:, :4], color=color, linestyle=":",
                                          marker="", labels=["std start pose"] * 4)

            # ax[0].plot(self.stds_trajectory, color=color, marker="^", linestyle=":", label="std trajectory")
            ax[3].plot(self.stds[:, 4], color=color, marker="^", linestyle=":", label="std heading")

    def plot_position_error(self, ax, color="crimson"):
        if self.data_logged:
            if ax.get_ylabel() == "":
                ax.set_ylabel("Error [m]", color=color)
                ax.tick_params(axis='y', labelcolor=color)

            ax.plot(self.error_ca_position, color=color, linestyle="-", label="Error on absolute position estimation")
            ax.plot(self.error_relative_transformation_est, color=color, linestyle="--",
                    label="Error on the relative transformation estimation")
            ax.plot(self.connected_agent.x_error[1:], color="k", linestyle="-", label="Error from slam")

    def plot_position_stds(self, ax, color="steelblue"):
        if self.data_logged:
            ax.plot(self.sigma_x_ca_0, color=color, linestyle="-", label="std on the start position estimation")
            ax.plot(self.sigma_x_ca, color=color, linestyle="--", label="std on the relative transformation estimation")

            axt = ax.twinx()
            if axt.get_ylabel() == "":
                axt.set_ylabel("Relative error [%]", color="k")
            axt.plot(self.sigma_x_ca_r, color="k", linestyle=":",
                     label="relative std on the relative transformation estimation")

    def plot_ukf_states(self):
        colums = 2
        rows = int(np.ceil(self.kf_variables / colums))
        _, ax = plt.subplots(rows, colums)

        labels = ["r", "az", "alt", "h_0", "x", "y", "z", "h", "h_drift"]
        for row in range(rows):
            for colum in range(colums):
                try:
                    i = colum + row * colums
                    ax[row, colum].plot(self.x[:, i], label=labels[i])
                    ax[row, colum].legend(loc="upper right")
                    twax = ax[row, colum].twinx()
                    twax.plot(self.stds[:, i], linestyle=":", label="std")
                except IndexError:
                    pass

    def plot_error_graph(self):
        if self.data_logged:

            fig, ax = plt.subplots(4, 2)
            fig.set_size_inches(18.5, 10.5)
            fig.set_dpi(100)
            fig.suptitle("Error and Variance graph: \n " + self.name)

            self.plot_relative_transformation(ax[:4, 0], label="Real")
            self.plot_relative_transformation_estimation(ax[:4, 0], label="Estimated")

            error_ax = [ax[i, 0].twinx() for i in range(4)]
            self.plot_relative_transformation_error(error_ax, label="Error")

            for axis in error_ax:
                axis.legend(loc="upper right")
                axis.grid(True)
            for axis in ax[:, 0]:
                axis.legend(loc="upper left")
                axis.grid(True)

            # self.plot_position_error(ax[0, 1])
            # self.plot_position_stds(ax[0, 1])

            self.plot_ukf_drift(ax[:2, 1])

            label = ["r", "az", "alt", "h_0", "x", "y", "z", "h"]
            for i in range(8):
                ax[2, -1].plot(self.x[:, i], label=label[i])

            # ax[1, -1].plot(self.ca_s, color="black", label="Travelled distance of the connected agent [m]")
            # ax[1, -1].plot(self.ha_s, color="green", label="Traveled distance of the host agent [m]")

            ax[3, -1].plot(self.likelihood, color="black", label="Likelihood")
            ax[3, -1].plot(self.weight, color="red", label="Weight")
            ax[3, -1].plot(self.los_state, color="Green", label="LOS state")

            for i in ax[:, -1]:
                # for i in a:
                i.legend()
                i.grid(True)
            if self.save_bool:
                save_name = self.prior_save_name + "_Error_and_variance_graph_" + self.posterior_save_name
                plt.savefig(self.save_folder + save_name)




    def plot_3d_data_on_1d_plots(self, ax, data, color, labels, alpha=1, marker="", linestyle="-"):
        for i, a in enumerate(ax):
            a.plot(data[:, i], color=color, linestyle=linestyle, alpha=alpha, marker=marker, label=labels[i])
            # a.legend()
            a.grid(True)

    def copy(self, ukf):
        copyDL = Datalogger(self.host_agent, self.connected_agent, ukf)

        copyDL.i = copy.deepcopy(self.i)
        copyDL.host_agent_trajectory = copy.deepcopy(self.host_agent_trajectory)
        copyDL.connected_agent_trajectory = copy.deepcopy(self.connected_agent_trajectory)
        copyDL.connected_agent_heading = copy.deepcopy(self.connected_agent_heading)

        copyDL.ca_position_estimation = copy.deepcopy(self.ca_position_estimation)
        copyDL.error_ca_position = copy.deepcopy(self.error_ca_position)

        copyDL.ca_heading_estimation = copy.deepcopy(self.ca_heading_estimation)
        copyDL.error_ca_heading = copy.deepcopy(self.error_ca_heading)
        copyDL.error_ca_position_slam = copy.deepcopy(self.error_ca_position_slam)
        copyDL.error_ca_heading_slam = copy.deepcopy(self.error_ca_heading_slam)

        copyDL.spherical_estimated_relative_transformation = copy.deepcopy(
            self.spherical_estimated_relative_transformation)
        copyDL.spherical_relative_transformation = copy.deepcopy(self.spherical_relative_transformation)
        copyDL.error_spherical_relative_transformation_estimation = copy.deepcopy(
            self.error_spherical_relative_transformation_estimation)

        copyDL.error_relative_transformation_est = copy.deepcopy(self.error_relative_transformation_est)
        copyDL.error_relative_heading_est = copy.deepcopy(self.error_relative_heading_est)
        copyDL.error_relative_heading_slam = copy.deepcopy(self.error_relative_heading_slam)
        copyDL.error_relative_transformation_slam = copy.deepcopy(self.error_relative_transformation_slam)

        copyDL.spherical_relative_start_position = copy.deepcopy(self.spherical_relative_start_position)
        copyDL.spherical_estimated_relative_start_position = copy.deepcopy(
            self.spherical_estimated_relative_start_position)

        copyDL.x = copy.deepcopy(self.x)
        copyDL.stds = copy.deepcopy(self.stds)
        copyDL.stds_trajectory = copy.deepcopy(self.stds_trajectory)
        copyDL.sigma_x_ca_0 = copy.deepcopy(self.sigma_x_ca_0)
        copyDL.sigma_x_ca = copy.deepcopy(self.sigma_x_ca)
        copyDL.sigma_x_ca_r = copy.deepcopy(self.sigma_x_ca_r)

        copyDL.los_state = copy.deepcopy(self.los_state)
        copyDL.likelihood = copy.deepcopy(self.likelihood)
        copyDL.weight = copy.deepcopy(self.weight)

        return copyDL

    def save_as_h5(self, name=None):
        if not name:
            name = self.prior_save_name + "_data_" + self.posterior_save_name
        hf = h5py.File(self.save_folder + name, 'w')
        hf.create_dataset("ha_trajectory", data=self.host_agent_trajectory.astype(np.float64))
        hf.create_dataset("ca_trajectory", data=self.connected_agent_trajectory.astype(np.float64))
        hf.create_dataset("ca_heading", data=np.array(self.connected_agent_heading).astype(np.float64))
        hf.create_dataset("likelihood", data=np.array(self.likelihood).astype(np.float64))
        hf.create_dataset("weight", data=np.array(self.weight).astype(np.float64))
        hf.create_dataset("stds", data=np.array(self.stds).astype(np.float64))
        hf.create_dataset("estimated_trjectory", data=self.ca_position_estimation.astype(np.float64))
        hf.create_dataset("estimated_heading", data=np.array(self.ca_heading_estimation).astype(np.float64))
        hf.create_dataset("error_heading", data=np.array(self.error_ca_heading).astype(np.float64))
        hf.create_dataset("error_pos_spherical",
                          data=self.error_spherical_relative_transformation_estimation.astype(np.float64))
        hf.create_dataset("estimated_pos_spherical",
                          data=self.spherical_estimated_relative_transformation.astype(np.float64))
        hf.create_dataset("real_pos_spherical", data=self.spherical_relative_transformation.astype(np.float64))
        string_dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("name", data=np.array([self.name], 'S4'), dtype=string_dt)
        hf.create_dataset("now", data=np.array(
            [self.now.year, self.now.month, self.now.day, self.now.hour, self.now.minute, self.now.second]).astype(
            np.float))

        hf.close()


def load_dataFile(filename):
    hf = h5py.File(filename, 'r')
    dl = Datalogger(host_agent=None, connected_agent=None, name="", ukf=None)
    dl.host_agent_trajectory = np.array(hf.get("ha_trajectory"))
    dl.connected_agent_trajectory = np.array(hf.get("ca_trajectory"))
    dl.connected_agent_heading = np.array(hf.get("ca_heading"))
    dl.likelihood = np.array(hf.get("likelihood"))
    dl.weight = np.array(hf.get("weight"))
    dl.stds = np.array(hf.get("stds"))
    dl.estimatedPose = np.array(hf.get("estimated_trjectory"))
    dl.estimated_heading = np.array(hf.get("estimated_heading"))
    dl.error_heading = np.array(hf.get("error_heading"))
    dl.error_pos_spherical = np.array(hf.get("error_pos_spherical"))
    dl.estimatedPose_Spherical = np.array(hf.get("estimated_pos_spherical"))
    dl.realPose_Spherical = np.array(hf.get("real_pos_spherical"))
    dl.name = str(np.array(hf.get("name"))[0])
    print(dl.name)
    dl.data_logged = True
    now = hf.get("now")
    dl.now = datetime(year=int(now[0]), month=int(now[1]), day=int(now[2]),
                      hour=int(now[3]), minute=int(now[4]), second=int(now[5]))
