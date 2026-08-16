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
import numpy as np
import copy

from upf_rte.ParticleFilter.ModifiedUKF import ModifiedUnscentedKalmanFilter, ModifiedMerweScaledSigmaPoints
from upf_rte.UtilityCode.utility_fuctions import cartesianToSpherical, sphericalToCartesian, limit_angle, \
    get_4d_rot_matrix, transform_matrix, inv_transformation_matrix, \
    get_states_of_transform, get_covariance_of_transform

import upf_rte.UtilityCode.Transformation_Matrix_Fucntions as TMF

ANGLE_STATE_INDICES = (1, 2, 3, 7, 8)


def subtract(x, y):
    dx = x - y
    for index in ANGLE_STATE_INDICES:
        if index < len(dx):
            dx[index] = limit_angle(dx[index])
    return dx


def state_mean(sigmas, weights):
    mean = np.dot(weights, sigmas)
    for index in ANGLE_STATE_INDICES:
        if index < mean.shape[0]:
            mean[index] = np.arctan2(
                np.sum(weights * np.sin(sigmas[:, index])),
                np.sum(weights * np.cos(sigmas[:, index])),
            )
    return mean


def subtract_spherical(x, y):
    # Does not work because a negative R becomes - pi in azimuth.
    dx = np.zeros(x.shape)
    dx[:3] = cartesianToSpherical(sphericalToCartesian(x[:3]) - sphericalToCartesian(y[:3]))
    dx[3] = limit_angle(x[3] - y[3])

    return dx


class TargetTrackingUKF:
    """
    Todo: Add documentation
    """

    def __init__(self, x_ha_0=np.zeros(4), weight=1., drift_correction_bool=True, parent=None):

        # ---- Input variables:
        # Dt_sj : Delta position odometry of the connected agent expressed in his local odometry frame.
        self.Dt_sj = np.zeros(4)
        # q_ca: noise matrix of the odometry of the connected agent.
        self.q_ca = np.zeros((4, 4))
        #  uwb_measurement : Measurement of the UWB.py sensor.
        self.uwb_measurement = 0
        #
        # # ---- Internal variables:
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
        self.P_t_si_sj = np.zeros((4, 4))
        self.t_cij_cji = np.zeros(4)
        self.t_oi_sj = np.zeros(4)
        self.t_oi_cji = np.zeros(4)
        self.t_cji_sj = np.zeros(4)
        self.D_t_sj = np.zeros(4)  # Apperently not uses, self.Dt_sj is used.

        # transformation from VIO reference frame to UWB antenna
        self.t_si_uwb = np.zeros(4)
        self.t_sj_uwb = np.zeros(4)

        # ---- Kalman filter variables:
        # kf_variables = r_0, theta_0, phi_0, h_ca_0,  x_ca_odom, y_ca_odom, z_ca_odom, h_ca_odom, h_ha_drift
        # r_0, theta_0, phi_0: Relative position of the connected agent wrt the host agent at connection time expressed in spherical coordinates.
        # h_ca_0: Heading of the connected agent at connection time in the reference frame of the host agent at connection time.
        # x_ca_odom, y_ca_odom, z_ca_odom, h_ca_odom : Odometry of the connected agent wrt the position of the connected agent at connection time.
        # h_ha_drift : Drift of the host agent heading.
        self.kf_variables = 9
        self.kf: ModifiedUnscentedKalmanFilter = None
        self.weight = weight
        self.dt = 0.1 # has to be set for the UKF, however is not used in the Prediction function since we use the processed values.
        self.alpha = 1
        self.kappa = -1
        self.beta = 2

        # ---- Kalman filter state:
        self.P_x_ca = np.zeros((4, 4))
        self.sigma_dh_ca = 0
        self.sigma_dx_ca = 0

        self.parent = parent

        # ---- Data logging variables:
        self.time_i = None
        # self.datalogger = None

    # -------------------------------------------------------------------------------------- #
    # --- Initialization functions:
    # -------------------------------------------------------------------------------------- #
    def set_ukf_properties(self, kappa=-1, alpha=1, beta=2):
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        points = ModifiedMerweScaledSigmaPoints(self.kf_variables, alpha, beta, kappa, subtract=subtract)
        self.kf = ModifiedUnscentedKalmanFilter(dim_x=self.kf_variables, dim_z=1, dt=self.dt, fx=self.fx, hx=self.hx,
                                                points=points, residual_x=subtract, x_mean_fn=state_mean)

    def set_initial_state(self, s_j, sigma_t):
        s_cor = self.calculate_initial_state(s_j)
        self.kf.x = np.array([s_cor[0], s_cor[1], s_cor[2], s_cor[3], 0, 0, 0, 0, 0])
        self.kf.P = np.diag(
            [(sigma_t[0]) ** 2, sigma_t[1] ** 2, sigma_t[2] ** 2, sigma_t[3] ** 2, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
        self.calculate_x_ca()
        self.calculate_P_x_ca()

    def calculate_initial_state(self, s):
        # Due to UWB having extrinsicity, the initial state of the connected agent has to be calculated using this and
        # The measured distance to the connected agent.
        t_cij = np.zeros(4)
        t_cij[:3] = sphericalToCartesian(s[:3])  # Vector from agent i to j in the UWB frame
        t_cij[3] = s[3]
        T_cij = TMF.transformation_matrix_from_4D_t(t_cij)
        T_si_uwb = TMF.transformation_matrix_from_4D_t(self.t_si_uwb)
        T_sj_uwb = TMF.transformation_matrix_from_4D_t(self.t_sj_uwb)
        T_ij = T_si_uwb @ T_cij @ TMF.inv_transformation_matrix(T_sj_uwb)
        t_ij = TMF.get_4D_t_from_matrix(T_ij)
        r = TMF.get_translation(TMF.inv_transformation_matrix(T_si_uwb) @ T_ij @ T_sj_uwb)
        r = np.linalg.norm(r)
        s_ij = np.zeros(4)
        s_ij[:3] = cartesianToSpherical(t_ij[:3])
        s_ij[3] = t_ij[3]
        return s_ij

    def set_uwb_extrinsicity(self, t_si_uwb, t_sj_uwb):
        self.t_si_uwb = t_si_uwb
        self.t_sj_uwb = t_sj_uwb

    # -------------------------------------------------------------------------------------- #
    # --- Filter functions: Main function
    # -------------------------------------------------------------------------------------- #
    def run_filter(self, dt_j, q_j, t_i, P_i, d_ij, sig_d, bool_drift=True, update_bool=True, time_i=None):
        self.time_i = time_i
        self.t_oi_si = t_i
        self.set_host_agent_uncertainty(P_i)
        self.predict(dt_j, q_j)
        if update_bool:
            self.update(d_ij, t_i, P_i, sig_d, bool_drift)
        self.calculate_x_ca()
        self.calculate_P_x_ca()
        return None

    # -------------------------------------------------------------------------------------- #
    # --- Prediction functions
    # -------------------------------------------------------------------------------------- #
    def predict(self, dx_ca, q=None):
        dx_ca = np.asarray(dx_ca, dtype=np.float64)
        if dx_ca.shape != (4,):
            raise ValueError("dx_ca must be a 4D displacement vector [x, y, z, heading]")
        if q is None:
            q = np.diag([self.sigma_dx_ca ** 2, self.sigma_dx_ca ** 2, self.sigma_dx_ca ** 2, self.sigma_dh_ca ** 2])
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (4, 4):
            raise ValueError("q must be a 4x4 odometry covariance matrix")
        DT_sj = transform_matrix(self.Dt_sj).astype(np.float64)
        f = get_covariance_of_transform(DT_sj).astype(np.float64)
        self.Dt_sj = self.Dt_sj + f @ dx_ca
        self.q_ca = self.q_ca + f @ q @ f.T
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
            # T_D_inv = transform_matrix(np.array([0, 0, 0, -x[-1]]))
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

        if self.drift_correction_bool:
            x[-2] = x[-2] - x[-1]

        # x[-2] = x[-2] - x[-1]
        # x[4:-1] = x_odom[:3]
        return x

    # -------------------------------------------------------------------------------------- #
    # --- Update functions
    # -------------------------------------------------------------------------------------- #
    def _state_to_transforms(self, x, t_oi_si=None, include_pending_dt=False):
        if t_oi_si is None:
            t_oi_si = self.t_oi_si
        T_oi_cij = transform_matrix(self.t_oi_cij)
        t_cij_cji = np.append(sphericalToCartesian(x[:3]), np.array([x[3]]))
        T_cij_cji = transform_matrix(t_cij_cji)
        T_cji_sj = transform_matrix(x[4:-1])
        T_si_oi = inv_transformation_matrix(t_oi_si)
        T_si_sj = T_si_oi @ T_oi_cij @ T_cij_cji @ T_cji_sj
        T_oi_sj = T_oi_cij @ T_cij_cji @ T_cji_sj
        if include_pending_dt:
            T_sj_sj = transform_matrix(self.Dt_sj)
            T_si_sj = T_si_sj @ T_sj_sj
            T_oi_sj = T_oi_sj @ T_sj_sj
        return T_si_sj, T_oi_sj, T_cij_cji, T_cji_sj

    def _uwb_vector_from_state(self, x, t_oi_si=None):
        T_si_sj, _, _, _ = self._state_to_transforms(x, t_oi_si=t_oi_si)
        T_uwbi_uwbj = inv_transformation_matrix(self.t_si_uwb) @ T_si_sj @ transform_matrix(self.t_sj_uwb)
        return get_states_of_transform(T_uwbi_uwbj)[:3]

    def _range_from_state(self, x, t_oi_si=None):
        return np.linalg.norm(self._uwb_vector_from_state(x, t_oi_si=t_oi_si))

    def _range_jacobian_host_pose(self, x_ha):
        jacobian = np.zeros(4)
        epsilons = np.array([1e-5, 1e-5, 1e-5, 1e-6])
        for index, epsilon in enumerate(epsilons):
            x_plus = np.array(x_ha, dtype=float).copy()
            x_minus = np.array(x_ha, dtype=float).copy()
            x_plus[index] += epsilon
            x_minus[index] -= epsilon
            jacobian[index] = (
                self._range_from_state(self.kf.x, t_oi_si=x_plus)
                - self._range_from_state(self.kf.x, t_oi_si=x_minus)
            ) / (2 * epsilon)
        return jacobian

    def calculate_r(self, sigma_uwb, P_x_ha=None):
        variance = sigma_uwb ** 2
        if P_x_ha is not None:
            P_x_ha = np.asarray(P_x_ha, dtype=np.float64)
            if P_x_ha.shape != (4, 4):
                raise ValueError("P_x_ha must be a 4x4 covariance matrix")
            jacobian = self._range_jacobian_host_pose(self.t_oi_si)
            variance += float(jacobian @ P_x_ha @ jacobian.T)
        self.kf.R = np.diag([max(variance, 0.0)])

    def update(self, z, x_ha, P_x_ha, sigma_uwb, bool_drift=True):
        # Prediction step
        self.calculate_q()
        self.kf.predict()
        # Update step
        self.uwb_measurement = z
        self.calculate_r(sigma_uwb, P_x_ha)
        self.kf.update(np.array([z]))
        # Reset
        self.t_oi_si_prev = x_ha
        self.reset_prediction_states()
        self.set_residual_drift(bool_drift)

    def hx(self, x):
        return np.array([self._range_from_state(x)])

    # -------------------------------------------------------------------------------------- #
    # --- Postprocessing functions
    # -------------------------------------------------------------------------------------- #
    def calculate_x_ca(self):
        T_si_sj, T_oi_sj, T_cij_cji, T_cji_sj = self._state_to_transforms(
            self.kf.x, include_pending_dt=True
        )
        self.t_si_sj = get_states_of_transform(T_si_sj)
        self.t_oi_sj = get_states_of_transform(T_oi_sj)
        T_oi_cji = transform_matrix(self.t_oi_cij) @ T_cij_cji
        self.t_oi_cji = get_states_of_transform(T_oi_cji)
        self.t_cji_sj = get_states_of_transform(T_cji_sj)
        return None

    def calculate_P_x_ca(self):
        sigmas = self.kf.points_fn.sigma_points(self.kf.x, self.kf.P)
        t_sigmas = np.array([
            get_states_of_transform(
                self._state_to_transforms(sigma, include_pending_dt=True)[0]
            )
            for sigma in sigmas
        ], dtype=np.float64)

        mean = np.dot(self.kf.Wm, t_sigmas)
        mean[3] = np.arctan2(
            np.sum(self.kf.Wm * np.sin(t_sigmas[:, 3])),
            np.sum(self.kf.Wm * np.cos(t_sigmas[:, 3])),
        )
        covariance = np.zeros((4, 4), dtype=np.float64)
        for weight, sigma_t in zip(self.kf.Wc, t_sigmas):
            residual = sigma_t - mean
            residual[3] = limit_angle(residual[3])
            covariance += weight * np.outer(residual, residual)
        if np.any(self.q_ca):
            covariance += self.q_ca.astype(np.float64)
        covariance = 0.5 * (covariance + covariance.T)

        self.P_t_si_sj = covariance
        self.P_x_ca = covariance.copy()
        self.sigma_x_ca_0 = np.linalg.norm(np.sqrt(np.maximum(np.diag(self.kf.P[:3, :3]), 0)))
        self.sigma_x_ca = np.linalg.norm(np.sqrt(np.maximum(np.diag(self.P_t_si_sj[:3, :3]), 0)))
        self.sigma_h_ca = np.sqrt(max(self.P_t_si_sj[-1, -1], 0))

    # -------------------------------------------------------------------------------------- #
    # --- Residual drift functions
    # -------------------------------------------------------------------------------------- #
    def set_host_agent_uncertainty(self, P_x_ha):
        T_oi_si = transform_matrix(self.t_oi_si_prev)
        f = get_covariance_of_transform(T_oi_si)
        # f = get_4d_rot_matrix(self.x_ha_prev[-1]).astype(np.float64)
        self.q_ha = self.q_ha + f @ P_x_ha.astype(np.float64) @ f.T

    def set_residual_drift(self, bool_drift=True):
        # Should be called at the PF level in case there is residual drift.
        self.kf.P[:, -1] = np.zeros(self.kf_variables)
        self.kf.P[-1, :] = np.zeros(self.kf_variables)
        self.kf.P[-1, -1] = 1e-20
        self.kf.x[-1] = 0
        if self.drift_correction_bool:
            self.q_ca[:3, :3] = self.q_ca[:3, :3] + self.q_ha[:3, :3]
            self.kf.P[-1, -1] = self.kf.P[-1, -1] + self.q_ha[-1, -1]
            if np.sqrt(self.kf.P[-1, -1]) > np.pi:
                self.kf.P[-1, -1] = np.pi ** 2
        self.q_ha = np.zeros((4, 4))

    # -------------------------------------------------------------------------------------- #
    # --- Copy functions
    # -------------------------------------------------------------------------------------- #
    def copy(self):
        copiedUKF = TargetTrackingUKF(x_ha_0=self.t_oi_cij, weight=self.weight,
                                      drift_correction_bool=self.drift_correction_bool, parent=self)
        copiedUKF.set_ukf_properties(self.kappa, self.alpha, self.beta)

        copiedUKF.kf.x = copy.deepcopy(self.kf.x)
        copiedUKF.kf.P = copy.deepcopy(self.kf.P)
        copiedUKF.kf.R = copy.deepcopy(self.kf.R)
        copiedUKF.kf.Q = copy.deepcopy(self.kf.Q)
        copiedUKF.kf._likelihood = copy.deepcopy(self.kf.likelihood)

        # Transformations.
        copiedUKF.t_oi_si = copy.deepcopy(self.t_oi_si)
        copiedUKF.t_oi_si_prev = copy.deepcopy(self.t_oi_si_prev)
        copiedUKF.t_oi_cij = copy.deepcopy(self.t_oi_cij)
        copiedUKF.t_si_sj = copy.deepcopy(self.t_si_sj)
        copiedUKF.t_cij_cji = copy.deepcopy(self.t_cij_cji)
        copiedUKF.t_oi_sj = copy.deepcopy(self.t_oi_sj)
        copiedUKF.t_oi_cji = copy.deepcopy(self.t_oi_cji)
        copiedUKF.t_cji_sj = copy.deepcopy(self.t_cji_sj)
        copiedUKF.D_t_sj = copy.deepcopy(self.D_t_sj)

        copiedUKF.t_si_uwb = copy.deepcopy(self.t_si_uwb)
        copiedUKF.t_sj_uwb = copy.deepcopy(self.t_sj_uwb)


        copiedUKF.Dt_sj = copy.deepcopy(self.Dt_sj)
        copiedUKF.q_ca = copy.deepcopy(self.q_ca)
        copiedUKF.uwb_measurement = copy.deepcopy(self.uwb_measurement)

        copiedUKF.q_ha = copy.deepcopy(self.q_ha)

        copiedUKF.sigma_x_ca_0 = copy.deepcopy(self.sigma_x_ca_0)
        copiedUKF.sigma_x_ca = copy.deepcopy(self.sigma_x_ca)
        copiedUKF.sigma_h_ca = copy.deepcopy(self.sigma_h_ca)
        copiedUKF.sigma_dh_ca = copy.deepcopy(self.sigma_dh_ca)
        copiedUKF.sigma_dx_ca = copy.deepcopy(self.sigma_dx_ca)

        copiedUKF.P_t_si_sj = copy.deepcopy(self.P_t_si_sj)
        copiedUKF.P_x_ca = copy.deepcopy(self.P_x_ca)


        copiedUKF.time_i = copy.deepcopy(self.time_i)
        copiedUKF.weight = copy.deepcopy(self.weight)

        return copiedUKF
