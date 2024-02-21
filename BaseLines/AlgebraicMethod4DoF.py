#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 2 13:28:51 2023

@author: yuri
"""

import numpy as np
from Simulation.RobotClass import NewRobot
from UtilityCode.utility_fuctions import get_4d_rot_matrix, limit_angle, sphericalToCartesian, cartesianToSpherical
import matplotlib.pyplot as plt
from scipy.linalg import null_space


def update_quaternion(q, w, dt):
    q = q + 0.5 * dt * np.array([w[0], w[1], w[2], 0]) @ q
    return q / np.linalg.norm(q)


def get_quaternion_rotation_matrix(q):
    q11 = q[0] ** 2
    q22 = q[1] ** 2
    q33 = q[2] ** 2
    q44 = q[3] ** 2
    q12 = q[0] * q[1]
    q13 = q[0] * q[2]
    q14 = q[0] * q[3]
    q23 = q[1] * q[2]
    q24 = q[1] * q[3]
    q34 = q[2] * q[3]

    return np.array([[1 - 2 * (q22 + q33), 2 * (q12 - q34), 2 * (q13 + q24)],
                     [2 * (q12 + q34), 1 - 2 * (q11 + q33), 2 * (q23 - q14)],
                     [2 * (q13 - q24), 2 * (q23 + q14), 1 - 2 * (q11 + q22)]])


class AlgebraicMethod6DoF:
    # Unfinished found the error in the 4dof case.
    def __init__(self, d0, x_ha=np.zeros(7)):
        self.eps = [1]
        self.d = [d0]
        self.x_ha_0 = x_ha
        self.x_ha_odom = np.zeros((1, 7))  # x,y, z, q1, q2, q3, q4
        self.x_ca_odom = np.zeros((1, 7))

    def get_update(self, d, v_ha, v_ca, dt):
        x_ha_odom = self.x_ha_odom[-1, :3] + get_quaternion_rotation_matrix(self.x_ha_odom[-1, 3:]) @ v_ha[:3] * dt
        x_ca_odom = self.x_ca_odom[-1, -3] + get_quaternion_rotation_matrix(self.x_ca_odom[-1, 3:]) @ v_ca[:3] * dt
        q_ha_odom = update_quaternion(self.x_ha_odom[-1, 3:], v_ha[3:], 1)
        q_ca_odom = update_quaternion(self.x_ca_odom[-1, 3:], v_ca[3:], 1)

        eps = 0.5 * (self.d[0] ** 2 + (x_ha_odom.T @ x_ha_odom) + (x_ca_odom.T @ x_ca_odom) - d ** 2)

        self.x_ha_odom = np.append(self.x_ha_odom, np.hstack((x_ha_odom, q_ha_odom)).reshape((1, 4)), axis=0)
        self.x_ca_odom = np.append(self.x_ca_odom, np.hstack((x_ca_odom, q_ha_odom)).reshape((1, 4)), axis=0)
        self.eps.append(eps)
        self.d.append(d)

    def find_relative_pose(self):
        A = np.zeros((1, 6))
        M = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 1]])
        for i in range(1, len(self.eps)):
            x2_k = self.x_ca_odom[i, 0]
            y2_k = self.x_ca_odom[i, 1]
            z2_k = self.x_ca_odom[i, 2]

            M_k = self.x_ha_odom[i, :3].T @ np.array(
                [[x2_k, 2 * y2_k, 2 * z2_k, 0, - x2_k, -2 * z2_k, -x2_k, 2 * y2_k, x2_k],
                 [-y2_k, 2 * x2_k, 0, 2 * z2_k, y2_k, 2 * z2_k, 0, -y2_k, -2 * x2_k, y2_k],
                 [-z2_k, 0, 2 * x2_k, -2 * y2_k, -z2_k, 2 * y2_k, 2 * x2_k, z2_k, 0, z2_k]])
            M = np.append(M, M_k.reshape(1, *M_k.shape), axis=0)

            A_k = np.hstack((self.x_ha_odom[i, :3].T, self.x_ca_odom[i, :3].T))
            A = np.append(A, A_k.reshape(1, *A_k.shape), axis=0)
        eps = -np.array(self.eps).reshape(-1, 1)
        Mat = np.concatenate((eps, M, A), axis=1)
        # M_kernel = Mat.T @ Mat
        # q, r = np.linalg.qr(M_kernel)


class AlgebraicMethod4DoF:
    """
    THis class implements the algebraic method for finding the relative position of another agent.
    THe method is described in the paper:
        N. Trawny, X. S. Zhou, K. Zhou, and S. I. Roumeliotis, “Interrobot transformations in 3-d,”
        IEEE Transactions on Robotics, vol. 26, no. 2, pp. 226–243, 2010.

    The method is adapted to the 4-dof case.
    """

    def __init__(self, d0, x_ha=np.zeros(4), sigma_uwb=0.1):
        self.eps = [1]
        self.d = [d0]
        self.p = []
        self.x_ha_0 = x_ha
        self.x_ha_odom = np.zeros((1, 4))
        self.x_ca_odom = np.zeros((1, 4))
        self.P_ha = np.zeros((1, 4, 4))
        self.P_ca = np.zeros((1, 4, 4))
        self.R = np.array([sigma_uwb ** 2])
        self.x_ca_0_alg = np.zeros(4)
        self.x_ca_0 = np.zeros(4)
        self.x_ca_r = np.zeros(4)
        self.x_ha_r = np.zeros(4)

        self.x_ca_r_alg = np.zeros(4)

        self.wls_bool = False
        # Logging variables:
        self.logging = False
        self.logger = None

        # Debug variables:
        self.debug_bool = False

    def init_logging(self, logger):
        self.logging = True
        self.logger = logger

    def get_update(self, d, dx_ha, dx_ca, q_ha=np.zeros((4, 4)), q_ca=np.zeros((4, 4))):
        c_ha = get_4d_rot_matrix(self.x_ha_odom[-1, -1])
        c_ca = get_4d_rot_matrix(self.x_ca_odom[-1, -1])

        # Calculate the state
        x_ha_odom = self.x_ha_odom[-1] + c_ha @ dx_ha
        x_ca_odom = self.x_ca_odom[-1] + c_ca @ dx_ca
        self.x_ha_odom = np.append(self.x_ha_odom, x_ha_odom.reshape((1, 4)), axis=0)
        self.x_ca_odom = np.append(self.x_ca_odom, x_ca_odom.reshape((1, 4)), axis=0)

        # Calculate the uncertainty
        P_ha = self.P_ha[-1] + c_ha @ q_ha @ c_ha.T
        P_ca = self.P_ca[-1] + c_ca @ q_ca @ c_ca.T
        self.P_ha = np.append(self.P_ha, P_ha.reshape((1, 4, 4)), axis=0)
        self.P_ca = np.append(self.P_ca, P_ca.reshape((1, 4, 4)), axis=0)

        # Calculate the error.
        eps = 0.5 * (self.d[0] ** 2 + (x_ha_odom[:3].T @ x_ha_odom[:3]) + (x_ca_odom[:3].T @ x_ca_odom[:3]) - d ** 2)
        self.eps.append(eps)
        self.d.append(d)
        if len(self.eps) > 10:
            self.find_relative_pose()
            #Removes the first measurment from the list.


    def trim_to_latest_measurements(self):
        if len(self.eps) > 10:
            self.eps.pop(1)
            self.d.pop(1)
            self.x_ha_odom = np.delete(self.x_ha_odom, 1, axis=0)
            self.x_ca_odom = np.delete(self.x_ca_odom, 1, axis=0)
            self.P_ha = np.delete(self.P_ha, 1, axis=0)
            self.P_ca = np.delete(self.P_ca, 1, axis=0)


    def find_relative_pose(self):
        A = np.zeros((1, 6))
        M = np.array([[1, 0, 1]])
        for i in range(1, len(self.eps)):
            x2_k = self.x_ca_odom[i, 0]
            y2_k = self.x_ca_odom[i, 1]
            z2_k = self.x_ca_odom[i, 2]

            M_k = self.x_ha_odom[i, :3].T @ np.array(
                [[-x2_k, 2 * y2_k, x2_k],
                 [-y2_k, -2 * x2_k, y2_k],
                 [z2_k, 0, z2_k]])
            M = np.append(M, M_k.reshape(1, *M_k.shape), axis=0)

            A_k = np.hstack((self.x_ha_odom[i, :3].T, self.x_ca_odom[i, :3].T))
            A = np.append(A, A_k.reshape(1, *A_k.shape), axis=0)

        eps = np.array(self.eps).reshape(-1, 1)
        Mat = np.concatenate((M, A), axis=1).astype(np.float64)
        try:
            x = np.linalg.lstsq(Mat, eps)[0]
            x = x / (x[0] + x[2])
            if not np.isnan(x).any() and not np.isinf(x).any():
                # self.x_ca_0_alg = np.zeros(4)
                x = np.concatenate((np.squeeze(x[3:6]), np.array(2 * np.arccos(np.sqrt(x[2])))))
                if not np.isnan(x).any() and not np.isinf(x).any():
                    self.x_ca_0_alg = x
                    self.wls_bool = True
        except np.linalg.LinAlgError:
            pass

        self.calculate_relative_position()

    def WLS_on_position(self):
        if not self.wls_bool:
            # self.calculate_relative_position_wls()
            self.wls_bool = False
            return
        z_mes = np.array(self.d[1:])
        z_mes = z_mes.reshape(-1, 1)
        p0 = self.x_ca_0_alg.copy()
        c0 = get_4d_rot_matrix(p0[-1])[:3, :3]
        e = 10000
        e_prev = 10000
        max_it = 100
        it = 0
        x = np.zeros((4, 1))
        lin_alg_error = False
        while e > 1e-8 and it < max_it:  # and e_prev >= e
            p0 = p0 + x.reshape(4)
            c0 = get_4d_rot_matrix(p0[-1])[:3, :3]
            it += 1
            p = np.empty((0, 3))
            # z = np.empty((0, 1))
            # H_theta = np.empty((0, 3))
            H_theta_a = np.empty((0, 1))
            H_p = np.empty((0, 3))
            H_r1 = np.empty((0, 4))
            H_r2 = np.empty((0, 4))
            z_est = np.empty((0, 1))
            W_inv = np.eye(len(self.eps) - 1)
            for i in range(1, len(self.eps)):
                c1_k = get_4d_rot_matrix(self.x_ha_odom[i, -1])[:3, :3].T
                p_k = c1_k @ (p0[:3] + c0 @ self.x_ca_odom[i, :3] - self.x_ha_odom[i, :3])
                p = np.append(p, p_k.reshape(1, *p_k.shape), axis=0)
                z_est_k = np.array([np.linalg.norm(p_k)])
                z_est = np.append(z_est, z_est_k.reshape(1, *z_est_k.shape), axis=0)
                # c0_p2_k = c0 @ self.x_ca_odom[i, :3]
                # c0_p2_k_x = np.array([[0, -c0_p2_k[2], c0_p2_k[1]],
                #                       [c0_p2_k[2], 0, -c0_p2_k[0]],
                #                       [-c0_p2_k[1], c0_p2_k[0], 0]])
                # H_theta_k = p_k.T @ c1_k @ c0_p2_k_x / self.d[i]
                # print(H_theta_k)
                # H_theta = np.append(H_theta, H_theta_k.reshape(1, *H_theta_k.shape), axis=0)

                # Alternative H_theta_k Is similar to above with dtheta_x and dtheta_y = 0.
                A = np.array([[-np.sin(self.x_ca_0[-1]) - np.cos(self.x_ca_0[-1]), 0, 0],
                              [0, np.cos(self.x_ca_0[-1]) - np.sin(self.x_ca_0[-1]), 0],
                              [0, 0, 0]])
                H_theta_a_k = - np.array([(p0[:3].T @ A @ self.x_ca_odom[i, :3] -
                                           self.x_ha_odom[i, :3].T @ A @ self.x_ca_odom[i, :3]) / \
                                          self.d[i]])
                H_theta_a = np.append(H_theta_a, H_theta_a_k.reshape(1, *H_theta_a_k.shape), axis=0)
                # print(H_theta_a_k)

                H_p0_k = p_k.T @ c1_k / self.d[i]
                H_p = np.append(H_p, H_p0_k.reshape(1, *H_p0_k.shape), axis=0)
                H_r1_k = np.concatenate((-p_k.T @ c1_k / self.d[i], np.zeros(1)))
                H_r1 = np.append(H_r1, H_r1_k.reshape(1, *H_r1_k.shape), axis=0)
                H_r2_k = np.concatenate((p_k.T @ c1_k @ c0 / self.d[i], np.zeros(1)))
                H_r2 = np.append(H_r2, H_r2_k.reshape(1, *H_r2_k.shape), axis=0)

                W_k = H_r1_k @ self.P_ha[i] @ H_r1_k.T + \
                      H_r2_k @ self.P_ca[i] @ H_r2_k.T + self.R
                W_inv[i - 1, i - 1] = 1. / W_k[0]
            H = np.concatenate((H_p, H_theta_a), axis=1)
            A = H.T @ W_inv @ H
            # print(z_mes - z_est)
            B = H.T @ W_inv @ (z_mes - z_est).reshape(-1, 1)
            # print(A.shape, B.shape)
            try:
                x = np.linalg.lstsq(A.astype(np.float64), B.astype(np.float64))[0]
            except np.linalg.LinAlgError:
                lin_alg_error = True
                if self.debug_bool:
                    print("singular matrix")
                break
            # print(x)

            # print(p_n)
            # p_n = p0
            e_prev = e
            e = np.linalg.norm(x)

            # print("error: ", e)
        if self.debug_bool:
            if e > e_prev:
                print("error increased")
            print("iterations: ", it)
        # print("p0: ", p0)
        if not lin_alg_error and it < max_it:
            self.x_ca_0 = p0.copy()
        self.calculate_relative_position_wls()

    def calculate_relative_position(self):
        c0 = get_4d_rot_matrix(self.x_ca_0_alg[-1])[:3, :3]
        c1 = get_4d_rot_matrix(self.x_ha_odom[-1, -1])[:3, :3]
        p = c1.T @ (self.x_ca_0_alg[:3] + c0 @ self.x_ca_odom[-1, :3] - self.x_ha_odom[-1, :3])
        h = limit_angle(self.x_ca_0_alg[-1] + self.x_ca_odom[-1, -1] - self.x_ha_odom[-1, -1])
        self.x_ca_r_alg = np.append(p, h)

    def calculate_relative_position_wls(self):
        c0 = get_4d_rot_matrix(self.x_ca_0[-1])[:3, :3]
        c1 = get_4d_rot_matrix(self.x_ha_odom[-1, -1])[:3, :3]
        p = c1.T @ (self.x_ca_0[:3] + c0 @ self.x_ca_odom[-1, :3] - self.x_ha_odom[-1, :3])
        h = limit_angle(self.x_ca_0[-1] + self.x_ca_odom[-1, -1] - self.x_ha_odom[-1, -1])
        self.x_ca_r = np.append(p, h)



class Algebraic4DoF_Logger:
    def __init__(self, alg_solver: AlgebraicMethod4DoF, host: NewRobot, connect: NewRobot):
        self.alg_solver = alg_solver

        self.x_ca_r_alg = np.zeros((1, 4))
        self.x_ca_r_alg_error = []
        self.x_ca_r_alg_heading_error = []

        self.x_ca_r_WLS = np.zeros((1, 4))
        self.x_ca_r_WLS_error = []
        self.x_ca_r_WLS_heading_error = []

        self.calculation_time_alg = []
        self.calculation_time_wls = []

        self.host_drone = host
        self.connect_drone = connect

    def log(self, i, calculation_time_alg=0, calculation_time_wls=0):
        self.calculation_time_alg.append(calculation_time_alg)
        self.calculation_time_wls.append(calculation_time_wls)

        ha_p_real = self.host_drone.x_real[i]
        ca_p_real = self.connect_drone.x_real[i]
        ha_h_real = self.host_drone.h_real[i]
        ca_h_real = self.connect_drone.h_real[i]

        relative_transformation = (ca_p_real - ha_p_real)
        spherical_relative_transformation = cartesianToSpherical(relative_transformation)
        spherical_relative_transformation[1] = limit_angle(spherical_relative_transformation[1] - ha_h_real)
        relative_transformation_r = sphericalToCartesian(spherical_relative_transformation)

        error_alg_est = np.linalg.norm(relative_transformation_r - self.alg_solver.x_ca_r_alg[:3])
        error_wls_est = np.linalg.norm(relative_transformation_r - self.alg_solver.x_ca_r[:3])

        self.x_ca_r_alg_error.append(error_alg_est)
        self.x_ca_r_WLS_error.append(error_wls_est)

        relative_heading = limit_angle(ca_h_real - ha_h_real)
        self.x_ca_r_WLS_heading_error.append(np.abs(limit_angle(self.alg_solver.x_ca_r[-1] - relative_heading)))
        self.x_ca_r_alg_heading_error.append(np.abs(limit_angle(self.alg_solver.x_ca_r_alg[-1] - relative_heading)))

    def plot_self(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(2, 1)
        ax[0].plot(self.x_ca_r_alg_error, label="alg")
        ax[0].plot(self.x_ca_r_WLS_error, label="WLS")
        ax[0].legend()
        ax[0].grid(True)
        ax[1].plot(self.x_ca_r_alg_heading_error, label="alg")
        ax[1].plot(self.x_ca_r_WLS_heading_error, label="WLS")
        ax[1].legend()
        ax[1].grid(True)

if __name__ == "__main__":
    run = "4dof"
    if run == "4dof":
        sigma_uwb = 1e-23
        dt = 0.2
        dv_ha = np.array([1, 0., 0., -0.2])
        dv_ca = np.array([1, 0., 1, 0.0])
        x_ha = np.zeros(4)
        x_ha_0 = x_ha.copy()
        x_ha_slam_error = np.zeros((1, 4))
        x_ca = np.array([2, 0, 1, np.pi / 2])
        x_ca_0 = x_ca.copy()
        x_ca_slam_error = np.zeros((1, 4))

        d0 = np.linalg.norm(x_ca[:3] - x_ha[:3]) + np.random.randn(1)[0] * sigma_uwb
        AM = AlgebraicMethod4DoF(d0, sigma_uwb=sigma_uwb)

        q = np.eye(4) * (1e-2) ** 2
        q[3, 3] = q[3, 3] * 0.01
        # q=np.zeros((4,4))

        for i in range(100):
            print("Iteration: ", i, " ---------------------")
            dv_ha = np.random.randn(4)
            dv_ca = np.random.randn(4)
            dx_ha = (dv_ha * dt)
            x_ha = x_ha + get_4d_rot_matrix(x_ha[-1]) @ dx_ha
            dx_ca = (dv_ca * dt)
            x_ca = x_ca + get_4d_rot_matrix(x_ca[-1]) @ dx_ca
            d = np.linalg.norm(x_ha[:3] - x_ca[:3]) + np.random.randn(1)[0] * sigma_uwb

            dx_ha = dx_ha + np.squeeze(np.random.randn(1, 4) @ np.sqrt(q) * dt)
            dx_ca = dx_ca + np.squeeze(np.random.randn(1, 4) @ np.sqrt(q) * dt)

            AM.get_update(d, dx_ha, dx_ca, q_ha=q * dt ** 2, q_ca=q * dt ** 2)
            AM.WLS_on_position()
            try:
                print(AM.P_ca[1], AM.P_ca[2])
            except IndexError:
                print(AM.P_ca)
            AM.trim_to_latest_measurements()
            p = get_4d_rot_matrix(x_ha[-1])[:3, :3] @ (x_ca[:3] - x_ha[:3])
            h = limit_angle(x_ca[-1] - x_ha[-1])
            # print(x_ca_slam_error[-1])
            # print(x_ha_slam_error[-1])
            # AM.find_relative_pose()

            # p_est, h_est = AM.WLS_on_position()
            print("real p: ", p, " estimated P: ", AM.x_ca_r[:3])
            print("real h: ", h, " estimated h: ", AM.x_ca_r[-1])
            print("Error position: ", np.linalg.norm(AM.x_ca_r[:3] - p))
            print("Error heading: ", np.abs(AM.x_ca_r[-1] - h))
            print(AM.x_ca_0_alg)
            print(AM.x_ca_0)
    if run == "6dof":
        pass
