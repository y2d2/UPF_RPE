#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:45:10 2023

@author: yuri
"""
from deprecated import deprecated
import h5py
import pickle as pkl
import numpy as np

from utility_fuctions import get_rot_matrix, get_4d_rot_matrix, sphericalToCartesian, cartesianToSpherical, limit_angle
import Experiments.Transformation_Matrix_Fucntions as TMF

def load_trajectory_from_pickle(pickle_file="./trajectory.pkl", sigma_dw=None, sigma_dv=None):
    with open(pickle_file, "rb") as f:
        trajectory_dict = pkl.load(f)
    return load_trajectory_from_dict_old(trajectory_dict, sigma_dw, sigma_dv)

def load_trajectory_from_dict(trajectory_dict):
    robot = NewRobot(trajectory_dict["x_start"], trajectory_dict["h_start"],
                     simulation_time_step= trajectory_dict["simulation_time_step"])
    robot.set_random_movement_variables(trajectory_dict["slowrate_v"], trajectory_dict["slowrate_w"],
                                        trajectory_dict["max_v"], trajectory_dict["max_w"])

    robot.x_real = trajectory_dict["x_real"]
    robot.h_real = trajectory_dict["h_real"]
    robot.dx_slam_real = trajectory_dict["dx_slam_real"]
    robot.dh_slam_real = trajectory_dict["dh_slam_real"]
    return robot

def load_trajectory_from_dict_old(trajectory_dict, sigma_dw=None, sigma_dv=None):
    dx_slam_real = trajectory_dict["dx_slam_real"]
    dh_slam_real = trajectory_dict["dh_slam_real"]

    robot = NewRobot(trajectory_dict["x_start"], trajectory_dict["h_start"],
                     simulation_time_step= trajectory_dict["simulation_time_step"])
    robot.set_random_movement_variables(trajectory_dict["slowrate_v"], trajectory_dict["slowrate_w"],
                                        trajectory_dict["max_v"], trajectory_dict["max_w"])

    # If the real trajcetory is already provide no need to re
    if "x_real" in trajectory_dict:
        robot.x_real = trajectory_dict["x_real"]
        robot.h_real = trajectory_dict["h_real"]
        robot.dx_slam_real = trajectory_dict["dx_slam_real"]
        robot.dh_slam_real = trajectory_dict["dh_slam_real"]

        # effectively regenerating the uncertainty
        if sigma_dv is not None and sigma_dw is not None:
            robot.calculate_d_slam(sigma_dv, sigma_dw)
            robot.calculate_d_slam_drift()
        # dx_slam =
        # for i in range(dx_slam_real.shape[0] - 1):
        #     robot.update_run_time_variables(dx_slam_real[i + 1], dh_slam_real[i + 1])
        # return robot
    else:
        if sigma_dv is None:
            sigma_dv = 0
        if sigma_dw is None:
            sigma_dw = 0
        robot.set_uncertainties(sigma_dv, sigma_dw)
        for i in range(dx_slam_real.shape[0] - 1):
            robot.update_run_time_variables(dx_slam_real[i + 1], dh_slam_real[i + 1])
    return robot


class NewRobot:
    # TODO: adapt x (x,y,z) and h (h) into single x (x,y,z,h) variable
    def __init__(self, x_start: np.ndarray = np.zeros(3), h_start: float = 0, simulation_time_step=0.01):
        # simulation parameters:
        # self.run_time_step = run_time_step
        self.simulation_time_step = simulation_time_step
        # self.steps = int(run_time_step / simulation_time_step)

        # Position simulation variables:
        self.x_start = x_start
        self.x_real = x_start.reshape((1, 3))
        self.x_real_sim = x_start.reshape((1, 3))
        self.x_real_odom = np.zeros((1, 3))
        self.dx_slam_real = np.zeros((1, 3))
        self.dx_slam = np.zeros((1, 3))
        self.x_slam = x_start.reshape((1, 3))
        self.x_error = np.zeros(1)
        self.v_slam_real = np.zeros((1, 3))

        # Heading simulation variables:
        self.h_start = h_start
        self.h_real = np.array([h_start])
        self.h_real_sim = np.array([h_start])
        self.h_real_odom = np.zeros(1)
        self.dh_slam_real = np.zeros(1)
        self.dh_slam = np.zeros(1)
        self.h_slam = np.array([h_start])
        self.h_error = np.zeros(1)
        self.w_slam_real = np.zeros(1)

        # speed variables:
        self.v_slam = np.zeros(3)
        self.w_slam = np.zeros(1)

        # Uncertainties variables:
        self.sigma_dv = 0
        self.sigma_dw = 0
        self.sigma_dx = 0
        self.sigma_dh = 0
        self.q = np.zeros((4, 4))

        # Wrost case drift variables:
        self.bool_worst_case = False
        self.speed_drift_modifier = 1
        self.heading_drift_modifier = 1
        self.x_wc_drift = x_start.reshape((1, 3))
        self.h_wc_drift = np.array([h_start])

        # random movement variables:
        self.target_x =  np.zeros(3)
        self.target_h = 0.
        self.slowrate_v = 0.
        self.slowrate_w = 0.
        self.max_v = 0.
        self.max_w = 0.

        # max range variables:
        self.max_range = 0
        self.max_range_bool = False
        self.origin_bool = False

        # Plotting variables:
        self.color = "k"
        self.mark = ""
        self.linestyle = "-"

        # integration varibale:
        self.dx_int = np.zeros(4)
        self.q_int = np.zeros((4, 4))

    #---------------------------
    # Setup functions:
    #---------------------------
    def set_uncertainties(self, sigma_dv, sigma_dw):
        self.sigma_dv = sigma_dv
        self.sigma_dw = sigma_dw
        self.sigma_dx = sigma_dv * self.simulation_time_step
        self.sigma_dh = sigma_dw * self.simulation_time_step
        self.q = np.diag([self.sigma_dx ** 2, self.sigma_dx ** 2, self.sigma_dx ** 2, self.sigma_dh ** 2])

    def set_random_movement_variables(self, slowrate_v, slowrate_w, max_v, max_w):
        self.slowrate_v = slowrate_v
        self.slowrate_w = slowrate_w
        self.max_v = max_v
        self.max_w = max_w
        self.set_new_random_target()

    def set_start_speed(self, v, w):
        self.v_slam = v
        self.w_slam = w

    def set_start_position(self, x, h):
        self.x_start = x
        self.h_start = h
        self.x_real = x.reshape((1, 3))
        self.h_real = np.array([h])

    def set_max_range(self, max_range, origin_bool=False):
        self.max_range = max_range
        self.max_range_bool = True
        self.origin_bool = origin_bool

    def set_worst_drift_case(self, bool_worst_case, speed_drift_modifier = 1, heading_drift_modifier = 1):
        self.bool_worst_case= bool_worst_case
        self.speed_drift_modifier = speed_drift_modifier
        self.heading_drift_modifier = heading_drift_modifier

    # ---------------------------
    # simulation functions:
    # ---------------------------

    def move(self, w=0, v=np.zeros(3)):
        dx = np.zeros(3)
        dh = 0
        dx = dx + get_rot_matrix(dh) @ v * self.simulation_time_step
        dh += w * self.simulation_time_step
        self.update_run_time_variables(dx, dh)

    def set_random_target_boundaries(self,origin = np.zeros(3), ranges=np.zeros(3) ):
        self.target_origin = origin
        self.target_ranges = ranges

    def set_new_random_target(self):
        self.target_x = np.random.uniform(self.target_origin - self.target_ranges, self.target_origin + self.target_ranges)
        self.target_h = np.random.randn(1)[0] * 2 * np.pi

    def move_randomly2(self):
        if np.linalg.norm(self.x_real - self.target_x) < 0.1:
            if not (self.h_real[-1] - self.target_h < 0.1):
                dw = (10 *  limit_angle(self.target_h - self.h_real[-1])) - self.w_slam_real[-1]
                if np.abs(dw) > self.slowrate_w:
                    w = self.w_slam_real[-1] + np.sign(dw) * self.slowrate_w
                else:
                    w = self.w_slam_real[-1] + dw
                if np.abs(w) > self.max_w:
                    w = np.sign(w)* self.max_w
                dh = w * self.simulation_time_step
                dx = np.zeros(3)
            else:
                self.set_new_random_target()
        else:
            #TODO: find bearings for moving in the right direction
            local_target_h = 0
            d = np.linalg.norm(self.target_x - self.x_real[-1])
            x_unit = (self.target_x - self.x_real[-1])/d
            x_unit = get_rot_matrix(self.h_real[-1]) @ x_unit
            desired_v = 10 * d
            dv = desired_v - np.linalg.norm(self.v_slam_real[-1])
            if np.abs(dv) > self.slowrate_v:
                v = self.v_slam_real[-1] + np.sign(dv) * self.slowrate_v
            else:
                v = self.v_slam_real[-1] + dv
            if np.linalg.norm(v) > self.max_v:
                v = v * self.max_v / np.linalg.norm(v)
            dx = v * self.simulation_time_step * x_unit
            dh = 0
        self.update_run_time_variables(dx, dh)

    def move_randomly(self):
        # TODO: Detailed trajectory
        self.calculate_speed()
        dx = self.v_slam * self.simulation_time_step
        dh = self.w_slam * self.simulation_time_step
        dx = self.check_max_range(dx)

        self.update_run_time_variables(dx, dh)

    def calculate_speed(self):
        v = self.v_slam + np.random.randn(3) * self.slowrate_v*self.simulation_time_step
        w = self.w_slam + np.random.randn(1)[0] * self.slowrate_w*self.simulation_time_step
        if np.linalg.norm(v) > self.max_v:
            v = v * self.max_v / np.linalg.norm(v)
        if np.abs(w) > self.max_w:
            w = np.sign(w) * self.max_w

        self.v_slam = v
        self.w_slam = w

    def check_max_range(self, dx):
        if self.max_range_bool:
            x_odom = self.x_real_odom[-1] + get_rot_matrix(self.h_real_odom[-1]) @ dx
            if self.origin_bool:
                x_odom = self.x_start + get_rot_matrix(self.h_start) @ x_odom
            s_odom = cartesianToSpherical(x_odom)
            if s_odom[0] > self.max_range:
                s_odom[0] = self.max_range
                x_odom = sphericalToCartesian(s_odom)
                self.v_slam = -1. * self.v_slam
                if self.origin_bool:
                    x_odom = get_rot_matrix(-self.h_start) @ (x_odom - self.x_start)
                dx = get_rot_matrix(-self.h_real_odom[-1]) @ (x_odom - self.x_real_odom[-1])
        return dx

    def update_run_time_variables(self, dx, dh):
        dh = np.array([dh])
        # dx = self.check_max_range(dx)
        x_real = self.x_real_odom[-1] + get_rot_matrix(self.h_real_odom[-1]) @ dx

        self.x_real_odom = np.concatenate((self.x_real_odom, x_real.reshape((1, 3))))

        h_real = self.h_real_odom[-1] + dh
        self.h_real_odom = np.concatenate((self.h_real_odom, h_real.reshape(1)))

        self.dx_slam_real = np.concatenate((self.dx_slam_real, dx.reshape((1, 3))))
        self.dh_slam_real = np.concatenate((self.dh_slam_real, dh.reshape(1)))

        self.dx_slam = np.concatenate((self.dx_slam, (dx + np.random.randn(3) * self.sigma_dx).reshape((1, 3))))
        self.dh_slam = np.concatenate((self.dh_slam, (dh + np.random.randn(1) * self.sigma_dh).reshape(1)))

        x_slam = self.x_slam[-1] + get_rot_matrix(self.h_slam[-1]) @ self.dx_slam[-1]
        self.x_slam = np.concatenate((self.x_slam, x_slam.reshape((1, 3))))
        h_slam = self.h_slam[-1] + self.dh_slam[-1]
        self.h_slam = np.concatenate((self.h_slam, h_slam.reshape(1)))

        if self.bool_worst_case:
            x_wc = self.x_wc_drift[-1] + get_rot_matrix(self.h_wc_drift[-1]) @ (dx + self.speed_drift_modifier * np.ones(3) * self.sigma_dx)
            self.x_wc_drift = np.concatenate((self.x_wc_drift, x_wc.reshape((1, 3))))
            h_wc = self.h_wc_drift[-1] + (dh + self.heading_drift_modifier * self.sigma_dh)
            self.h_wc_drift = np.concatenate((self.h_wc_drift, h_wc.reshape(1)))

        self.v_slam_real = np.concatenate((self.v_slam_real, self.v_slam.reshape((1, 3))))
        self.w_slam_real = np.concatenate((self.w_slam_real, self.w_slam.reshape(1)))

        x_real = self.x_start + get_rot_matrix(self.h_start) @ self.x_real_odom[-1]
        self.x_real = np.concatenate([self.x_real, x_real.reshape((1, 3))])
        h_real = self.h_start + self.h_real_odom[-1]
        self.h_real = np.concatenate([self.h_real, np.array([h_real])])

        x_error = np.linalg.norm(self.x_real[-1] - self.x_slam[-1])
        h_error = np.abs(self.h_real[-1] - self.h_slam[-1])
        self.x_error = np.concatenate((self.x_error, x_error.reshape(1)))
        self.h_error = np.concatenate((self.h_error, h_error.reshape(1)))
        return None

    @deprecated("Seems like an older function. x_real is normally already calculated at update.")
    def get_real_position(self):
        for i, h_real_odom in enumerate(self.h_real_odom):
            self.x_real = np.concatenate([self.x_real, (
                    self.x_start + get_rot_matrix(self.h_start) @ self.x_real_odom[i]).reshape((1, 3))])
            self.h_real = np.concatenate([self.h_real, np.array([self.h_start + h_real_odom])])

    # -------------------
    # Plotting functions
    # -------------------
    def set_plotting_settings(self, color="k", mark="", linestyle="-"):
        self.color = color
        self.mark = mark
        self.linestyle = linestyle

    def plot_real_position(self, ax, annotation="", alpha=1, i=-1, history=None):

        if history is None or history > i:
            j=0
        else:
            j = i - history
            if j < 0:
                j = 0


        if annotation is None:
            ax.plot3D(self.x_real[j:i, 0], self.x_real[j:i, 1], self.x_real[j:i, 2],
                      color=self.color, marker=self.mark, linestyle=self.linestyle, alpha=alpha)
        else:
            ax.plot3D(self.x_real[j:i, 0], self.x_real[j:i, 1], self.x_real[j:i, 2],
                      color=self.color, marker=self.mark, linestyle=self.linestyle, alpha=alpha,
                      label=annotation + " " + "real position")
        ax.plot3D(self.x_real[j, 0], self.x_real[j, 1], self.x_real[j, 2],
                  color=self.color, marker="o", linestyle=self.linestyle, alpha=alpha)
        ax.plot3D(self.x_real[i, 0], self.x_real[i, 1], self.x_real[i, 2],
                  color=self.color, marker="x", linestyle=self.linestyle, alpha=alpha)

    def plot_slam_position(self, ax, annotation="", linestyle=":", alpha=1, i=-1):

        ax.plot3D(self.x_slam[:i, 0], self.x_slam[:i, 1], self.x_slam[:i, 2],
                  color=self.color, marker=self.mark, linestyle=linestyle, alpha=alpha,
                  label=annotation + " " + "slam position")
        ax.plot3D(self.x_slam[0, 0], self.x_slam[0, 1], self.x_slam[0, 2],
                  color=self.color, marker="o", linestyle=linestyle, alpha=alpha)
        ax.plot3D(self.x_slam[i, 0], self.x_slam[i, 1], self.x_slam[i, 2],
                  color=self.color, marker="x", linestyle=linestyle, alpha=alpha)

    def plot_trajectory(self, ax, color="k"):
        ax.plot3D(self.x_real[:, 0], self.x_real[:, 1], self.x_real[:, 2], color=color)



    # -------------------
    # Experiment functions
    # -------------------

    def from_experimental_data(self, T_vicon, DT_vio, Q, sample_frequency ):
        self.set_vicon_tracking(T_vicon)
        self.set_vio_slam(DT_vio, Q)
        self.simulation_time_step = 1/sample_frequency

    def set_vicon_tracking(self, Ts):
        self.T0 = Ts[0]
        self.x_start = TMF.get_translation(Ts[0])
        self.h_start = TMF.get_rotation_vector(Ts[0])[-1]
        x_real = np.empty((0, 3))
        h_real = []
        for T in Ts:
            x_real = np.vstack((x_real, TMF.get_translation(T)))
            h_real.append(TMF.get_rotation_vector(T)[-1])
        self.x_real = x_real
        self.h_real = h_real

    def set_vio_slam(self, DTs, Q):
        T_slam = self.T0
        x_slam = np.empty((0, 3))
        h_slam = []
        dx_slam = np.empty((0,3 ))
        dh_slam = []
        self.q = Q
        for DT in DTs:
            dx_slam = np.vstack((dx_slam,TMF.get_translation(DT)))
            dh_slam.append(TMF.get_rotation_vector(DT)[-1])
            T_slam = T_slam@DT
            x_slam = np.vstack((x_slam, TMF.get_translation(T_slam)))
            h_slam.append(TMF.get_rotation_vector(T_slam)[-1])
        self.dx_slam = dx_slam
        self.dh_slam = dh_slam
        self.x_slam = x_slam
        self.h_slam = h_slam
    # -------------------
    # Integration functions
    # -------------------
    def integrate_odometry(self, i):
        dslam = np.concatenate((self.dx_slam[i], np.array([self.dh_slam[i]])))
        h = get_4d_rot_matrix(self.dx_int[-1])
        self.dx_int = self.dx_int + h @ dslam
        self.q_int = self.q_int + h @ self.q @ h.T

    def reset_integration(self):
        dx = self.dx_int.copy()
        q = self.q_int.copy()
        self.dx_int = np.zeros(4)
        self.q_int = np.zeros((4, 4))
        return dx, q

    # -------------------
    # Uncertainty functions
    # -------------------
    def calculate_d_slam(self, sigma_v, sigma_w):
        self.set_uncertainties(sigma_v, sigma_w)
        self.dx_slam = self.dx_slam_real + np.random.randn(*self.dx_slam_real.shape)*sigma_v*self.simulation_time_step
        self.dh_slam = self.dh_slam_real + np.random.randn(*self.dh_slam_real.shape)*sigma_w*self.simulation_time_step
        return self.dx_slam, self.dh_slam

    def calculate_d_slam_drift(self):
        self.x_slam = self.x_real[0].reshape((1, 3))
        self.h_slam = self.h_real[0].reshape((1))
        for i, dx in enumerate(self.dx_slam):
            if i != 0:
                x_slam = self.x_slam[-1] + get_rot_matrix(self.h_slam[-1]) @ dx
                self.x_slam = np.concatenate([self.x_slam, x_slam.reshape((1, 3))])
                self.h_slam = np.concatenate([self.h_slam, np.array([self.h_slam[-1] + self.dh_slam[i]])])



    # -------------------
    # Saving functions
    # -------------------
    def get_trajectory_dict(self):
        traj_dict = {"x_start": self.x_start, "h_start": self.h_start,
                     "max_v": self.max_v, "max_w": self.max_w,
                     "slowrate_v": self.slowrate_v,"slowrate_w": self.slowrate_w,
                     "x_real": self.x_real, "h_real": self.h_real,
                     "dx_slam_real": self.dx_slam_real, "dh_slam_real": self.dh_slam_real,
                     "simulation_time_step": self.simulation_time_step}
        return traj_dict

    def save_trajectory_dict(self, file="./trajectory_dict.pkl"):
        traj_dict = self.get_trajectory_dict()
        with open(file, "wb") as f:
            pkl.dump(traj_dict, f)
        return None

if __name__ == "__main__":
    pass
