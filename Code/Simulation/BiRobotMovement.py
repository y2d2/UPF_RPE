#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:27:37 2023

@author: yuri
"""
from typing import List

import numpy as np
from Code.Simulation.RobotClass import NewRobot
from Code.UtilityCode.utility_fuctions import limit_angle


def drone_flight(start_pose, start_velocity=np.zeros(4), sigma_dv=0.01, sigma_dw=0.01, max_range=None,
                 origin_bool=False, simulation_time_step=0.1, slowrate_v=0.05, slowrate_w=0.001, max_v=1, max_w=0.05):

    drone = NewRobot(start_pose[:3], start_pose[3], simulation_time_step=simulation_time_step)
    drone.set_start_speed(start_velocity[:3], start_velocity[3])
    drone.set_uncertainties(sigma_dv=sigma_dv, sigma_dw=sigma_dw)
    if max_range is not None: drone.set_max_range(max_range, origin_bool=origin_bool)
    drone.set_random_movement_variables(slowrate_v=slowrate_v, slowrate_w=slowrate_w, max_v=max_v, max_w=max_w)
    return drone


def run_multi_drone_simulation(time_steps, list_of_drones: List[NewRobot], move_function, kwargs={}):
    for i in range(time_steps):
        kwargs["i"] = i
        move_function(list_of_drones, **kwargs)


def run_simulation(time_steps, host_agent: NewRobot, connected_agent: NewRobot, move_function, kwargs={}):
    print(kwargs)
    for i in range(time_steps):
        kwargs["i"] = i
        move_function(host_agent, connected_agent, **kwargs)

def fix_host_fix_connected(host: NewRobot, connected: NewRobot, **kwargs):
    connected.move(w=0, v=np.array([0, 0, 0]))
    host.move(w=0, v=np.array([0, 0, 0]))

def fix_connected_2D_host(host: NewRobot, connected: NewRobot, **kwargs):
    control2d = kwargs.get("control2d")
    control2d.set_control()
    connected.move(w=0, v=np.array([0, 0, 0]))

def both_3D_control(host: NewRobot, connected: NewRobot, **kwargs):
    control_host = kwargs.get("control_host")
    control_host.set_control()
    control_connected = kwargs.get("control_connected")
    control_connected.set_control()
    # connected.move(w=0, v=np.array([0, 0, 0]))
    # host.move(w=0, v=np.array([0, 0, 0]))



def fix_host_random_movement_connected(host: NewRobot, connected: NewRobot, **kwargs):
    connected.move_randomly()
    host.move(w=0, v=np.array([0, 0, 0]))


def fix_host_jumping_y_connected(host: NewRobot, connected: NewRobot, **kwargs):
    host.move(w=0, v=np.array([0, 0, 0]))
    i = kwargs.get("i")
    connected.move(w=0, v=(-1) ** i * np.array([0, 2, 0]))


def moving_gps_tracked_host_fix_connected(host: NewRobot, connected: NewRobot, **kwargs):
    connected.move(w=0, v=np.array([0, 0, 0]))
    host.move_randomly()


def moving_sinusoidal_gps_tracked_host_random_connected(host: NewRobot, connected: NewRobot, **kwargs):
    i = kwargs.get("i")
    speed = kwargs.get("speed")
    w = kwargs.get("w")
    w_heading = kwargs.get("w_Heading")

    connected.move_randomly()
    z_speed = speed * np.sin(w * i)
    host.move(v=np.array([speed, 0, z_speed]), w=w_heading)


def random_movements_host_random_movements_connected(host: NewRobot, connected: NewRobot, **kwargs):
    host.move_randomly()
    connected.move_randomly()


def random_moving_drones(list_of_drones: List[NewRobot], **kwargs):
    for drone in list_of_drones:
        drone.move_randomly()


# 2 agent scenarios

class Control1D():
    def __init__(self, agent: NewRobot,
                 max_v=0.5, max_dot_v = 0.2, max_w=0.5, max_dot_w = 0.1, p_angle=1., p_pos = 1.,
                 frequency = 10):
        self.frequency = frequency
        self.max_v = max_v
        self.max_dot_v = max_dot_v
        self.max_delta_v = max_dot_v / frequency
        self.max_w = max_w
        self.max_dot_w = max_dot_w
        self.max_delta_w = max_dot_w/frequency

        self.center = np.array([0,0,0])
        self.radius = 2

        self.target = np.array([0,0,0])

        self.p_angle = p_angle
        self.p_pos = p_pos

        self.w = 0
        self.v = 0

        self.agent = agent

    def set_boundries(self, center=np.array([0,0,0]), radius = 2):
        self.center = center
        self.radius = radius

    def set_random_target(self):
        r = np.random.uniform(0, self.radius)
        theta = np.random.uniform(0, 2*np.pi)
        self.target = self.center + np.array([r*np.cos(theta), r*np.sin(theta), 0])


    def set_control(self):
        x = self.agent.x_real[-1]
        theta =  self.agent.h_real[-1]
        v_real = self.agent.v_slam_real[-1]
        v_norm = np.linalg.norm(v_real)
        if v_norm == 0:
            v_ax = np.array([1,0,0])
        else:
            v_ax = v_real/v_norm
        w_real = self.agent.w_slam_real[-1]

        dx = self.target - x

        if np.linalg.norm(dx) < 0.1:
            self.set_random_target()
            dx = self.target - x

        angle = limit_angle(np.arctan2(dx[1], dx[0]) - theta)

        #Define turn angle
        w_tar = self.p_angle*angle
        dot_w_tar = w_tar - w_real
        if np.abs(dot_w_tar) > self.max_dot_w:
            w_tar = w_real +  self.max_dot_w * np.sign(dot_w_tar)
        if np.abs(w_tar) > self.max_w:
            w_tar = self.max_w * np.sign(w_tar)

        # Slow down to halt to turn
        if np.abs(angle) > 0.1:
            if np.linalg.norm(v_real) < self.max_dot_v:
                v_tar = 0
            else:
                v_tar = v_norm - self.max_dot_v
        # Move towards target.
        else:
            v_tar = self.p_pos * np.linalg.norm(dx)
            dot_v_tar = v_tar - v_norm
            if np.abs(dot_v_tar) > self.max_dot_v:
                v_tar  = v_norm + self.max_dot_v * np.sign(dot_v_tar)
            if np.abs(v_tar) > self.max_v:
                v_tar = self.max_v * np.sign(v_tar)

        v = v_ax * v_tar
        self.agent.move(w_tar, v)



class Control3D(Control1D):
    def __init__(self, agent: NewRobot,
                 max_v=0.5, max_dot_v=0.2, max_w=0.5, max_dot_w = 0.1, p_angle=1., p_pos = 1.,
             frequency = 10):
        super().__init__(agent, max_v, max_dot_v, max_w, max_dot_w , p_angle, p_pos, frequency)
        self.angle_target=0
        self.height = 2.
        self.target_time_max = 10
        self.target_time = 0
        self.current_target_time = 0

    def set_random_target(self):
        self.current_target_time = 0
        r = np.random.uniform(0, self.radius)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        h = np.random.uniform(0, self.height)
        self.target = self.center + np.array([r * np.cos(phi)* np.cos(theta), r *  np.cos(phi) * np.sin(theta), r *  np.sin(phi)])
        self.angle_target = np.random.uniform(0, 2*np.pi)
        self.target_time = np.random.uniform(self.target_time_max/2, self.target_time_max)

    def set_control(self):
        # x = self.agent.x_real[-1]
        # theta =  self.agent.h_real[-1]
        # v_real = self.agent.v_slam_real[-1]
        # v_norm = np.linalg.norm(v_real)
        #
        # w_real = self.agent.w_slam_real[-1]
        #
        #
        #
        # dx = self.target - x
        #
        # if np.linalg.norm(dx) < 0.1:
        #     self.set_random_target()
        #     dx = self.target - x
        #
        # v_ax = dx / np.linalg.norm(dx)
        # angle = limit_angle(self.angle_target - theta)
        #
        #
        #
        # #Define turn angle
        # w_tar = self.p_angle*angle
        # dot_w_tar = w_tar - w_real
        # if np.abs(dot_w_tar) > self.max_dot_w:
        #     w_tar = w_real +  self.max_dot_w * np.sign(dot_w_tar)
        # if np.abs(w_tar) > self.max_w:
        #     w_tar = self.max_w * np.sign(w_tar)
        #
        #
        #
        # # Slow down to halt to turn
        # if np.abs(angle) > 0.1:
        #     v_tar = 0
        # else:
        #     v_tar = self.max_v
        #    # Move towards target.
        # v_tar = np.linalg.norm(dx)
        # v_dot_tar = self.p_pos * (v_tar - v_norm)
        #
        # if np.abs(v_dot_tar) > self.max_dot_v:
        #     v_dot_tar  =  self.max_dot_v * np.sign(v_dot_tar)
        #
        # v = v_real + v_ax * v_dot_tar
        # if np.linalg.norm(v) > self.max_v:
        #     v = self.max_v * v / np.linalg.norm(v)
        # print(self.target, dx, w_tar, v_tar, v)
        #
        # self.agent.move(w_tar, v)

        # Extract current and target positions and yaw
        if self.current_target_time >= self.target_time:
            self.set_random_target()
        self.current_target_time += 1/self.frequency

        current_position = self.agent.x_real[-1]
        current_yaw = self.agent.h_real[-1]

        current_velocity = self.agent.v_slam_real[-1]
        current_rotational_speed = self.agent.w_slam_real[-1]


        dx = self.target - current_position

        # if np.linalg.norm(dx) < 0.1:
        #     self.set_random_target()

        target_position = self.target
        target_yaw = self.angle_target



        # Compute position error
        position_error = target_position - current_position
        rot_matrix = np.array([[np.cos(current_yaw), np.sin(current_yaw), 0],
                               [-np.sin(current_yaw), np.cos(current_yaw), 0],
                               [0, 0, 1]])
        position_error1 = rot_matrix @ position_error
        position_error2 = self.p_pos * position_error1

        v_dot_vec = position_error2 - current_velocity
        v_dot_tar = np.linalg.norm(v_dot_vec)
        if np.abs(v_dot_tar) > self.max_delta_v/self.frequency:
            v_dot_tar = self.max_delta_v/self.frequency * np.sign(v_dot_tar)

        try:
            v = current_velocity + v_dot_vec * v_dot_tar/np.linalg.norm(v_dot_vec)
        except RuntimeWarning:
            v = current_velocity + v_dot_vec * v_dot_tar


        # Compute yaw error (normalize to [-pi, pi])
        yaw_error = target_yaw - current_yaw
        yaw_error_1 =(yaw_error - np.pi )% (2 * np.pi) - np.pi
        w_tar = self.p_angle * yaw_error_1
        dot_w = w_tar - current_rotational_speed
        if np.abs(dot_w) > self.max_delta_w:
            w_tar = current_rotational_speed +  self.max_delta_w * np.sign(dot_w)



        # Compute velocity commands
        velocity_command = np.clip(v, -self.max_v, self.max_v)
        yaw_command = np.clip(w_tar, -self.max_w, self.max_w)
        # print(self.target, dx, velocity_command, v)
        # print(yaw_command, yaw_error_1, yaw_error, dot_w)

        self.agent.move(yaw_command, velocity_command)