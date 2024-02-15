#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:27:37 2023

@author: yuri
"""
from typing import List

import numpy as np
from Simulation.RobotClass import NewRobot


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