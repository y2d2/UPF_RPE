#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:28:22 2023

@author: yuri
"""
import numpy as np


def get_rot_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.],
                     [0., 0., 1.]], dtype=np.float64)


# def get_transform_matrix(angle):
#     return np.array([[np.cos(angle), -np.sin(angle), 0., 0.],
#                      [np.sin(angle), np.cos(angle), 0., 0.],
#                      [0., 0., 1., 0.],
#                      [0., 0., 0., 1.]], dtype=object)

def limit_angle(angle: float) -> float:
    angle = angle % (2 * np.pi)
    while angle <= -np.pi:
        angle = angle + 2 * np.pi
    while angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def cartesianToSpherical(x):
    if len(x) != 3:
        raise ValueError("The input vector has to be of length 3")
    R = np.linalg.norm(x[:3])
    theta = np.arctan2(x[1], x[0])
    phi = np.arctan2(x[2], np.linalg.norm(x[:2]))
    return np.array([R, theta, phi])


def cartesian_to_spherical_delta_x(x, dx):
    x_e = x + dx
    r_0 = np.linalg.norm(x)
    r_e = np.linalg.norm(x_e)
    theta_0 = limit_angle(np.arctan2(x[1], x[0]))
    theta_e = limit_angle(np.arctan2(x_e[1], x_e[0]))
    phi_e = np.arctan2(x_e[2], np.linalg.norm(x_e[:2]))
    d_theta = theta_e - theta_0
    if np.abs(d_theta) > np.pi / 2:
        d_theta = limit_angle(d_theta + np.pi)
        theta_e = theta_0 + d_theta
        phi_e = limit_angle(np.pi - phi_e)

    return np.array([r_e, theta_e, phi_e])


def sphericalToCartesian(s):
    """
    

    Parameters
    ----------
    s : TYPE
        Spherical vector np array with s = [ r, azimuth, altitude].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if len(s) != 3:
        raise ValueError("The input vector has to be of length 3")
    x = s[0] * np.cos(s[1]) * np.cos(s[2])
    y = s[0] * np.sin(s[1]) * np.cos(s[2])
    z = s[0] * np.sin(s[2])
    return np.array([x, y, z])


def get_4d_rot_matrix(h):
    rot = np.eye(4)
    rot[0, 0] = np.cos(h)
    rot[0, 1] = -np.sin(h)
    rot[1, 0] = np.sin(h)
    rot[1, 1] = np.cos(h)
    return rot

def get_transformation_matrix(x):
    trans = get_4d_rot_matrix(x[-1])
    trans[:3,-1] = x[:3]
    return trans



def transform_matrix(t):
    """
    t is the 4x1 vector representing the transformation
    x is the 4x1 vector representing the point to be transformed
    returns transformed vector y and the 4x4 transformation matrix needed for the covariance.
    """
    T = np.eye(5)
    T[:4, :4] = get_4d_rot_matrix(t[-1])
    T[:4, -1] = t[:4]
    # X = np.eye(5)
    # X[:4, :4] = get_4d_rot_matrix(x[-1])
    # X[:4, -1] = x[:4]
    # # x = np.concatenate([x, np.array([1])])
    # Y = T @ X
    # y = Y[:4,-1]
    # y[4] = limit_angle(y[4])
    return T

def inv_transformation_matrix(t):
    T = transform_matrix(t)
    R = np.transpose(T[:4,:4])
    t_conj = - R @ np.copy(t)

    #
    # t_conj = -np.copy(t)
    #
    #
    #
    #
    #
    # q_0 = np.cos(t[-1] / 2)
    # q_z = np.sin(t[-1] / 2)
    # t_conj[0] = -t[0] + 2*(q_z**2 *t[0] - q_0*q_z*t[1])
    # t_conj[1] = -t[1] + 2*(-q_0*q_z*t[0] + q_z**2*t[1])
    T_inv =transform_matrix(t_conj)
    I = T_inv @ T
    return T_inv

def get_states_of_transform(T):
    return T[:4,-1]

def get_covariance_of_transform(T):
    return T[:4,:4]
