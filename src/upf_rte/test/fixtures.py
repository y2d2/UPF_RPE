#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fixtures and utilities for UPF_RPE unit tests.

This module centralizes common test data, simulation setups, and helper functions
to avoid code duplication across test modules.
"""

import numpy as np

__all__ = ['set_deterministic_seed', 'create_test_drone_poses', 'generate_simple_trajectory']


def set_deterministic_seed(seed=42):
    """Set deterministic random seed for reproducible tests."""
    np.random.seed(seed)
    return seed


def create_test_drone_poses(n_poses=10, max_range=5.0):
    """Create test drone poses in SE(2) with random positions and headings."""
    set_deterministic_seed()
    x = np.random.uniform(-max_range, max_range, n_poses)
    y = np.random.uniform(-max_range, max_range, n_poses)
    z = np.zeros(n_poses)  # 2D poses for simplicity
    heading = np.random.uniform(0, 2 * np.pi, n_poses)
    
    poses = np.column_stack([x, y, z, heading])
    return poses


def generate_simple_trajectory(dt=0.1, duration=10.0, linear_vel=1.0, angular_vel=0.1):
    """Generate a simple trajectory with constant velocities."""
    set_deterministic_seed()
    
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    # Simple circular trajectory
    x = linear_vel * np.cos(angular_vel * t)
    y = linear_vel * np.sin(angular_vel * t)
    z = np.zeros(n_steps)
    heading = angular_vel * t
    
    positions = np.column_stack([x, y, z])
    headings = heading
    
    return positions, headings
