import numpy as np
import quaternion as quaternion

def get_roll_pitch_yaw_from_SO3(R):
    q = quaternion.from_rotation_matrix(R)
    roll, pitch, yaw = quaternion.as_euler_angles(q)
    return np.array([roll, pitch, yaw])

def get_w_from_SO3(R):
    w = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(R))
    return w

# def get_S3_rotation_matrix(w):
#     theta = np.linalg.norm(w)
#     if theta == 0:
#         return np.eye(3)
#     k = w/np.linalg.norm(w)
#     K = so3_hat(k)
#     R = (np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K))
#     return R

def get_SO3_rotation_matrix(w, dt=None):
    if dt is not None:
        w = w*dt
    theta = np.linalg.norm(w)
    if theta == 0:
        return np.eye(3)
    k = w/np.linalg.norm(w)
    K = so3_hat(k)
    R = (np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K))
    return R

def get_q_from_SO3(R):
    return quaternion.as_float_array(quaternion.from_rotation_matrix(R))

def so3_hat(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]])


def get_eps(fm, wm):
    eps = np.zeros((5,5))
    eps [:3, :3] = so3_hat(wm)
    eps[:3, 3] = fm
    return eps
