import numpy as np
import quaternion as quaternion


def get_SO3_rotation_matrix(w, dt):
    theta = np.linalg.norm(w)*dt
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
