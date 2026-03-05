import numpy as np
import quaternion as quaternion

def limit_angle_old(angle: float) -> float:
    angle = angle % (2 * np.pi)
    while angle <= -np.pi:
        angle = angle + 2 * np.pi
    while angle > np.pi:
        angle = angle - 2 * np.pi
    return angle

def limit_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))
#########################################################################
# SO(3) Functions
#########################################################################
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
    # Rodrigues' rotation formula: R = I + sin(theta)*K + (1 - cos(theta))*K^2
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

#########################################################################
# SE(3) Functions
#########################################################################
def SE3_inverse(T):
    R = T[:3, :3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, -1] = -R.T @ T[:3, -1]
    return T_inv

def SE3_get_translation(T):
    return T[:3, -1]

def SE3_get_rotation_matrix(T):
    return T[:3, :3]

def SE3_get_quaternion(T):
    return quaternion.as_float_array(quaternion.from_rotation_matrix(T[:3,:3]))

def SE3_get_rotation_vector(T):
    # try:
    # w = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(T[:3,:3]))

    q = quaternion.from_rotation_matrix(T[:3, :3])
    if isinstance(q, np.ndarray):
        q = q[0]  # Extract the single quaternion from the array

    w = quaternion.as_rotation_vector(q)

    w_amplitutede = np.linalg.norm(w)
    if w_amplitutede == 0:
        return np.zeros(3)
    w_unit = w/np.linalg.norm(w)
    w_amplitutede = limit_angle(w_amplitutede)
    w = w_unit*w_amplitutede
    return w

def SE3_from_rot_vec_and_trans(w, t):
    T = np.eye(4)
    T[:3,:3] = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(w))
    T[:3,-1] = t
    return T

#########################################################################
# SE_2(3) Functions
#########################################################################
# X is the SE23 element, which is a 5x5 matrix of the form:
# - [ R, v, t ],
# T is the SE3 element, which is a 4x4 matrix of the form:
# - [ R, t ],

def SE23_trim_to_SE3(X):
    T = np.eye(4)
    T[:3, :3] = X[:3, :3]
    T[:3, 3] = X[:3, 4]
    return T

def SE23_from_w_v_t(w, v, t):
    X = np.eye(5)
    # w = is omega x dt -> already the rotation vector, not angular velocity vector.
    X[:3, :3] = get_SO3_rotation_matrix(w)
    X[:3, 3] = v
    X[:3, 4] = t
    return X

def SIM23_from_a_w_dt(X, a, w, dt):
    # Note this is for small increments only
    # a = acceleration, w = angular velocity expressed in the body frame
    # TODO: This gives actually not SE23, but a SIM23 element. (Automorphisme of SE23)
    dX = np.eye(5)
    dX[:3, :3] = get_SO3_rotation_matrix(w, dt)
    dX[:3, 3] = a * dt
    dX[:3, 4] = 0.5 * a * dt**2
    dX[3,4] = dt #This makes dX not element of SE23, but of SIM23. Needed to make dX independent of the previous value.
    return dX

def SIM23_from_v_w_dt(X, v, w, dt):
    # In case we have velocity and not acceleration (Normally then we should use SE(3), but I want code that can be used for both cases)
    # v = velocity, w = angular velocity expressed in the body frame
    R = X[:3,:3]

    # a = (dR@v - X[:3,3])/dt
    a = (v - np.transpose(R)@X[:3,3])/dt
    return SIM23_from_a_w_dt(X, a, w, dt), a

def SE23_from_a_w_dt(X, a, w, dt):
    # Note this is for small increments only
    # a = acceleration, w = angular velocity expressed in the body frame
    dX = np.eye(5)
    dX[:3, :3] = get_SO3_rotation_matrix(w, dt)
    dX[:3, 3] = a * dt
    dX[:3, 4] = 0.5 * a * dt**2 + X[:3,3]*dt # Requires knowledge of previous velocity, so dX is not independent of the previous value. This is the actual SE23 element, but it is not independent of the previous value, which makes it less useful for covariance propagation.
    return dX

def SE23_from_v_w_dt(X, v, w, dt):
    # Note this is for small increments only
    # V = velocity, w = angular velocity expressed in the body frame
    R = X[:3, :3]
    a = (v - np.transpose(R) @ X[:3, 3]) / dt
    return SE23_from_a_w_dt(X, a, w, dt)

def SE23_from_t_v_rot(t, v, rot_vec):
    X = np.eye(5)
    X[:3, :3] = get_SO3_rotation_matrix(rot_vec)
    X[:3, 3] = v
    X[:3, 4] = t
    return X

def SE23_inverse(X):
    R = X[:3, :3]
    X_inv = np.eye(5)
    X_inv[:3, :3] = R.T
    X_inv[:3, 3] = -R.T @ X[:3, 3]
    X_inv[:3, 4] = -R.T @ X[:3, 4]
    return X_inv

def SE23_from_eps(eps):
    w = np.array([eps[2,1], eps[0,2], eps[1,0]])
    v = eps[:3,3]
    t = eps[:3,4]
    return w, v, t

def SE23_from_SE3s(X1, T2,  t1, t2 ):
    # X1 and T2 have the same base: X1 = X_OR1, T2 = T_OR2. We want to find X2 = X_OR2, so we can find the transformation from X1 to X2, which is the transformation from R1 to R2, which is what we want to find.
    # Returns X2 and dX = X1^-1 @ X2, which is the transformation from R1 to R2.
    X2 = np.eye(5)
    X2[:3,:3] = T2[:3,:3]
    X2[:3,4] = T2[:3,3]
    v2 = (T2[:3,-1] - X1[:3, -1]) / (t2 - t1)
    dv2 = (v2 - X1[:3,3])
    X2[:3,3] = X1[:3,:3]@dv2
    dX  = SE23_inverse(X1)@X2
    return X2, dX

# def integrate_se23(,eps, dt):


