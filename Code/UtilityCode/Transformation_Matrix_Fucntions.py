import numpy as np
import quaternion as quaternion
from Code.UtilityCode.utility_fuctions import limit_angle
import warnings
# warnings.filterwarnings("error")
def transformation_matrix_from_R(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = t
    return T

def transformation_matrix_from_q(q, t):
    T = np.eye(4)
    T[:3,:3] = quaternion.as_rotation_matrix(quaternion.from_float_array(q))
    T[:3,-1] = t
    return T

def transformation_matrix_from_4D_t(t):
    """
    t is a transfromation vector where the first 3 are the coordinates and the last element is the orientation arround the z-axis.
    """
    return transformation_matrix_from_rot_vect(t[3]*np.array([0,0,1]), t[:3])

def transformation_matrix_from_rot_vect(w, t):
    T = np.eye(4)
    T[:3,:3] = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(w))
    T[:3,-1] = t
    return T

def get_translation(T):
    return T[:3,-1]

def get_rotation(T):
    return T[:3,:3]

def get_quaternion(T):
    return quaternion.as_float_array(quaternion.from_rotation_matrix(T[:3,:3]))

def get_rotation_vector(T):
    # try:
    w = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(T[:3,:3]))
    w_amplitutede = np.linalg.norm(w)
    if w_amplitutede == 0:
        return np.zeros(3)
    w_unit = w/np.linalg.norm(w)
    w_amplitutede = limit_angle(w_amplitutede)
    w = w_unit*w_amplitutede
    return w
    # except ValueError:
    #     pass
    # except DeprecationWarning:
    #     pass
    # except RuntimeWarning:
    #     warnings.warn("Rotation vector is not unique")
    #     w = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(T[:3,:3]))
    #     w_amplitutede = np.linalg.norm(w)
    #     print(w, w_amplitutede)
    #     return np.zeros(3)

def inv_transformation_matrix(T):
    R = T[:3,:3]
    T_inv = np.eye(4)
    T_inv[:3,:3] = R.T
    T_inv[:3,-1] = -R.T @ T[:3,-1]
    return T_inv


if __name__ =="__main__":
    m = np.array([[0.00841304382668272, 0.9999593387705992, -0.0032467674747619813, 0.06894152758200835],
                  [0.9999106295642108, -0.008446281862132154, -0.010363069391232478, 0.0014153612943595456],
                  [-0.010390071129323313, -0.0031592923527704654, -0.9999410309082012, -0.02218574324172486],
                  [0.0, 0.0, 0.0, 1.0]])
    print(inv_transformation_matrix(m))

    m2 = np.array([[0.00939559657427575, 0.9999397270390503, -0.005680233629263832, -0.006468132734210166],
                   [0.9999445048770904, -0.009422409797897041, -0.004712256329086513, 0.0015585677697427032],
                   [0.004765493796447525, -0.005635643944577747, -0.9999727644222147, -0.022454074444970752],
                   [0.0, 0.0, 0.0, 1.0]])
    print(inv_transformation_matrix(m2))