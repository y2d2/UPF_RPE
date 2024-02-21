import numpy as np
import UtilityCode.Transformation_Matrix_Fucntions as TMF

class RealRobot():
    def __init__(self, name):
        self.name = name
        self.T = np.eye(4)
        self.Q = np.zeros(4)

    def integrate_odometry(self, DT, DQ):
        DQ[:3, :3] = TMF.get_rotation(self.T)@DQ[:3,:3]
        self.Q = DQ + self.Q
        self.T = self.T @ DT

    def reset_integration(self):
        x = np.concatenate((TMF.get_translation(self.T),TMF.get_rotation_vector(self.T)))
        Q = self.Q.copy()
        self.T = np.eye(4)
        self.Q = np.zeros(4)

        return x, Q