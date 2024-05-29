# Solve the following MIP:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import *
import numpy as np
import Code.UtilityCode.Transformation_Matrix_Fucntions as TMF

class QCQP:
    def __init__(self, horizon = 10, sigma_uwb=0.1):
        self.horizon = horizon
        self.sigma_uwb = 1.5*sigma_uwb

        # QCQP model
        # solution vector (see eq. 39 in paper)
        # x = [ t_x, t_y, t_z, cos(theta), sin(theta), t_x cos(theta) + t_y sin(theta), t_y cos(theta) - t_x sin(theta), t_x^2 + t_y^2 + t_z^2, 1]
        self.names = ["t_x", "t_y", "t_z", "cos(theta)", "sin(theta)", "t_x cos(theta) + t_y sin(theta)", "t_y cos(theta) - t_x sin(theta)", "t_x^2 + t_y^2 + t_z^2", "1"]

        # variables
        self.dt_i_s = np.empty((0, 4))
        self.dt_j_s = np.empty((0, 4))
        self.dij_s = np.empty(0)

        self.t_i = np.empty((0,4))
        self.t_j = np.empty((0,4))

        self.t_si_sj = np.zeros(4)

        # Variables for the optimization problem
        self.B = np.empty((0, 9))
        self.S = np.zeros((self.horizon, self.horizon))
        self.Sigma = self.sigma_uwb**2 * np.eye(self.horizon)
        self.PO = np.zeros((9,9))

        # Constrains see eq 45 in paper
        self.P1 = np.zeros((9,9), dtype = np.float64)
        self.P1[3,3] = 1
        self.P1[4,4] = 1
        self.r1 = 1

        self.P2 = np.zeros((9,9), dtype = np.float64)
        self.P2[0,3] = 1
        self.P2[1,4] = 1
        self.P2[8,5] = -1
        self.r2 = 0

        self.P3 = np.zeros((9,9), dtype = np.float64)
        self.P3[1,3] = 1
        self.P3[0,4] = -1
        self.P3[8,6] = -1
        self.r3 = 0

        self.P4 = np.zeros((9,9), dtype = np.float64)
        self.P4[0,0] = 1
        self.P4[1,1] = 1
        self.P4[2,2] = 1
        self.P4[7,8] = -1
        self.r4 = 0

        self.P5 = np.zeros((9,9), dtype = np.float64)
        self.P5[0,0] = 1
        self.P5[1,1] = 1
        self.P5[2,2] = 1
        self.r5 = None

    def prune_matrices(self):
        if len(self.dt_i_s) > self.horizon:
            self.dt_i_s = self.dt_i_s[1:]
            self.dt_j_s = self.dt_j_s[1:]
            self.dij_s = self.dij_s[1:]
            # self.B = self.B[1:]

    def calculate_trajectories(self):
        self.t_j = np.empty((0, 4))
        self.t_i = np.empty((0, 4))
        prev_T_i = np.eye(4)
        prev_T_j = np.eye(4)
        for i in range(len(self.dt_i_s)):
            DT_i = TMF.transformation_matrix_from_4D_t(self.dt_i_s[i])
            T_i = prev_T_i @ DT_i
            prev_T_i = T_i
            t_i = TMF.get_4D_t_from_matrix(T_i)
            self.t_i = np.append(self.t_i,  np.array([t_i]), axis=0)

            DT_j = TMF.transformation_matrix_from_4D_t(self.dt_j_s[i])
            T_j = prev_T_j @ DT_j
            prev_T_j = T_j
            t_j = TMF.get_4D_t_from_matrix(T_j)
            self.t_j = np.append(self.t_j, np.array([t_j]), axis=0)

        # print(self.t_i)
        # print(self.t_j)

    def calculate_B_matrix(self):
        # See equation 40  in paper.

        self.B = np.empty((0, 9))
        for i in range(len(self.t_i)):
            a_1 =-2 * self.t_i[i][0]
            a_2 =-2 * self.t_i[i][1]
            a_3 = 2 * (self.t_j[i][2]- self.t_i[i][2])
            a_4 = -2 * (self.t_j[i][0] * self.t_i[i][0] + self.t_j[i][1] * self.t_i[i][1])
            a_5 = 2 * (self.t_j[i][1] * self.t_i[i][0] - self.t_j[i][0] * self.t_i[i][1])
            a_6 = 2 * self.t_j[i][0]
            a_7 = 2 * self.t_j[i][1]
            a_8 = 1
            # how to calculate v_bar? v = - (2 d_ij n + n^2 )
            # I assume that the expected value of d_ij n = 0 since the expected value of n = 0.
            # However how can we calculate the expected value of n^2?
            # Chat gpt using E[X^2] = Var[X] + E[X]^2  (with X guassian distribution as n)
            # E[X^2] = sigma^2 + 0^2 = sigma^2
            s_i = self.dij_s[i] ** 2 - self.sigma_uwb ** 2
            a_9 = self.t_i[i][:3].T @ self.t_i[i][:3] + self.t_j[i][:3].T @ self.t_j[i][:3] - 2* self.t_i[i][2] * self.t_j[i][2] - s_i

            t_i = self.t_i[i]
            t_j = self.t_j[i]
            B_i = np.array([a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9]).reshape(1, 9)
            self.B = np.append(self.B, B_i, axis=0)
        # print(self.B)

    def calculate_S_matrix(self):
        # See equation 35 in paper.
        # It seems the Si,j is zero by construction makes sense if we assume i.i.d measurements.
        self.S= np.zeros((self.horizon, self.horizon))
        for i, d_i in enumerate(self.dij_s):
            self.S[i][i] = self.sigma_uwb ** 2 * (4 * d_i ** 2 + self.sigma_uwb ** 2)
            # self.S[i][i] = self.sigma_uwb ** 2


        # # Seems infeasible with this setup
        # for i, d_i in enumerate(self.dij_s):
        #     for j, d_j in enumerate(self.dij_s):
        #         if i != j:
        #             self.S[i][j] = self.sigma_uwb ** 2 * (4 * d_i * d_j + self.sigma_uwb ** 2)
        #         else:
        #             self.S[i][i] = self.sigma_uwb ** 2 *( 4 * d_i **2 + self.sigma_uwb ** 2)

        # print(self.S)

    def calculate_P0(self):
        self.calculate_S_matrix()
        self.calculate_B_matrix()
        if self.dij_s.size == self.horizon:
            self.PO = self.B.T @ np.linalg.inv(self.S) @ self.B
        return

    def update(self, dt_i, dt_j, dij):
        self.dt_i_s = np.append(self.dt_i_s, np.array([dt_i]), axis=0)
        self.dt_j_s = np.append(self.dt_j_s,  np.array([dt_j]), axis=0)
        self.dij_s = np.append(self.dij_s,  np.array([dij]), axis=0)

        if self.dij_s.size >= self.horizon:
            self.prune_matrices()
            self.r5 = self.dij_s[0] ** 2
            self.calculate_trajectories()
            self.calculate_P0()
            self.optimize()
            self.calculate_relative_pose()

    def calculate_relative_pose(self):
        try:
            theta = np.arctan2(self.x[4].X, self.x[3].X)
            t_origin = np.array([self.x[0].X, self.x[1].X, self.x[2].X, theta])
            T_oi_oj = TMF.transformation_matrix_from_4D_t(t_origin)

            T_si_oi = TMF.inv_transformation_matrix(TMF.transformation_matrix_from_4D_t(self.t_i[-1]))
            T_oj_sj = TMF.transformation_matrix_from_4D_t(self.t_j[-1])
            T_si_sj = T_si_oi @ T_oi_oj @ T_oj_sj
            self.t_si_sj = TMF.get_4D_t_from_matrix(T_si_sj)
        except:
            print("no solution found")

    def optimize(self):
        if self.dij_s.size == self.horizon:
            self.m = Model("qcqp")
            self.m.Params.NonConvex = 2
            self.m.Params.Threads = 1
            self.x = [self.m.addVar(name=name, vtype= GRB.CONTINUOUS) for name in self.names]

            #Bound sin and cos
            self.x[3].LB = -1.
            self.x[3].UB = 1.
            self.x[4].LB = -1.
            self.x[4].UB = 1.

            # Set the last one to 1
            self.x[8].UB = 1.
            self.x[8].LB = 1.

            dis = 100.
            # Set bounds for tx, ty, tz (Should not have to do this)
            self.x[0].LB = -dis
            self.x[1].LB = -dis
            self.x[2].LB = -dis

            self.x[0].UB = dis
            self.x[1].UB = dis
            self.x[2].UB = dis

            # "t_x cos(theta) + t_y sin(theta)", "t_y cos(theta) - t_x sin(theta)", "t_x^2 + t_y^2 + t_z^2",
            self.x[5].LB = -2*dis
            self.x[5].UB = 2*dis
            self.x[6].LB = -2*dis
            self.x[6].UB = 2*dis

            # Bound the norm
            self.x[7].LB = 0.
            self.x[7].UB = 3*dis**2



            obj = np.array(self.x).T @ self.PO @ np.array(self.x)
            self.m.setObjective(obj, GRB.MINIMIZE)
            # self.m.setMObjective(Q= self.PO, c= None, constant = 0, xQ_L = self.x, xQ_R = self.x, xc = None, sense = GRB.MINIMIZE)
            # self.m.addMQConstr(self.P1, c = None, sense= "=", rhs = self.r1, xQ_L = self.x, xQ_R = self.x, name = "c1")
            # self.m.addMQConstr(self.P2, c = None, sense= "=", rhs = self.r2, xQ_L = self.x, xQ_R = self.x, name = "c1")
            # self.m.addMQConstr(self.P3, c = None, sense= "=", rhs = self.r3, xQ_L = self.x, xQ_R = self.x, name = "c1")
            # self.m.addMQConstr(self.P4, c = None, sense= "=", rhs = self.r4, xQ_L = self.x, xQ_R = self.x, name = "c1")
            # self.m.addMQConstr(self.P5, c = None, sense= "=", rhs = self.r5, xQ_L = self.x, xQ_R = self.x, name = "c1")

            # self.m.addQConstr( self.x.T @ self.P1 @ self.x, GRB.EQUAL, self.r1, "c1")
            # self.m.addQConstr( self.x.T @ self.P2 @ self.x, GRB.EQUAL, self.r2, "c2")
            # self.m.addQConstr( self.x.T @ self.P3 @ self.x, GRB.EQUAL, self.r3, "c3")
            # self.m.addQConstr( self.x.T @ self.P4 @ self.x, GRB.EQUAL, self.r4, "c4")
            # self.m.addQConstr( self.x.T @ self.P5 @ self.x, GRB.EQUAL, self.r5, "c5")

            self.m.addQConstr(self.x[3]*self.x[3] + self.x[4]*self.x[4], GRB.EQUAL, 1, "c1")
            self.m.addQConstr(self.x[0]*self.x[3] + self.x[1]*self.x[4] - self.x[5]*self.x[8], GRB.EQUAL, 0, "c2")
            self.m.addQConstr(self.x[1]*self.x[3] - self.x[0]*self.x[4] - self.x[6]*self.x[8], GRB.EQUAL, 0, "c3")
            self.m.addQConstr(self.x[0]*self.x[0] + self.x[1]*self.x[1] + self.x[2]*self.x[2] - self.x[7]*self.x[8], GRB.EQUAL, 0, "c4")
            self.m.addQConstr(self.x[0]*self.x[0] + self.x[1]*self.x[1] + self.x[2]*self.x[2], GRB.EQUAL, self.r5, "c5")

            self.m.update()

            self.m.optimize()















#TODO add the correct formules from paper.
if __name__=="__main__":
    # Create a new model
    m = Model("matrix_qcqp")

    # Variables el R^(9x1)
    # tx, ty, tz, cos(theta), sin(theta), tx cos(theta) + ty sin(theta), ty cos(theta) - tx sin(theta), tx^2 + ty^2 + tz^2, 1

    # Create a 2D array of variables
    x = []
    for i in range(5):
        x.append([])
        for j in range(5):
            x[i].append(m.addVar(name="x_" + str(i) + "_" + str(j)))

    # Update model to integrate new variables
    m.update()

    #TODO: Calculate B matrix


    # TODO: Calculate the S matrix

    # Set objective: sum of all x_ij^2
    obj = quicksum(x[i][j] * x[i][j] for i in range(5) for j in range(5))
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraint: sum of all x_ij >= 1
    m.addConstr(quicksum(x[i][j] for i in range(5) for j in range(5)) >= 1, "c0")

    # Add non-convex quadratic constraint: sum of all x_ij^2 <= 1
    m.addQConstr(quicksum(x[i][j] * x[i][j] for i in range(5) for j in range(5)) <= 1, "qc0")

    # Set to non-convex
    m.Params.NonConvex = 2

    # Optimize model
    m.optimize()
