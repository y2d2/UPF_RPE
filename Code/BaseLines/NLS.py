from scipy.optimize import root
from scipy.spatial.distance import mahalanobis
import numpy as np

import matplotlib.pyplot as plt

from Code.UtilityCode.utility_fuctions import limit_angle, \
    get_4d_rot_matrix, transform_matrix, get_states_of_transform


class NLS:
    """
    Will use the method as described in the paper:
    T. Ziegler, M. Karrer, P. Schmuck, and M. Chli, “Distributed formation
    estimation via pairwise distance measurements,” IEEE Robotics and
    Automation Letters, vol. 6, no. 2, pp. 3017–3024, 2021

    However adapted to the 2 drone case (and so not distributed.)
    But it requires a commen reference frame.

    """

    def __init__(self, agents, horizon=1, sigma_uwb=0.1):
        """
        x_0 will be a vector of the initial positions of the drones in the common reference frame.
        x_0 will have shape (m,4) where m is the number of drones.
        """
        # number of frames
        self.horizon = horizon

        # Dict of agents (names + drone)
        # number of agents
        self.m = len(agents)
        x_0 = np.zeros((self.m, 4))
        self.names = []
        self.agents = []
        i = 0
        for agent in agents:
            self.names.append(agent)
            self.agents.append(agents[agent])
            x_0[i] = np.concatenate((agents[agent].x_start, np.array([agents[agent].h_start])))
            i += 1

        # Input should be a list of start positions of the different drones in the common reference frame.

        # list of distance measurements:
        self.d = np.zeros((self.horizon, self.m, self.m))
        # list of odom measurements:
        self.x_odom = np.zeros((self.horizon, self.m, 4))
        self.q = np.zeros((self.horizon, self.m, 4, 4))

        # list of estimated transformations:
        self.x_origin = np.ones((self.horizon, self.m, 4))
        for s in range(self.horizon):
            self.x_origin[s] = x_0
            for i in range(self.m):
                self.q[s, i] = np.eye(4) * 1e-6
        self.calculate_original_d()

        self.x = self.x_origin[0]

        self.x_rel = np.zeros((self.m, self.m, 4))

        # sigma_variables
        self.vi_uwb = np.array([sigma_uwb ** -2])
        # single_agent_cov = np.eye(4) * 1.

        self.x_cov = np.zeros((self.horizon * self.m * 4, self.horizon * self.m * 4))
        # for i in range(self.n * self.m):
        #     self.x_cov[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = single_agent_cov
        self.res = []

        self.nls_logger = None

    def init_logging(self, nls_logger):
        self.nls_logger = nls_logger

    def set_best_guess(self, dict_of_best_guesses):
        for agent in dict_of_best_guesses:
            i = self.names.index(agent)
            for j in range(self.horizon):
                self.x_origin[j, i] = dict_of_best_guesses[agent]
            self.x[i] = dict_of_best_guesses[agent]

    def calculate_original_d(self):
        for s in range(self.horizon):
            for i in range(self.m):
                for k in range(self.m - i - 1):
                    j = i + k + 1
                    self.d[s, i, j] = np.linalg.norm(self.x_origin[s, i, :3] - self.x_origin[s, j, :3])

    # --- Update Functions

    def update(self, d, dx_odom, q_odom):
        try:
            x_origin_ravel = self.x_origin.copy()
            x_origin_ravel = np.ravel(x_origin_ravel)
            x = self.x_origin.copy()
            self.add_measurement(d, dx_odom, q_odom)
            x = np.ravel(x)
            sol = root(self.optimise, x, method="lm")
            self.x_cov = sol.cov_x.copy()
            x = sol.x.reshape(self.horizon, self.m, 4)
            self.x_origin = x
            self.calculate_relative_poses()
        except AttributeError:
            print("NLS did not converged")


    def add_measurement(self, d, dx_odom, q_odom):
        x_odom = np.zeros((self.m, 4))
        for i in range(self.m):
            x_odom[i, :] = self.x_odom[-1, i, :] + get_4d_rot_matrix(self.x_odom[-1, i, -1]) @ dx_odom[i, :]
        self.x_odom = np.vstack((self.x_odom, x_odom.reshape(1, *x_odom.shape)))
        self.d = np.vstack((self.d, d.reshape(1, *d.shape)))
        self.q = np.vstack((self.q, q_odom.reshape(1, *q_odom.shape)))

        # Remove old odom measurements:
        if self.x_odom.shape[0] > self.horizon:
            self.x_odom = np.delete(self.x_odom, 0, 0)
            self.d = np.delete(self.d, 0, 0)
            self.q = np.delete(self.q, 0, 0)

    def optimise(self, x):
        x_test = x.copy()
        x_test = x_test.reshape(self.horizon, self.m, 4)
        res = []
        res = self.calculate_mesurement_error(x_test, res)
        res = self.calculate_drift_error(x_test, res)
        res = self.calculate_prior_error(x, res)
        return res

    def calculate_prior_error(self, x, res):
        x_prev = self.x_origin.copy()
        x_prev = np.ravel(x_prev)
        dx = x - x_prev
        for i in range(self.horizon * self.m * 4):
            if self.x_cov[i, i] != 0:
                dx[i] = mahalanobis(np.array([0]), np.array([dx[i]]), np.array([1 / self.x_cov[i, i]]))
            res.append(dx[i])
        return res

    def calculate_mesurement_error(self, x_origin, res):
        x = np.zeros((self.horizon, self.m, 4))
        for s in range(self.horizon):
            for i in range(self.m):
                x[s, i] = x_origin[s, i] + get_4d_rot_matrix(x_origin[s, i, -1]) @ self.x_odom[s, i, :]
        for s in range(self.horizon):
            for i in range(self.m):
                for k in range(self.m - i - 1):
                    j = i + k + 1
                    distance = np.linalg.norm(x[s, i] - x[s, j])
                    error = mahalanobis(np.array([self.d[s, i, j]]), np.array([distance]), self.vi_uwb)
                    res.append(error)
        return res

    def calculate_drift_error(self, x, res):
        for s in range(self.horizon):
            for i in range(self.m):
                dx = x[s, i] - self.x_origin[s, i]
                dx[3] = limit_angle(dx[3])
                VI = np.linalg.inv(self.q[s, i])
                error = mahalanobis(dx, np.zeros(4), VI)
                res.append(error)
        return res

    # --- Calculate relative poses.
    def calculate_poses(self):
        for i in range(self.m):
            f = get_4d_rot_matrix(self.x_origin[-1, i, -1])
            self.x[i] = self.x_origin[-1, i] + f @ self.x_odom[-1, i]

    def calculate_relative_poses(self):
        self.calculate_poses()
        # Todo: Why do I go over the frames s, It seems s is not even used?
        # for s in range(self.n):
        for i in range(self.m):
            for k in range(self.m - i - 1):
                j = i + k + 1

                x_rel_ij = self.x[j] - self.x[i]
                x_rel_ji = - x_rel_ij

                fi = get_4d_rot_matrix(self.x[i, -1])
                x_rel_ij = fi.T @ x_rel_ij

                fj = get_4d_rot_matrix(self.x[j, -1])
                x_rel_ji = fj.T @ x_rel_ji

                self.x_rel[i, j] = x_rel_ij
                self.x_rel[j, i] = x_rel_ji


class NLSDataLogger:
    def __init__(self, nls_solver: NLS):
        self.nls_solver = nls_solver
        self.n_agents = self.nls_solver.m
        self.x_rel = np.zeros((1, self.n_agents, self.n_agents, 4))
        self.estimated_ca_position  = np.empty((1, 2, 3))
        self.x_ca_r_error = np.zeros((1, self.n_agents, self.n_agents))
        self.x_ca_r_heading_error = np.zeros((1, self.n_agents, self.n_agents))

        self.calculation_time = []

        self.results = {}
        for name in self.nls_solver.names:
            self.results[name] = {"NLS" : {"error_x_relative": [], "error_h_relative": [],
                                           "calculation_time": []}}

        self.data_logged = False

    def log(self, t, calculation_time=0):
        self.data_logged = True
        self.calculation_time.append(calculation_time)
        x_rel = np.zeros((self.n_agents, self.n_agents, 4))
        error_rel = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for k in range(self.n_agents - i - 1):
                j = i + k + 1
                agent_i = self.nls_solver.agents[i]
                agent_j = self.nls_solver.agents[j]

                x_rel_ij = agent_j.x_real[t] - agent_i.x_real[t]
                x_rel_ij = np.append(x_rel_ij, np.array([agent_j.h_real[t] - agent_i.h_real[t]]))
                x_rel_ji = - x_rel_ij

                fi = get_4d_rot_matrix(agent_i.h_real[t])
                x_rel_ij = fi.T @ x_rel_ij

                fj = get_4d_rot_matrix(agent_j.h_real[t])
                x_rel_ji = fj.T @ x_rel_ji

                x_rel[i, j] = x_rel_ij
                x_rel[j, i] = x_rel_ji
        error = self.nls_solver.x_rel - x_rel
        error_position = np.linalg.norm(error[:, :, :3], axis=-1)
        self.x_ca_r_error = np.append(self.x_ca_r_error, error_position.reshape(1, *error_position.shape), axis=0)
        error_h = np.abs(error[:, :, -1])
        for i in range(self.n_agents):
            for k in range(self.n_agents):
                error_h[i,k] = np.abs(limit_angle(error_h[i,k]))
        self.x_ca_r_heading_error = np.append(self.x_ca_r_heading_error, error_h.reshape(1, *error_h.shape), axis=0)

        # for i in range(self.n_agents):
        #     for k in range(self.n_agents - i - 1):
        #         j = i + k + 1
        self.results[self.nls_solver.names[0]]["NLS"]["error_x_relative"].append(self.x_ca_r_error[-1, 0, 1])
        self.results[self.nls_solver.names[0]]["NLS"]["error_h_relative"].append(np.abs(limit_angle(self.x_ca_r_heading_error[-1, 0, 1])))
        self.results[self.nls_solver.names[0]]["NLS"]["calculation_time"].append(self.calculation_time[-1]/2)

        self.results[self.nls_solver.names[1]]["NLS"]["error_x_relative"].append(self.x_ca_r_error[-1, 1, 0])
        self.results[self.nls_solver.names[1]]["NLS"]["error_h_relative"].append(np.abs(limit_angle(self.x_ca_r_heading_error[-1, 1, 0])))
        self.results[self.nls_solver.names[1]]["NLS"]["calculation_time"].append(self.calculation_time[-1] / 2)

        self.x_rel = np.append(self.x_rel, self.nls_solver.x_rel.reshape(1, *self.nls_solver.x_rel.shape), axis=0)
        self.calculate_pose(t)
    def plot_self(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(2, 1)
        ax[0].plot(self.x_ca_r_error[:, 0, 1], label="Agent 1")
        ax[0].plot(self.x_ca_r_error[:, 1, 0], label="Agent 2")
        ax[0].legend()
        ax[0].grid(True)
        ax[1].plot(self.x_ca_r_heading_error[:, 0, 1], label="Agent 1")
        ax[1].plot(self.x_ca_r_heading_error[:, 1, 0], label="Agent 2")
        ax[1].legend()
        ax[1].grid(True)

    def calculate_pose(self, i):
        #TODO: adapt to multi agent? (Only made for 2 agents.)
        a0_p_real = self.nls_solver.agents[0].x_real[i]
        a0_h_real = self.nls_solver.agents[0].h_real[i]
        a1_p_real = self.nls_solver.agents[1].x_real[i]
        a1_h_real = self.nls_solver.agents[1].h_real[i]

        t_G_s0 = np.append(a0_p_real, np.array([a0_h_real]))
        t_G_s1 = np.append(a1_p_real, np.array([a1_h_real]))
        T_G_s0 = transform_matrix(t_G_s0)
        T_G_s1 = transform_matrix(t_G_s1)

        t_s0_s1 = self.x_rel[-1, 0, 1]
        est_T_G_s1 = T_G_s0 @ transform_matrix(t_s0_s1)
        est_t_G_s1 = get_states_of_transform(est_T_G_s1)

        t_s1_s0 = self.x_rel[-1, 1, 0]
        est_T_G_s0 = T_G_s1 @ transform_matrix(t_s1_s0)
        est_t_G_s0 = get_states_of_transform(est_T_G_s0)

        est_t_s = np.array([est_t_G_s1, est_t_G_s0])
        self.estimated_ca_position = np.append(self.estimated_ca_position, est_t_s[:, :3].reshape(1, 2, 3), axis=0)

    def plot_corrected_estimated_trajectory(self, ax, agent=0, color="k", alpha=1, linestyle="--", marker="", label=None, i=-1, history=None):
        try:
            if history is None or history > i:
                self.plot_trajectory(self.estimated_ca_position[ 1:i,agent,:], ax, color, alpha, linestyle, marker, label)
            else:
                j = i - history
                if j < 1:
                    j = 1
                self.plot_trajectory(self.estimated_ca_position[j:i, agent, :], ax, color, alpha, linestyle, marker, label)
        except IndexError:
            print("index error")

    def plot_trajectory(self, data, ax, color="k", alpha=1, linestyle="-", marker="", label=None):
        if self.data_logged:
            ax.plot3D(data[:, 0], data[:, 1], data[:, 2],
                      marker=marker, alpha=alpha, linestyle=linestyle, label=label, color=color)
            ax.plot3D(data[0, 0], data[0, 1], data[0, 2],
                      marker="o", alpha=alpha, color=color)
            ax.plot3D(data[-1, 0], data[-1, 1], data[-1, 2],
                      marker="x", alpha=alpha, color=color)

    def plot_estimated_pose(self,ax, id, i=-1):

        pass