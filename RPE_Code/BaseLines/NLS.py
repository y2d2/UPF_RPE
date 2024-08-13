import copy

from numpy.linalg import LinAlgError
from scipy.optimize import root, least_squares
from scipy.spatial.distance import mahalanobis
import numpy as np

# from cyipopt import minimize_ipopt

import matplotlib.pyplot as plt

from RPE_Code.UtilityCode.utility_fuctions import limit_angle, \
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

    def __init__(self, agents, horizon=10, sigma_uwb=0.1):
        """
        x_0 will be a vector of the initial positions of the drones in the common reference frame.
        x_0 will have shape (m,4) where m is the number of drones.
        """
        # number of frames
        self.horizon = horizon
        self.likelihood = 1.0
        self.distances = 0.
        self.agents = agents
        # Dict of agents (names + drone)
        # number of agents
        self.m = len(agents)
        x_0 = np.zeros((self.m, 4))
        self.names = []
        self.agents_list = []
        i = 0
        for agent in agents:
            self.names.append(agent)
            self.agents_list.append(agents[agent])
            x_0[i] = np.concatenate((agents[agent].x_start, np.array([agents[agent].h_start])))
            i += 1

        # Input should be a list of start positions of the different drones in the common reference frame.

        # list of distance measurements:
        self.d = np.zeros((self.horizon, self.m, self.m))
        # list of odom measurements:
        self.x_odom = np.zeros((self.horizon, self.m, 4))
        self.x_odom_prev = np.zeros((self.m,4))
        self.q_prev = np.zeros((self.m, 4,4))
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
        self.sigma_uwb = sigma_uwb
        self.vi_uwb = np.array([self.sigma_uwb ** -2])
        # single_agent_cov = np.eye(4) * 1.

        #Give a bit of initial uncertainty otherwise to stiff and initial bad measurements can pull the solution of to much.
        self.x_cov = 0.01*np.ones((self.horizon * self.m * 4, self.horizon * self.m * 4))
        # for i in range(self.n * self.m):
        #     self.x_cov[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = single_agent_cov
        self.res = []

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

    def calculate_likelihood(self):
        dist_sq = self.distances ** 2
        det_cov = self.sigma_uwb**(2)
        norm_factor = (2 * np.pi) ** (1/ 2) * np.sqrt(det_cov)
        self.likelihood = (1 / norm_factor) * np.exp(-0.5 * dist_sq)

    # --- Update Functions
    def objective_function(self, x):
        return np.sum(np.array(self.optimise(x)) ** 2)

    def update(self, d, dx_odom, q_odom, update= True):
        try:

            # x_origin_ravel = self.x_origin.copy()
            # x_origin_ravel = np.ravel(x_origin_ravel)
            x = self.x_origin.copy()
            self.add_measurement(d, dx_odom, q_odom, update)
            x = np.ravel(x)
            if update:
                try:

                    # sol = minimize_ipopt(
                    #     fun=self.objective_function,
                    #     x0=x,
                    #     # bounds=(lb, ub),
                    #     options={'maxiter': 100, 'disp': 5}
                    # )
                    sol = root(self.optimise, x, method='lm')
                    self.x_cov = sol.cov_x.copy()
                    # self.x_cov = np.linalg.inv(sol.jac.T @ sol.jac)  # approximate covariance matrix
                    x = sol.x.reshape(self.horizon, self.m, 4)
                    self.calculate_likelihood()
                    self.x_origin = x
                    self.calculate_relative_poses()
                except LinAlgError as e:
                    self.calculate_relative_poses(converged=False)
                    print("LinAlgError")


        except AttributeError:
            print("NLS did not converged")


    def add_measurement(self, d, dx_odom, q_odom, update):
        for i in range(self.m):
            self.q_prev[i] = self.q_prev[i] + get_4d_rot_matrix(self.x_odom_prev[i, -1]) @ q_odom[i] @ get_4d_rot_matrix(self.x_odom_prev[i, -1]).T
            self.x_odom_prev[i, :] = self.x_odom_prev[i, :] + get_4d_rot_matrix(self.x_odom_prev[ i, -1]) @ dx_odom[i, :]



        if update:
            self.x_odom = np.vstack((self.x_odom, self.x_odom_prev.reshape(1, *self.x_odom_prev.shape)))
            self.d = np.vstack((self.d, d.reshape(1, *d.shape)))
            self.q = np.vstack((self.q, self.q_prev.reshape(1, *self.q_prev.shape)))
            self.q_prev = np.zeros((self.m, 4, 4))

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
        res = self.calculate_prior_error(x_test, res)
        return res

    def calculate_prior_error(self, x, res):
        x_prev = self.x_origin.copy()
        # x_prev = np.ravel(x_prev)
        dx = np.ravel(x - x_prev)
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
                    self.distances = error
        # return res

        # x = np.zeros((self.horizon, self.m, 4))
        # for s in range(self.horizon):
        #     for i in range(self.m):
        #         x[s, i] = x_origin[s, i] + get_4d_rot_matrix(x_origin[s, i, -1]) @ self.x_odom[s, i, :]
        # for s in range(self.horizon):
        #     for i in range(self.m):
        #         for k in range(self.m - i - 1):
        #             j = i + k + 1
        #             distance = np.linalg.norm(x[s, i] - x[s, j])
        #             error = mahalanobis(np.array([self.d[s, i, j]]), np.array([distance]), self.vi_uwb)
        #             res.append(error)
        return res

    def calculate_drift_error(self, x, res):
        for s in range(self.horizon):
            for i in range(self.m):
                dx = x[s, i] - self.x_origin[s, i]
                dx[3] = limit_angle(dx[3])
                VI = np.linalg.inv(self.q[s, i])
                error = mahalanobis(dx, np.zeros(4), VI)
                res.append(error)

        # for s in range(self.horizon):
        #     for i in range(self.m):
        #         dx = x[s, i] - self.x_origin[s, i]
        #         dx[3] = limit_angle(dx[3])
        #         VI = np.linalg.inv(self.q[s, i])
        #         error = mahalanobis(dx, np.zeros(4), VI)
        #         res.append(error)
        return res

    # --- Calculate relative poses.
    def calculate_poses(self):
        for i in range(self.m):
            f = get_4d_rot_matrix(self.x_origin[-1, i, -1])
            self.x[i] = self.x_origin[-1, i] + f @ self.x_odom[-1, i]

    def calculate_relative_poses(self, converged=True):
        self.calculate_poses()

        for i in range(self.m):
            for k in range(self.m - i - 1):
                j = i + k + 1
                if converged:
                    x_rel_ij = self.x[j] - self.x[i]
                    x_rel_ji = - x_rel_ij

                    fi = get_4d_rot_matrix(self.x[i, -1])
                    x_rel_ij = fi.T @ x_rel_ij

                    fj = get_4d_rot_matrix(self.x[j, -1])
                    x_rel_ji = fj.T @ x_rel_ji

                    self.x_rel[i, j] = x_rel_ij
                    self.x_rel[j, i] = x_rel_ji
                else:
                    self.x_rel[i, j] = np.array([np.NaN, np.NaN, np.NaN, np.NaN])
                    self.x_rel[j, i] = np.array([np.NaN, np.NaN, np.NaN, np.NaN])


    def copy(self):
        copy_NLS = NLS(self.agents, self.horizon, self.sigma_uwb)
        copy_NLS.x_origin = copy.deepcopy(self.x_origin)
        copy_NLS.x = copy.deepcopy(self.x)
        copy_NLS.x_rel = copy.deepcopy(self.x_rel)
        copy_NLS.d = copy.deepcopy(self.d)
        copy_NLS.x_odom = copy.deepcopy(self.x_odom)
        copy_NLS.x_odom_prev = copy.deepcopy(self.x_odom_prev)
        copy_NLS.q = copy.deepcopy(self.q)
        copy_NLS.q_prev = copy.deepcopy(self.q_prev)
        copy_NLS.x_cov = copy.deepcopy(self.x_cov)
        copy_NLS.vi_uwb = copy.deepcopy(self.vi_uwb)
        copy_NLS.likelihood = copy.deepcopy(self.likelihood)
        copy_NLS.distances = copy.deepcopy(self.distances)
        return copy_NLS

