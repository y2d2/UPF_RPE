from matplotlib.gridspec import GridSpec

from Code.BaseLines.NLS import NLS
import numpy as np
import matplotlib.pyplot as plt
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix, limit_angle, transform_matrix, get_states_of_transform
import copy

class NLSDataLogger:
    def __init__(self, nls_solver: NLS):
        self.nls_solver = nls_solver
        self.n_agents = self.nls_solver.m
        self.x_rel = np.zeros((1, self.n_agents, self.n_agents, 4))
        self.estimated_ca_position  = np.empty((1, 2, 3))
        self.x_ca_r_error = np.zeros((1, self.n_agents, self.n_agents))
        self.x_ca_r_heading_error = np.zeros((1, self.n_agents, self.n_agents))

        self.calculation_time = []
        self.likelihood = []

        self.results = {}
        for name in self.nls_solver.names:
            self.results[name] = {"NLS" : {"error_x_relative": [], "error_h_relative": [],
                                           "calculation_time": []}}

        self.data_logged = False

    def log_data(self, t, calculation_time=0):
        self.data_logged = True
        self.calculation_time.append(calculation_time)
        self.likelihood.append(self.nls_solver.likelihood)
        x_rel = np.zeros((self.n_agents, self.n_agents, 4))
        error_rel = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for k in range(self.n_agents - i - 1):
                j = i + k + 1
                agent_i = self.nls_solver.agents_list[i]
                agent_j = self.nls_solver.agents_list[j]

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
        # ax[0].plot(self.x_ca_r_error[:, 1, 0], label="Agent 2")
        ax[0].legend()
        ax[0].set_ylabel("Error [m]")
        ax[0].grid(True)
        ax[1].plot(self.x_ca_r_heading_error[:, 0, 1], label="Agent 1")
        # ax[1].plot(self.x_ca_r_heading_error[:, 1, 0], label="Agent 2")
        ax[1].legend()
        ax[1].set_ylabel("Error [rad]")
        ax[1].grid(True)

    def calculate_pose(self, i):
        #TODO: adapt to multi agent? (Only made for 2 agents.)
        a0_p_real = self.nls_solver.agents_list[0].x_real[i]
        a0_h_real = self.nls_solver.agents_list[0].h_real[i]
        a1_p_real = self.nls_solver.agents_list[1].x_real[i]
        a1_h_real = self.nls_solver.agents_list[1].h_real[i]

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

    def plot_corrected_estimated_trajectory(self, ax, agent=0, color="k", alpha=1., linestyle="--", marker="", label=None, i=-1, history=None):
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

    def plot_trajectory(self, data, ax, color="k", alpha=1., linestyle="-", marker="", label=None):
        if self.data_logged:
            ax.plot3D(data[:, 0], data[:, 1], data[:, 2],
                      marker=marker, alpha=alpha, linestyle=linestyle, label=label, color=color)
            ax.plot3D(data[0, 0], data[0, 1], data[0, 2],
                      marker="o", alpha=alpha, color=color)
            ax.plot3D(data[-1, 0], data[-1, 1], data[-1, 2],
                      marker="x", alpha=alpha, color=color)

    def plot_ca_corrected_estimated_trajectory(self, ax, color, alpha=0.1,linestyle=":", label=None, history=None):
        # self.plot_corrected_estimated_trajectory(ax, agent=0,
        #                                          color =color, alpha=alpha, linestyle=linestyle, label=label,
        #                                          history=history)
        self.plot_corrected_estimated_trajectory(ax, agent=1,
                                                 color=color, alpha=alpha, linestyle=linestyle, label=label,
                                                 history=history)

    def plot_graphs(self, fig= None):
        if fig is None:
            fig = plt.figure(figsize=(18, 10))

        fig.suptitle("NLS datalogger")
        ax = []
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        ax_3d = fig.add_subplot(gs[:3, :3], projection="3d")
        self.plot_corrected_estimated_trajectory(ax_3d, color="red", linestyle="--", label="NLS estimate")

        self.nls_solver.agents_list[0].set_plotting_settings(color="green", label="Estimating Agent")
        self.nls_solver.agents_list[0].plot_real_position(ax_3d)
        self.nls_solver.agents_list[1].set_plotting_settings(color="black", label="Estimated Agent")
        self.nls_solver.agents_list[1].plot_real_position(ax_3d)
        ax_3d.legend()
        ax_error =  [fig.add_subplot(gs[i, -1]) for i in range(2)]
        self.plot_self(ax_error)




            # , layout="constrained")
    def plot_estimated_pose(self,ax, id, i=-1):

        pass

    def copy(self, nls_solver):
        copy_NLS_DL = NLSDataLogger(nls_solver)
        copy_NLS_DL.x_rel = copy.deepcopy(self.x_rel)
        copy_NLS_DL.estimated_ca_position = copy.deepcopy(self.estimated_ca_position)
        copy_NLS_DL.x_ca_r_error = copy.deepcopy(self.x_ca_r_error)
        copy_NLS_DL.x_ca_r_heading_error = copy.deepcopy(self.x_ca_r_heading_error)
        copy_NLS_DL.calculation_time = copy.deepcopy(self.calculation_time)
        copy_NLS_DL.likelihood = copy.deepcopy(self.likelihood)
        copy_NLS_DL.results = copy.deepcopy(self.results)
        copy_NLS_DL.data_logged = copy.deepcopy(self.data_logged)
        return copy_NLS_DL