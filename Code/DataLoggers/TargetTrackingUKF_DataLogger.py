

from datetime import datetime
import numpy as np
import h5py
import copy
import matplotlib.pyplot as plt
from Code.ParticleFilter.TargetTrackingUKF import TargetTrackingUKF
from Code.UtilityCode.utility_fuctions import cartesianToSpherical, sphericalToCartesian, limit_angle, \
    get_4d_rot_matrix, transform_matrix, inv_transformation_matrix, \
    get_states_of_transform, get_covariance_of_transform
from Code.Simulation.RobotClass import NewRobot
from deprecated import deprecated



class UKFDatalogger():
    def __init__(self, hostAgent: NewRobot, connectedAgent: NewRobot, ukf: TargetTrackingUKF,
                 name="Undefined Testcase"):

        self.name = name
        self.host_agent: NewRobot = hostAgent
        self.connected_agent: NewRobot = connectedAgent
        self.ukf = ukf

        # Color properties for plotting:
        self.estimation_color = "steelblue"
        self.slam_color = "crimson"
        self.absolute_linestyle = "-"
        self.relative_linestyle = "-"
        self.mean_linestyle = "-"
        self.stds_linestyle = "--"

        if self.ukf is not None:
            self.kf_variables = self.ukf.kf_variables
            # ---- Save to file variables:
            self.save_bool = False
            self.now = None
            self.save_folder = None
            self.prior_save_name = ""
            self.posterior_save_name = ""

            # ---- logged variables:
            self.data_logged = False
            self.i = 0

            self.host_agent_trajectory = np.empty((0, 3))
            self.connected_agent_trajectory = np.empty((0, 3))
            self.connected_agent_heading = []

            self.ca_position_estimation = np.empty((0, 3))
            self.error_ca_position = []

            self.ca_heading_estimation = []
            self.error_ca_heading = []
            self.error_ca_position_slam = []
            self.error_ca_heading_slam = []

            self.spherical_estimated_relative_transformation = np.empty((0, 3))
            self.spherical_relative_transformation = np.empty((0, 3))
            self.error_spherical_relative_transformation_estimation = np.empty((0, 3))

            self.error_relative_transformation_est = []
            self.error_relative_heading_est = []
            self.error_relative_heading_slam = []
            self.error_relative_transformation_slam = []

            self.spherical_relative_start_position = np.empty((0, 3))
            self.spherical_estimated_relative_start_position = np.empty((0, 4))

            self.x = np.empty((0, self.kf_variables))
            self.stds = np.empty((0, self.kf_variables))
            self.stds_trajectory = []
            self.sigma_x_ca_0 = []
            self.sigma_x_ca = []
            self.sigma_x_ca_r = []
            self.sigma_h_ca = []
            self.likelihood = []
            self.weight = []

            self.ca_s = []
            self.ha_s = []

            self.estimated_ca_position = np.empty((0, 3))
            self.NIS = []
            self.NEES = []


    def save_graphs(self, folder="./"):
        self.now = datetime.now()
        self.save_bool = True
        self.save_folder = folder
        self.prior_save_name = "_".join(self.name.split(":")[0].split(" "))
        self.posterior_save_name = "_".join([str(self.now.hour), str(self.now.minute), str(self.now.second)])
        if self.save_folder[-1] != "/":
            self.save_folder += "/"

    def log_data(self, i):
        self.data_logged = True
        if self.ukf.time_i is None:
            self.i = i
        else:
            self.i = self.ukf.time_i
        self.log_spherical_data(self.i)
        self.log_cartesian_data(self.i)
        self.log_slam_data(self.i)
        self.log_variances()
        self.log_nis_nees()

    def log_variances(self):
        self.likelihood.append(self.ukf.kf.likelihood)
        self.weight.append(self.ukf.weight)
        self.x = np.append(self.x, np.reshape(self.ukf.kf.x, (1, self.kf_variables)), axis=0)
        std = np.array([np.sqrt(self.ukf.kf.P[i, i]) for i in range(self.ukf.kf.P.shape[0])])
        self.stds = np.append(self.stds, np.reshape(std, (1, self.kf_variables)), axis=0)
        self.stds_trajectory.append(np.linalg.norm(std[5:]))

        self.sigma_x_ca_0.append(self.ukf.sigma_x_ca_0)
        self.sigma_x_ca.append(self.ukf.sigma_x_ca)
        # self.sigma_x_ca_r.append(self.ukf.sigma_x_ca_r)
        self.sigma_h_ca.append(self.ukf.sigma_h_ca)

    def log_spherical_data(self, i):
        ca_p_real = self.connected_agent.x_real[i]
        ca_h_real = self.connected_agent.h_real[i]
        ha_p_real = self.host_agent.x_real[i]
        ha_h_real = self.host_agent.h_real[i]



        # relative_transformation = (ca_p_real - ha_p_real)
        # spherical_relative_transformation = cartesianToSpherical(relative_transformation)
        # spherical_relative_transformation[1] = limit_angle(spherical_relative_transformation[1] - ha_h_real)
        # relative_transformation_r = sphericalToCartesian(spherical_relative_transformation)
        #
        # error_relative_transformation_est = np.linalg.norm(
        #     relative_transformation_r - sphericalToCartesian(self.ukf.s_ca_r))
        t_G_si = np.append(ha_p_real, np.array([ha_h_real]))
        T_G_si = transform_matrix(t_G_si)
        T_si_G = inv_transformation_matrix(np.append(ha_p_real, np.array([ha_h_real])))
        T_G_sj = transform_matrix(np.append(ca_p_real, np.array([ca_h_real])))
        T_si_sj = T_si_G @ T_G_sj
        t_si_sj = get_states_of_transform(T_si_sj)
        e_t_si_sj = t_si_sj - self.ukf.t_si_sj

        error_relative_transformation_est = np.linalg.norm(e_t_si_sj[:3])
        self.error_relative_transformation_est.append(error_relative_transformation_est)

        spherical_relative_transformation = cartesianToSpherical(t_si_sj[:3])
        self.spherical_relative_transformation = np.append(self.spherical_relative_transformation,
                                                           spherical_relative_transformation.reshape(1, 3), axis=0)

        spherical_relative_transformation_estimation = np.reshape(cartesianToSpherical(self.ukf.t_si_sj[:3]), (1, 3))
        self.spherical_estimated_relative_transformation = np.append(self.spherical_estimated_relative_transformation,
                                                                     spherical_relative_transformation_estimation,
                                                                     axis=0)

        self.spherical_estimated_relative_start_position = np.append(self.spherical_estimated_relative_start_position,
                                                                     np.reshape(self.ukf.kf.x[:4], (1, 4)), axis=0)

        error_s = spherical_relative_transformation - spherical_relative_transformation_estimation
        error_s[0, 1] = limit_angle(error_s[0, 1])
        error_s[0, 2] = limit_angle(error_s[0, 2])
        error_s = np.reshape(np.abs(error_s), (1, 3))
        self.error_spherical_relative_transformation_estimation = np.append(
            self.error_spherical_relative_transformation_estimation, error_s, axis=0)

        est_T_G_sj = T_G_si @ transform_matrix(self.ukf.t_si_sj)
        est_t_G_sj = get_states_of_transform(est_T_G_sj)
        self.estimated_ca_position = np.append(self.estimated_ca_position, est_t_G_sj[:3].reshape(1, 3), axis=0)

    def log_cartesian_data(self, i):
        # self.ca_s.append(self.connected_agent.getLogData("Odom s")[i])
        # self.ha_s.append(self.host_agent.getLogData("Odom s")[i])
        ca_p_real = self.connected_agent.x_real[i]
        ha_p_real = self.host_agent.x_real[i]
        ca_h_real = self.connected_agent.h_real[i]
        ha_h_real = self.host_agent.h_real[i]

        self.host_agent_trajectory = np.append(self.host_agent_trajectory, np.reshape(ha_p_real, (1, 3)), axis=0)
        self.connected_agent_trajectory = np.append(self.connected_agent_trajectory, np.reshape(ca_p_real, (1, 3)),
                                                    axis=0)

        self.connected_agent_heading.append(limit_angle(ca_h_real))

        ca_position_estimation = self.ukf.t_oi_sj[:3]
        self.error_ca_position.append(np.linalg.norm(ca_position_estimation - ca_p_real))
        self.ca_position_estimation = np.append(self.ca_position_estimation, np.reshape(ca_position_estimation, (1, 3)),
                                                axis=0)
        self.ca_heading_estimation.append(limit_angle(self.ukf.t_oi_sj[-1]))
        self.error_ca_heading.append(np.abs(limit_angle(ca_h_real - self.ukf.t_oi_sj[-1])))

        relative_heading_estimation = limit_angle(self.ukf.t_si_sj[3])
        relative_heading = limit_angle(ca_h_real - ha_h_real)
        self.error_relative_heading_est.append(np.abs(limit_angle(relative_heading_estimation - relative_heading)))

    def log_slam_data(self, i):
        """
        This module tries to estimate the relative estimated error of the real RPE vs if the SLAM would be used without correction.
        It does include error due to drift in orientation. Therefore the estimation of the slam error should be the same between 2 agents,
        however the stiamtion of the RPE error should be different, due to different drifts in own orientation.
        """
        ca_p_slam = self.connected_agent.x_slam[i]
        ca_p_real = self.connected_agent.x_real[i]
        ha_p_slam = self.host_agent.x_slam[i]
        ca_h_slam = self.connected_agent.h_slam[i]
        ca_h_real = self.connected_agent.h_real[i]
        ha_h_slam = self.host_agent.h_slam[i]

        relative_transformation_slam = (ca_p_slam - ha_p_slam)
        spherical_relative_transformation = cartesianToSpherical(relative_transformation_slam)
        spherical_relative_transformation[1] = limit_angle(spherical_relative_transformation[1] - ha_h_slam)
        relative_transformation_slam = sphericalToCartesian(spherical_relative_transformation)

        relative_transformation = sphericalToCartesian(self.spherical_relative_transformation[-1])

        self.error_relative_transformation_slam.append(
            np.linalg.norm(relative_transformation_slam - relative_transformation))
        relative_heading_slam = limit_angle(ca_h_slam - ha_h_slam)
        relative_heading = limit_angle(self.connected_agent.h_real[i] - self.host_agent.h_real[i])
        self.error_relative_heading_slam.append(np.abs(limit_angle(relative_heading_slam - relative_heading)))

        self.error_ca_position_slam.append(np.linalg.norm(ca_p_slam - ca_p_real))
        self.error_ca_heading_slam.append(np.abs(limit_angle(ca_h_real - ca_h_slam)))

    def log_nis_nees(self):
        e = np.array([self.error_spherical_relative_transformation_estimation[-1][0]])
        nis = e @ self.ukf.kf.SI @ e
        self.NIS.append(nis)


    def plot_ukf_drift(self, ax):
        # plot drift of the position
        ax[0].set_ylabel("Position [m]")
        # ax[0].plot(self.error_ca_position, linestyle=self.relative_linestyle, color="darkgreen",
        #            label="ca position estimation error")
        ax[0].plot(self.error_relative_transformation_est, linestyle=self.relative_linestyle,
                   color=self.estimation_color, label="Error on position")
        # ax[0].plot(self.error_relative_transformation_slam, linestyle=self.relative_linestyle, color=self.slam_color,
        #            label="relative transformation slam")
        # ax[0].plot(self.sigma_x_ca_0, color=self.estimation_color, linestyle=self.stds_linestyle, label="std on the start position estimation")
        ax[0].plot(self.sigma_x_ca, color="red", linestyle=self.stds_linestyle,
                   label="Estimated std on the position by the UPF.")
        ax[0].grid(True)
        ax[1].legend()

        # plot drift of the heading
        ax[1].set_ylabel("Heading [(rad)]")
        ax[1].plot(self.error_relative_heading_est, linestyle=self.relative_linestyle,
                   color=self.estimation_color, label="Error on orientation")
        # ax[1].plot(self.error_relative_heading_slam, linestyle=self.relative_linestyle, color=self.slam_color)
        ax[1].plot(self.sigma_h_ca, color="red", linestyle=self.stds_linestyle,
                   label="Estimated std on the position by the UPF.")
        ax[1].grid(True)
        ax[1].legend()

    def create_3d_plot(self):
        if self.data_logged:
            fig = plt.figure()
            fig.set_size_inches(18.5, 10.5)
            fig.set_dpi(100)

            ax = plt.axes(projection="3d")
            self.plot_trajectory(self.host_agent_trajectory, ax, color="darkgreen",
                                 label="Host Agent real trajectory [m]")
            self.plot_trajectory(self.connected_agent_trajectory, ax, color="k",
                                 label="Connected Agent real trajectory [m]")
            self.plot_trajectory(self.ca_position_estimation, ax, color="r", linestyle="--",
                                 label="Estimated trajectory of the connected agent [m]")

            plt.title(" 3D trajectories \n" + self.name)
            ax.legend()

            if self.save_bool:
                save_name = self.prior_save_name + "_Trajectories_" + self.posterior_save_name

                plt.savefig(self.save_folder + save_name)

    #TODO: What is difference between self.estimated_ca_position and self.ca_position_estimation?
    def plot_ca_corrected_estimated_trajectory(self, ax, color="k", alpha=1, linestyle="--", marker="", label=None, i=-1, history=None):
        try:
            if history is None or history > i:
                self.plot_trajectory(self.estimated_ca_position[:i,:], ax, color, alpha, linestyle, marker, label)
            else:
                j = i - history
                if j < 0:
                    j = 0
                self.plot_trajectory(self.estimated_ca_position[j:i,:], ax, color, alpha, linestyle, marker, label)
        except IndexError:
            print("index error")
    def plot_ca_estimated_trajectory(self, ax, color="k", alpha=1, linestyle="--", marker="", label=None, i=-1):
        self.plot_trajectory(self.ca_position_estimation[:i,:], ax, color, alpha, linestyle, marker, label)

    def plot_trajectory(self, data, ax, color="k", alpha=1, linestyle="-", marker="", label=None):
        if self.data_logged:
            ax.plot3D(data[:, 0], data[:, 1], data[:, 2],
                      marker=marker, alpha=alpha, linestyle=linestyle, label=label, color=color)
            ax.plot3D(data[0, 0], data[0, 1], data[0, 2],
                      marker="o", alpha=alpha, color=color)
            ax.plot3D(data[-1, 0], data[-1, 1], data[-1, 2],
                      marker="x", alpha=alpha, color=color)

    def plot_relative_transformation(self, ax, color="k", alpha=1, linestyle="-", marker="", label="Real"):
        """
        ax should be a 4x1 array of axes.
        """
        if self.data_logged:
            self.plot_3d_data_on_1d_plots(ax[:3], self.spherical_relative_transformation, color=color,
                                          linestyle=linestyle, alpha=alpha, marker=marker, labels=[label] * 3)
            ax[3].plot(self.connected_agent_heading, color=color, linestyle=linestyle, alpha=alpha, marker=marker,
                       label=label)

    def plot_relative_transformation_estimation(self, ax, color="crimson", alpha=1, linestyle="--", marker="",
                                                label="Estimated"):
        """
        ax should be a 4x1 array of axes.
        """
        if self.data_logged:
            if ax[0].get_ylabel() == "":
                ylabel = ["Relative distance [m]", "Relative azimuth [rad]", "Relative altitude [rad]",
                          "Relative heading [rad]"]
                for i in range(4):
                    ax[i].set_ylabel(ylabel[i], color=color)
                    ax[i].tick_params(axis='y', labelcolor=color)

                self.plot_3d_data_on_1d_plots(ax[:3], self.spherical_estimated_relative_transformation, color=color,
                                              linestyle=linestyle, alpha=alpha, marker=marker, labels=[label] * 3)
                ax[3].plot(self.ca_heading_estimation, color=color, linestyle=linestyle, alpha=alpha, marker=marker,
                           label=label)

                self.plot_3d_data_on_1d_plots(ax, self.spherical_estimated_relative_start_position, color=color,
                                              linestyle=":", alpha=alpha, marker=marker,
                                              labels=[label + " start pose"] * 4)

    def plot_relative_transformation_error(self, ax, color="steelblue", label="Error"):
        if self.data_logged:
            if ax[0].get_ylabel() == "":
                ylabel = ["[m]", "[rad]", "[rad]", "[rad]"]
                for i in range(4):
                    ax[i].set_ylabel("std and error " + ylabel[i], color=color)
                    ax[i].tick_params(axis='y', labelcolor=color)

            self.plot_3d_data_on_1d_plots(ax[:3], self.error_spherical_relative_transformation_estimation, color=color,
                                          linestyle="-", labels=[label] * 3)
            ax[3].plot(self.error_ca_heading, color=color, linestyle="-", label=label)

            self.plot_3d_data_on_1d_plots(ax, self.stds[:, :4], color=color, linestyle=":",
                                          marker="", labels=["std start pose"] * 4)

            # ax[0].plot(self.stds_trajectory, color=color, marker="^", linestyle=":", label="std trajectory")
            ax[3].plot(self.stds[:, 4], color=color, marker="^", linestyle=":", label="std heading")

    def plot_position_error(self, ax, color="crimson"):
        if self.data_logged:
            if ax.get_ylabel() == "":
                ax.set_ylabel("Error [m]", color=color)
                ax.tick_params(axis='y', labelcolor=color)

            ax.plot(self.error_ca_position, color=color, linestyle="-", label="Error on absolute position estimation")
            ax.plot(self.error_relative_transformation_est, color=color, linestyle="--",
                    label="Error on the relative transformation estimation")
            ax.plot(self.connected_agent.x_error[1:], color="k", linestyle="-", label="Error from slam")

    def plot_position_stds(self, ax, color="steelblue"):
        if self.data_logged:
            ax.plot(self.sigma_x_ca_0, color=color, linestyle="-", label="std on the start position estimation")
            ax.plot(self.sigma_x_ca, color=color, linestyle="--", label="std on the relative transformation estimation")

            axt = ax.twinx()
            if axt.get_ylabel() == "":
                axt.set_ylabel("Relative error [%]", color="k")
            axt.plot(self.sigma_x_ca_r, color="k", linestyle=":",
                     label="relative std on the relative transformation estimation")

    def plot_ukf_states(self):
        colums = 2
        rows = int(np.ceil(self.kf_variables / colums))
        _, ax = plt.subplots(rows, colums)

        labels = ["r", "az", "alt", "h_0", "x", "y", "z", "h", "h_drift"]
        for row in range(rows):
            for colum in range(colums):
                try:
                    i = colum + row * colums
                    ax[row, colum].plot(self.x[:, i], label=labels[i])
                    ax[row, colum].legend(loc="upper right")
                    twax = ax[row, colum].twinx()
                    twax.plot(self.stds[:, i], linestyle=":", label="std")
                except IndexError:
                    pass

    def plot_error_graph(self):
        if self.data_logged:

            fig, ax = plt.subplots(4, 2)
            fig.set_size_inches(18.5, 10.5)
            fig.set_dpi(100)
            fig.suptitle("Error and Variance graph: \n " + self.name)

            self.plot_relative_transformation(ax[:4, 0], label="Real")
            self.plot_relative_transformation_estimation(ax[:4, 0], label="Estimated")

            error_ax = [ax[i, 0].twinx() for i in range(4)]
            self.plot_relative_transformation_error(error_ax, label="Error")

            for axis in error_ax:
                axis.legend(loc="upper right")
                axis.grid(True)
            for axis in ax[:, 0]:
                axis.legend(loc="upper left")
                axis.grid(True)

            # self.plot_position_error(ax[0, 1])
            # self.plot_position_stds(ax[0, 1])

            self.plot_ukf_drift(ax[:2, 1])

            label = ["r", "az", "alt", "h_0", "x", "y", "z", "h"]
            for i in range(8):
                ax[2, -1].plot(self.x[:, i], label=label[i])

            # ax[1, -1].plot(self.ca_s, color="black", label="Travelled distance of the connected agent [m]")
            # ax[1, -1].plot(self.ha_s, color="green", label="Traveled distance of the host agent [m]")

            ax[3, -1].plot(self.likelihood, color="black", label="Likelihood")
            ax[3, -1].plot(self.weight, color="red", label="Weight")
            ax[3, -1].plot(self.NIS, color="blue", label="NIS")

            for i in ax[:, -1]:
                # for i in a:
                i.legend()
                i.grid(True)
            if self.save_bool:
                save_name = self.prior_save_name + "_Error_and_variance_graph_" + self.posterior_save_name
                plt.savefig(self.save_folder + save_name)




    def plot_3d_data_on_1d_plots(self, ax, data, color, labels, alpha=1, marker="", linestyle="-"):
        for i, a in enumerate(ax):
            a.plot(data[:, i], color=color, linestyle=linestyle, alpha=alpha, marker=marker, label=labels[i])
            # a.legend()
            a.grid(True)

    def copy(self, ukf: TargetTrackingUKF):
        copyDL = UKFDatalogger(self.host_agent, self.connected_agent, ukf)

        copyDL.i = copy.deepcopy(self.i)

        copyDL.host_agent_trajectory = copy.deepcopy(self.host_agent_trajectory)
        copyDL.connected_agent_trajectory = copy.deepcopy(self.connected_agent_trajectory)
        copyDL.connected_agent_heading = copy.deepcopy(self.connected_agent_heading)

        copyDL.ca_position_estimation = copy.deepcopy(self.ca_position_estimation)
        copyDL.error_ca_position = copy.deepcopy(self.error_ca_position)

        copyDL.ca_heading_estimation = copy.deepcopy(self.ca_heading_estimation)
        copyDL.error_ca_heading = copy.deepcopy(self.error_ca_heading)
        copyDL.error_ca_position_slam = copy.deepcopy(self.error_ca_position_slam)
        copyDL.error_ca_heading_slam = copy.deepcopy(self.error_ca_heading_slam)

        copyDL.spherical_estimated_relative_transformation = copy.deepcopy(
            self.spherical_estimated_relative_transformation)
        copyDL.spherical_relative_transformation = copy.deepcopy(self.spherical_relative_transformation)
        copyDL.error_spherical_relative_transformation_estimation = copy.deepcopy(
            self.error_spherical_relative_transformation_estimation)

        copyDL.error_relative_transformation_est = copy.deepcopy(self.error_relative_transformation_est)
        copyDL.error_relative_heading_est = copy.deepcopy(self.error_relative_heading_est)
        copyDL.error_relative_heading_slam = copy.deepcopy(self.error_relative_heading_slam)
        copyDL.error_relative_transformation_slam = copy.deepcopy(self.error_relative_transformation_slam)

        copyDL.spherical_relative_start_position = copy.deepcopy(self.spherical_relative_start_position)
        copyDL.spherical_estimated_relative_start_position = copy.deepcopy(
            self.spherical_estimated_relative_start_position)

        copyDL.x = copy.deepcopy(self.x)
        copyDL.stds = copy.deepcopy(self.stds)
        copyDL.stds_trajectory = copy.deepcopy(self.stds_trajectory)
        copyDL.sigma_x_ca_0 = copy.deepcopy(self.sigma_x_ca_0)
        copyDL.sigma_x_ca = copy.deepcopy(self.sigma_x_ca)
        copyDL.sigma_x_ca_r = copy.deepcopy(self.sigma_x_ca_r)
        copyDL.sigma_h_ca = copy.deepcopy(self.sigma_h_ca)

        copyDL.likelihood = copy.deepcopy(self.likelihood)
        copyDL.weight = copy.deepcopy(self.weight)

        copyDL.ca_s = copy.deepcopy(self.ca_s)
        copyDL.ha_s = copy.deepcopy(self.ha_s)
        copyDL.estimated_ca_position = copy.deepcopy(self.estimated_ca_position)

        copyDL.data_logged = copy.deepcopy(self.data_logged)

        return copyDL

    def save_as_h5(self, name=None):
        if not name:
            name = self.prior_save_name + "_data_" + self.posterior_save_name
        hf = h5py.File(self.save_folder + name, 'w')
        hf.create_dataset("ha_trajectory", data=self.host_agent_trajectory.astype(np.float64))
        hf.create_dataset("ca_trajectory", data=self.connected_agent_trajectory.astype(np.float64))
        hf.create_dataset("ca_heading", data=np.array(self.connected_agent_heading).astype(np.float64))
        hf.create_dataset("likelihood", data=np.array(self.likelihood).astype(np.float64))
        hf.create_dataset("weight", data=np.array(self.weight).astype(np.float64))
        hf.create_dataset("stds", data=np.array(self.stds).astype(np.float64))
        hf.create_dataset("estimated_trjectory", data=self.ca_position_estimation.astype(np.float64))
        hf.create_dataset("estimated_heading", data=np.array(self.ca_heading_estimation).astype(np.float64))
        hf.create_dataset("error_heading", data=np.array(self.error_ca_heading).astype(np.float64))
        hf.create_dataset("error_pos_spherical",
                          data=self.error_spherical_relative_transformation_estimation.astype(np.float64))
        hf.create_dataset("estimated_pos_spherical",
                          data=self.spherical_estimated_relative_transformation.astype(np.float64))
        hf.create_dataset("real_pos_spherical", data=self.spherical_relative_transformation.astype(np.float64))
        string_dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("name", data=np.array([self.name], 'S4'), dtype=string_dt)
        hf.create_dataset("now", data=np.array(
            [self.now.year, self.now.month, self.now.day, self.now.hour, self.now.minute, self.now.second]).astype(
            np.float))

        hf.close()


def load_dataFile(filename):
    hf = h5py.File(filename, 'r')
    dl = UKFDatalogger(host_agent=None, connected_agent=None, name="", ukf=None)
    dl.host_agent_trajectory = np.array(hf.get("ha_trajectory"))
    dl.connected_agent_trajectory = np.array(hf.get("ca_trajectory"))
    dl.connected_agent_heading = np.array(hf.get("ca_heading"))
    dl.likelihood = np.array(hf.get("likelihood"))
    dl.weight = np.array(hf.get("weight"))
    dl.stds = np.array(hf.get("stds"))
    dl.estimatedPose = np.array(hf.get("estimated_trjectory"))
    dl.estimated_heading = np.array(hf.get("estimated_heading"))
    dl.error_heading = np.array(hf.get("error_heading"))
    dl.error_pos_spherical = np.array(hf.get("error_pos_spherical"))
    dl.estimatedPose_Spherical = np.array(hf.get("estimated_pos_spherical"))
    dl.realPose_Spherical = np.array(hf.get("real_pos_spherical"))
    dl.name = str(np.array(hf.get("name"))[0])
    print(dl.name)
    dl.data_logged = True
    now = hf.get("now")
    dl.now = datetime(year=int(now[0]), month=int(now[1]), day=int(now[2]),
                      hour=int(now[3]), minute=int(now[4]), second=int(now[5]))
