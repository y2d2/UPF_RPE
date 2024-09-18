import numpy as np
from Code.Simulation.RobotClass import NewRobot
from Code.BaseLines.QCQP import QCQP
import Code.UtilityCode.Transformation_Matrix_Fucntions as TMF
from Code.UtilityCode.utility_fuctions import limit_angle
import matplotlib.pyplot as plt

class QCQP_Log:
    def __init__(self, QCQP : QCQP, host: NewRobot, connected: NewRobot):
        self.qcqp = QCQP
        self.host = host
        self.connected = connected

        self.estimated_ca_position = np.empty((0 , 3))
        self.estimated_ca_heading = []
        self.x_ca_r_error = []
        self.x_ca_r_heading_error = []

        self.calculation_time = []

        self.results = {"QCQP" : {"error_x_relative": [], "error_h_relative": [], "calculation_time": []}}

        self.data_logged = False

    def log(self, time_step, calculation_time=0):
        self.data_logged = True
        self.calculation_time.append(calculation_time)
        self.results["QCQP"]["calculation_time"].append(calculation_time)

        self.calculate_errors(time_step)
        self.calculate_estimated_trajectory(time_step)

    def calculate_errors(self, time_step):
        T_o_si = TMF.transformation_matrix_from_4D_t(np.append(self.host.x_real[time_step], np.array(self.host.h_real[time_step])))
        T_o_sj = TMF.transformation_matrix_from_4D_t(np.append(self.connected.x_real[time_step], np.array(self.connected.h_real[time_step])))
        T_si_sj = TMF.inv_transformation_matrix(T_o_si) @ T_o_sj
        t_sj_sj = TMF.get_4D_t_from_matrix(T_si_sj)

        x_ca_r_error = np.linalg.norm(t_sj_sj[:3] - self.qcqp.t_si_sj[:3])
        self.x_ca_r_error.append(x_ca_r_error)
        x_ca_r_heading_error = np.abs(limit_angle(t_sj_sj[3] - self.qcqp.t_si_sj[3]))
        self.x_ca_r_heading_error.append(x_ca_r_heading_error)

        self.results["QCQP"]["error_x_relative"].append(x_ca_r_error)
        self.results["QCQP"]["error_h_relative"].append(x_ca_r_heading_error)

    def calculate_estimated_trajectory(self, time_step):
        T_o_si = TMF.transformation_matrix_from_4D_t(np.append(self.host.x_real[time_step], np.array(self.host.h_real[time_step])))
        T_si_sj = TMF.transformation_matrix_from_4D_t(self.qcqp.t_si_sj)
        T_o_sj = T_o_si @ T_si_sj
        t_o_sj = TMF.get_4D_t_from_matrix(T_o_sj)

        self.estimated_ca_position = np.append(self.estimated_ca_position, t_o_sj[:3].reshape(1, 3), axis=0)
        self.estimated_ca_heading.append(t_o_sj[3])

    def plot_self(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(2, 1)
        ax[0].plot(self.x_ca_r_error)
        # ax[0].set_xlabel("Time step")
        ax[0].set_ylabel("Position error [m]")
        ax[0].legend()
        ax[0].grid(True)
        ax[1].plot(self.x_ca_r_heading_error)
        ax[1].set_xlabel("Time step")
        ax[1].set_ylabel("Orientation error [(rad)]")
        ax[1].legend()
        ax[1].grid(True)


    def plot_corrected_estimated_trajectory(self, ax, color="k", alpha=1, linestyle="--", marker="", label=None,
                                            i=-1, history=None):
        try:
            if history is None or history > i:
                self.plot_trajectory(self.estimated_ca_position, ax, color, alpha, linestyle, marker, label)
            else:
                j = i - history
                if j < 1:
                    j = 1
                self.plot_trajectory(self.estimated_ca_position, ax, color, alpha, linestyle, marker, label)
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


    def plot_estimated_pose(self, ax, id, i=-1):
        pass