import matplotlib.pyplot as plt
import numpy as np
from RPE_Code import UtilityCode as TMF
import RPE_Code.UtilityCode.turtlebot4 as TB
from scipy.optimize import minimize, Bounds


class UWB:
    def __init__(self, name = "uwb"):
        #Raw data
        self.name = name
        self.d = []
        self.t = []
        self.delta_d = [0]
        self.thress = 1e3

        # Processed data
        self.sampled_d = []
        self.sampled_t = []
        self.real_d = []

    #--------------
    # Measurement and sampling
    #--------------
    def get_measuremend(self, msg_data):
        t = msg_data.header.stamp.sec + msg_data.header.stamp.nanosec * 1e-9
        d = msg_data.uwbrange.distance
        if d < self.thress:
            self.t.append(t)
            self.d.append(d)
            if len(self.d) > 1:
                self.delta_d.append(self.d[-1] - self.d[-2])

    def sample(self, frequency, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.t[0]
        if end_time is None:
            end_time = self.t[-1]
        dt = 1/frequency
        sample_time= start_time + dt
        prev_t = self.t[0]
        prev_d = self.d[0]
        for i in range(len(self.t)):
            if sample_time> end_time:
                break
            while self.t[i] > sample_time:
                self.sampled_t.append(sample_time)
                d = prev_d + (sample_time-prev_t) * (self.d[i] - prev_d) / (self.t[i] - prev_t)
                self.sampled_d.append(d)
                sample_time += dt
            prev_t = self.t[i]
            prev_d = self.d[i]

    #-----------------------
    # Transformations and errors
    #-----------------------
    def set_real_distances(self, ds):
        self.real_d = ds

    def evaluate_distances(self, tb2_p, tb3_p, i_0=0, i_e=-1):
        real_distances = np.linalg.norm(tb2_p - tb3_p, axis=1)
        error = np.abs(real_distances - self.sampled_d)
        mean = np.mean(error[i_0:i_e])
        std = np.std(error[i_0:i_e])
        return mean, std

    def optimisation_function(self, x, tb2, tb3, i_0=0, i_e=-1):
        R_0 = np.eye(3)
        T_uwb_tb2 = TMF.transformation_matrix_from_R(R_0, x[:3])
        T_uwb_tb3 = TMF.transformation_matrix_from_R(R_0, x[3:])
        tb2_p, _, _ = tb2.vicon_frame.get_corrected_transformation(T_uwb_tb2)
        tb3_p, _, _ = tb3.vicon_frame.get_corrected_transformation(T_uwb_tb3)
        m, std = self.evaluate_distances(tb2_p, tb3_p, i_0, i_e)
        print(x)
        print(m, std)
        return m**2 + std**2

    def optimise_uwb_T(self, tb2 : TB.Turtlebot4, tb3: TB.Turtlebot4, i_0=0, i_e=-1):
        x0 = np.zeros(6)
        bounds_val =  np.array([0.4, 0.4, 0.1, 0.4, 0.4, 0.1])
        bounds = Bounds(-bounds_val, bounds_val)
        res = minimize(self.optimisation_function, x0, method='nelder-mead',args=(tb2, tb3, i_0, i_e),
                       options={'xatol': 1e-8, 'disp': True}, bounds=bounds)
        return res

    def change_data(self, index, value, sigma_uwb):
        for i in range(index, len(self.sampled_d)):
            self.sampled_d[i] = self.real_d[i] + value + np.random.rand()*sigma_uwb

    #---------------
    # Data
    #---------------
    def load_raw_data(self, uwb):
        self.t = uwb.t
        self.d = uwb.d
        self.delta_d = uwb.delta_d
        self.thress = uwb.thress
        self.name = uwb.name

    def load_sampled_data(self, uwb):
        self.sampled_t = uwb.sampled_t
        self.sampled_d = uwb.sampled_d
        # self.real_d = uwb.real_d
        self.name = uwb.name
        self.thress = uwb.thress

    def split_data(self, id_0, id_1):
        uwb = UWB()
        uwb.name = self.name
        uwb.sampled_t = self.sampled_t[id_0:id_1]
        uwb.sampled_d = self.sampled_d[id_0:id_1]
        # uwb.delta_d = self.delta_d[id_0:id_1]
        uwb.thress = self.thress
        return uwb

    #--------------
    # Plot methods
    #---------------

    def plot(self):
        plt.figure()
        plt.title(self.name)
        plt.plot(self.t, self.d)
        plt.plot(self.t, self.delta_d)
        print(self.name)
        print("mean: ", np.mean(self.delta_d))
        print("std: ", np.std(self.delta_d))

    def plot_sampled(self):
        plt.figure()
        plt.title(self.name+" sampled")
        plt.plot(self.t, self.d, label="raw")
        plt.plot(self.sampled_t, self.sampled_d, '*', label="sampled")
        plt.legend()
        plt.grid()
        plt.figure()
        plt.title(self.name+" index")
        plt.plot( self.sampled_d, '*', label="sampled")
        plt.legend()
        plt.grid()

    def plot_real(self, factor=1, ax = None):
        if ax == None:
            plt.figure()
            ax = plt
        # plt.figure()
        # ax.title(self.name+" real")
        ax.plot([d for i, d in enumerate(self.real_d) if (i % factor == 0)], label="Real $d$ [m]",  linewidth=3, color="k")
        ax.plot([d for i, d in enumerate(self.sampled_d) if (i % factor == 0)], '--', color="tab:purple",  label=r"Measured $\tilde{d}$ [m]", linewidth=3)
        er = np.abs(np.array(self.sampled_d) - self.real_d)
        ax.plot([d for i, d in enumerate(er) if (i % factor == 0)], '--', color="tab:red", label=r"Error $\epsilon_{d} = |d - \tilde{d} |$ [m]", linewidth=3)
        print("mean error: ", np.mean(er), "std error: ", np.std(er))
        # ax.legend(fontsize=12, loc="upper left")
        # plt.grid()


    def plot_indices(self):
        plt.figure()
        plt.title(self.name + " indices")
        plt.plot(self.sampled_d, '*', label="sampled")
        plt.plot(self.sampled_d, '*', label="sampled")
        plt.grid()

