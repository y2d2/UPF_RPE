import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from Code.UtilityCode.Measurement import create_experiment, create_experimental_sim_data

import unittest

import rosbags.rosbag2 as rb2
import unittest
from rosbags.serde import deserialize_cdr

import Code.Simulation.MultiRobotClass as MRC
from Code.UtilityCode.turtlebot4 import Turtlebot4
import numpy as np

from Code.UtilityCode.Measurement import Measurement, create_experiment, create_experimental_data
from Code.Analysis import TwoAgentAnalysis as TAA

from Code.Simulation.RobotClass import NewRobot
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

class MyTestCase(unittest.TestCase):
    def test_run_LOS_exp_gen(self, exp_file):
        # From the data sig_v =0.1, sig_w=0.1 and sig_uwb = 0.35 (dependable on the set... ) are the best values.
        sig_v = 0.08
        sig_w = 0.05
        sig_uwb = 0.35

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/experiment_outlier_rejection_3/10hz"


        methods = [
                    # "NLS|frequency=1.0|horizon=100|perfect_guess=0",
                   # "NLS|frequency=1.0|horizon=10",
                     "NLS|frequency=1.0|horizon=10|perfect_guess=0",
                     "NLS|frequency=1.0|horizon=10",
        ]

        experiment_data, measurements = create_experimental_data(exp_file, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = False
        tas.run_experiment(methods=methods, redo_bool=True, experiment_data=experiment_data)


    def test_create_sim_data_from_real_NLS(self,exp_file):
        sig_v = 0.08
        sig_w = 0.05
        sig_uwb = 0.25

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/NLS"
        data_folder = "Measurements_correction/"

        experiment_data, measurements = create_experimental_sim_data(exp_file, sig_v, sig_w, sig_uwb)
        methods = [
                    #"losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "algebraic|frequency=1.0|horizon=10",
                   # "algebraic|frequency=10.0|horizon=1000",
                   # "algebraic|frequency=1.0|horizon=10",
                   "NLS|frequency=1.0|horizon=10",
                   "NLS|frequency=1.0|horizon=10|perfect_guess=0",
                   # "NLS|frequency=1.0|horizon=10|perfect_guess=0",
                   # "QCQP|frequency=10.0|horizon=100",
                   # "QCQP|frequency=1.0|horizon=10"
                   # "QCQP|frequency=10.0|horizon=1000"
                   ]

        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = False
        tas.run_experiment(methods=methods, redo_bool=True, experiment_data=experiment_data, res_type="simulation", prefix="sim_")

        return tas, measurements

    def test_nls_exp_1(self):
        exp_file = "Measurements_correction/exp1_los_sampled.pkl"
        # self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_2(self):
        exp_file = "Measurements_correction/exp2_los_sampled.pkl"
        # self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_3(self):
        exp_file = "Measurements_correction/exp3_los_sampled.pkl"
        # self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_4(self):
        exp_file = "Measurements_correction/exp4_los_sampled.pkl"
        # self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_5(self):
        exp_file = "Measurements_correction/exp5_los_sampled.pkl"
        # self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)


    def test_NLS_time_analysis(self):
        # result_folder = "./Experiments/LOS_exp/Results/new_nls_correct_init_test/"
        result_folders = [
            # "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/1hz",
            "./Experiments/LOS_exp/Results/NLS"
        ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        methods_order = [

            "NLS|frequency=1.0|horizon=10",
            "NLS|frequency=1.0|horizon=10|perfect_guess=0",

        ]

        methods_color = {
            "NLS|frequency=1.0|horizon=10": "tab:purple",
            "NLS|frequency=1.0|horizon=10|perfect_guess=0": "tab:green"
        }

        methods_legend = {
            "NLS|frequency=1.0|horizon=10": "NLS_perfect",
            "NLS|frequency=1.0|horizon=10|perfect_guess=0": "NLS",}
        # taa.delete_data()
        taa.create_panda_dataframe()
        taa.time_analysis(sigma_uwbs=[0.35], sigma_vs=[0.08], frequencies=[1.0, 10.0], start_time=0,
                          methods_order=methods_order, methods_color=methods_color, methods_legend=methods_legend,
                          sigma_bound=False, save_fig=False)
        # taa.boxplot_LOS_comp_time(save_fig=False)
        # taa.calculation_time(save_fig=False)
        plt.show()

    def test_NLS_sim(self):
        result_folder = "../../../Data/Results/Sim_LOS_06_2024/NLS"
        result_folder = "./Experiments/LOS_exp/Results/NLS"
        # result_folder = ("../../../Data/Results/test_files")
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)

        nls_10 = {"Method": "NLS|frequency=1.0|horizon=10",
                   "Variables": {
                       "Type": ["experiment", "simulation"],
                       "Variable": ["error_x_relative", "error_h_relative", "calculation_time"],
                       # "Sigma_dv": [0.01, 0.1],
                       # "Sigma_uwb": [0.1, 1.],
                       # "Sigma_dw": [],
                       # "Frequency": [1.0],
                   },
                   "Color": "tab:green",
                   "Legend": "NLS",
                   }
        nls_10p = {"Method": "NLS|frequency=1.0|horizon=10|perfect_guess=0",
                   "Variables": {
                       "Type": ["experiment", "simulation"],
                       "Variable": ["error_x_relative", "error_h_relative", "calculation_time"],
                       #"Sigma_dv": [0.01, 0.1],
                       # "Sigma_uwb": [0.1, 1.],
                       # "Sigma_dw": [],
                       # "Frequency": [1.0],
                   },
                   "Color": "tab:red",
                   "Legend": "NLS perfect",
                   }
        # nls_100 = {"Method": "NLS|frequency=1.0|horizon=100",
        #            "Variables": {
        #                "Type": ["simulation"],
        #                "Variable": ["error_x_relative", "error_h_relative"],
        #                "Sigma_dv": [0.01, 0.1],
        #                "Sigma_uwb": [0.1, 1.],
        #                # "Sigma_dw": [],
        #                "Frequency": [1.0],
        #            },
        #            "Color": "tab:green",
        #            "Legend": "Ours",
        #            }

        methods_order = [nls_10, nls_10p
                         # nls_100,
                         ]

        df, methods_names, methods_colors,  methods_styles, methods_legends = taa.filter_methods_new(methods_order)
        taa.print_statistics(methods_names, ["error_x_relative", "error_h_relative", "calculation_time"], df)
        g = taa.boxplot_exp(df, methods_color=methods_colors, methods_legend=methods_legends,
                            hue_variable="Name", hue_order=methods_names,
                            col_variable="Variable",
                            row_variable="Sigma_dv",
                            x_variable="Type",
                            )
        g.fig.set_size_inches(8, 4)
        for ax in g.axes_dict:
            if "error_x_relative" in ax:
                g.axes_dict[ax].set_yscale("log")
                if 0.1 in ax:
                    g.axes_dict[ax].set_ylabel(r"$\sigma_v = 0.1 \frac{m}{s}$")
                if 0.01 in ax:
                    g.axes_dict[ax].set_ylabel(r"$\sigma_v = 0.01 \frac{m}{s}$")
            if 0.1 in ax:
                g.axes_dict[ax].set_xlabel(r"$\sigma_{uwb} [m]$")
            if 0.01 in ax:
                if "error_h_relative" in ax:
                    g.axes_dict[ax].set_title(taa.y_label["error_h_relative"])
                if "error_x_relative" in ax:
                    g.axes_dict[ax].set_title(taa.y_label["error_x_relative"])
        # plt.figure(0).set_size_inches(10, 10)
        sns.move_legend(g, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=5)
        plt.subplots_adjust(top=0.8, bottom=0.12, left=0.12, right=0.99)
        plt.suptitle("Monte Carlo simulation")
        plt.show()

if __name__ == '__main__':
    unittest.main()