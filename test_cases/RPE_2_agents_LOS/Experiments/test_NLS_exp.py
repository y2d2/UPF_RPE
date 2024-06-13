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
import pickle as pkl

class MyTestCase(unittest.TestCase):
    def test_run_LOS_exp_gen(self, exp_file):
        # From the data sig_v =0.1, sig_w=0.1 and sig_uwb = 0.35 (dependable on the set... ) are the best values.
        sig_v = 0.08
        sig_w = 0.12
        sig_uwb = 0.35

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/experiment_outlier_rejection_3/10hz"


        methods = [
                    "NLS|frequency=1.0|horizon=100|perfect_guess=0",
                   # "NLS|frequency=1.0|horizon=10",
                   #   "NLS|frequency=1.0|horizon=10|perfect_guess=0",
        ]

        experiment_data, measurements = create_experimental_data(exp_file, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = False
        tas.run_experiment(methods=methods, redo_bool=True, experiment_data=experiment_data)


    def test_create_sim_data_from_real_NLS(self,exp_file):
        sig_v = 0.08
        sig_w = 0.12
        sig_uwb = 0.25

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/sim2real_2/10hz"
        data_folder = "Measurements_correction/"

        experiment_data, measurements = create_experimental_sim_data(exp_file, sig_v, sig_w, sig_uwb)
        methods = [
                    #"losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "algebraic|frequency=1.0|horizon=10",
                   # "algebraic|frequency=10.0|horizon=1000",
                   # "algebraic|frequency=1.0|horizon=10",
                   "NLS|frequency=1.0|horizon=100|perfect_guess=0",
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
        self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_2(self):
        exp_file = "Measurements_correction/exp2_los_sampled.pkl"
        self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_3(self):
        exp_file = "Measurements_correction/exp3_los_sampled.pkl"
        self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_4(self):
        exp_file = "Measurements_correction/exp4_los_sampled.pkl"
        self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)

    def test_nls_exp_5(self):
        exp_file = "Measurements_correction/exp5_los_sampled.pkl"
        self.test_run_LOS_exp_gen(exp_file)
        self.test_create_sim_data_from_real_NLS(exp_file)


    def test_NLS_time_analysis(self):
        # result_folder = "./Experiments/LOS_exp/Results/new_nls_correct_init_test/"
        result_folders = [
            # "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/1hz",
            "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/10hz"
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


if __name__ == '__main__':
    unittest.main()