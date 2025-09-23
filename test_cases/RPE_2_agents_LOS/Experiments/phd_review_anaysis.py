import os
import pandas as pd

import rosbags.rosbag2 as rb2
import unittest
from rosbags.serde import deserialize_cdr
from sympy.physics.units import frequency

import Code.Simulation.MultiRobotClass as MRC
from Code.UtilityCode.turtlebot4 import Turtlebot4
import numpy as np

from Code.UtilityCode.Measurement import Measurement, create_experiment, create_experimental_data
from Code.Analysis import TwoAgentAnalysis as TAA

from Code.Simulation.RobotClass import NewRobot
import matplotlib
matplotlib.use('Qt5Agg')
import itertools

import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns

class MyTestCase(unittest.TestCase):

    def find_methods(self, path):
        """
        This function finds all the methods in the given path
        :param path: The path to the folder containing the results
        :return: A list of methods found in the folder
        """
        methods = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl"):
                    methods.append(file[:-4])


    def load_chaning_hz_data(self):
        """
                This data changed the frequency
        """

        sigma_vs = [0.1, 0.01]
        sigma_uwbs = [0.01, 0.1, 0.25]
        freqs = [1.0, 2.0, 5.0, 10.0]
        freq_color_upf = ["darkgreen", "green", "limegreen", "greenyellow"]
        freq_color_nodriftupf = ["darkred", "brown", "red", "salmon"]
        variable = ["error_x_relative"]
        methods = {}

        param_combinations = list(itertools.product(sigma_vs, sigma_uwbs, freqs))
        for sig_v, sig_uwb, freq in param_combinations:
            sigma_dw = [sig_v]
            sigma_dv = [sig_v]
            sigma_uwb = [sig_uwb]
            i = freqs.index(freq)

            methods[f"upf||frequency={freq}|sigma_v={sig_v}|sigma_uwb={sig_uwb}"] = \
                {"Method":f"losupf|frequency={freq}|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                           "Variables": {
                               "Type": ["simulation"],
                               "Variable": variable,
                               "Sigma_dv": sigma_dv,
                               "Sigma_dw": sigma_dw,
                               "Sigma_uwb": sigma_uwb,
                               "Frequency": [freq],
                           },
                           "Color": freq_color_upf[i],
                           "Legend": f"UPF with pseudo-state, freq={freq}, sig_v={sig_v}, sig_uwb={sig_uwb}",
                           }
            methods[f"nodriftupf||frequency={freq}|sigma_v={sig_v}|sigma_uwb={sig_uwb}"] = \
                {"Method": f"nodriftupf|frequency={freq}|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                              "Variables": {
                                  "Type": ["simulation"],
                                  "Variable": variable,
                                  "Sigma_dv": sigma_dv,
                                  "Sigma_dw": sigma_dw,
                                  "Sigma_uwb": sigma_uwb,
                                  "Frequency": [freq],
                              },
                              "Color": freq_color_nodriftupf[i],
                              "Legend":f"UPF without pseudo-state, freq={freq}, sig_v={sig_v}, sig_uwb={sig_uwb}",
                              }
        return methods

    def test_slam_error(self):
        results =  "../../../Results/SLAM_SIM"
        taa = TAA.TwoAgentAnalysis(result_folders=results)
        taa.laod_directly_to_df( )
        taa.save_df("./Results_review/SLAM_test_sim.pkl")
        print("test")

    def test_load_slam_error(self):
        taa = TAA.TwoAgentAnalysis(result_folders=None)
        taa.load_df("./Results_review/SLAM_test_sim.pkl")
        df = taa.df[(taa.df["Variable"] == "error_h_relative") & (taa.df["Time"] > 0) & taa.df["Class"].isin(
            ["Unkown"])]
        label_map = {
            "losupf_per": "UPF",
            "Unkown": "SLAM",
        }

        # Apply mapping
        df["Class"] = df["Class"].map(label_map)
        median_errors = df.groupby(["Class", "Frequency"])["value"].median().reset_index()
        average_errors = df.groupby(["Class", "Frequency"])["value"].mean().reset_index()
        std_errors = df.groupby(["Class", "Frequency"])["value"].std().reset_index()
        print(median_errors, average_errors, std_errors)


    def test_create_results_pkl(self):
        # result_folders = [
        #     "Results/experiments",
        #     "Results/sim2real",
        # ]
        # results =  "Results_review/Chanign_hz_exp/"
        # results = "./Results_review/Sim_odom_real_uwb/freq=10c0_sig_v=0c08_sig_w=0c08_sig_uwb=0c25"
        # results = "./Results_review/change_uwb_sig/freq=10c0_sig_v=0c08_sig_w=0c08_sig_uwb=0c25"
        # results = "./Results_review/UWB_Outlier_rejection"
        # results = "../../../Results/sim2real"
        results = "../../../Results/experiments/exp_losupf|frequency=10c0|resample_factor=0c1|sigma_uwb_factor=1c0|exp1_los_sampled|s_uwb=0c25|s_dv=0c08|s_dw=0c08.pkl"
        # results = "./Results_review/Sim_uwb_real_odom/freq=10c0_sig_v=0c08_sig_w=0c08_sig_uwb=0c25"
        # results = "./Results_review/UWB_model"
        # results = "./Results_review/UWB_model_outlier"
        # path = "./Results_review/Changing_hz_sim"
        # results = [f"./Results_review/Changing_hz_sim/{dir}" for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
        taa = TAA.TwoAgentAnalysis(result_folders=results)
        taa.laod_directly_to_df()
        print("test")

        # taa.df = pd.concat(taa.dfs)
        # print(taa.df.head())
        # taa.save_df("./Results_review/Sim_to_real_gap_analysis/results_exp_uwb_model_outlier.pkl")
        # return methods

    def test_changing_hz_analysis(self):
        # run_id = "exp1_los_sampled"
        methods = self.load_chaning_hz_data()
        methods_order = [methods[key] for key in methods.keys()]
        taa = TAA.TwoAgentAnalysis(result_folders=None)
        taa.load_df("./Results_review/results_changing_freq_exp.pkl")
        df = taa.df[(taa.df["Variable"] == "error_x_relative") & (taa.df["Time"] > 200) & taa.df["Class"].isin(["losupf_per", "nodriftupf_per"])]
        label_map = {
            "losupf_per": "Proposed solution with pseudo-state on rotational drift",
            "nodriftupf_per": "Proposed solution without pseudo-state on rotational drift",
        }

        # Apply mapping
        df["Class"] = df["Class"].map(label_map)

        average_errors = df.groupby(["Class", "Frequency"])["value"].mean().reset_index()
        print(average_errors)
        # sns.boxplot(data=df, x="Frequency", y="value", hue="Class")
        sns.barplot(data=average_errors, x="Frequency", y="value", hue="Class")
        # sns.stripplot(data=df, x="Frequency", y="value", hue="Class", dodge=True,
        #               jitter=True, alpha=0.5, marker="o")
        plt.legend(title="Class",  loc='lower left')
        plt.title("Average Error per Method and Frequency")
        plt.ylabel("Average Error [m]")
        plt.show()


    def test_chanign_hz_sim_analysis(self):

        taa = TAA.TwoAgentAnalysis(result_folders=None)
        taa.load_df("./Results_review/results_changing_freq_sim.pkl")
        df = taa.df[(taa.df["Variable"] == "error_x_relative") & (taa.df["Time"] > 200) & taa.df["Class"].isin(
            ["losupf_per", "nodriftupf_per"])]
        label_map = {
            "losupf_per": "Proposed solution with pseudo-state on rotational drift",
            "nodriftupf_per": "Proposed solution without pseudo-state on rotational drift",
        }

        # Apply mapping
        df["Class"] = df["Class"].map(label_map)
        for run in df["Run"].unique():
            df_run = df[df["Run"] == run]
            g = sns.catplot(
                data=df_run,
                x="Frequency",
                y="value",
                hue="Class",
                col="Sigma_uwb",# Facet by run
                row="Sigma_dv",  # Facet by sigma_dv
                kind="bar",  # Or "box", "strip", etc.
                # col_wrap=4,  # Wrap after 4 columns (optional)
                height=4,
                aspect=1
            )
            g.set_titles(
                row_template="Sigma odometry: {row_name}",
                col_template="Sigma UWB: {col_name}"
            )
            g.set_axis_labels("Frequency", "Error [m]")
            g._legend.set_bbox_to_anchor((0.5, 0.97))
            g._legend.set_loc("upper center")
            g._legend.set_title("Class")
            plt.tight_layout()
        plt.show()

    def test_changing_uwb_sig(self):
        """
        This data changed the UWB simulated data to have non zero mean gaussian noise
        """
        def filter_data(resutsFile, var, classes, start_time, type):
            taa = TAA.TwoAgentAnalysis(result_folders=None)
            taa.load_df(resutsFile)
            df = taa.df
            df_local = df[(df["Variable"].isin(var)) & (df["Class"].isin(classes)) & (df["Time"] > start_time)]
            # Change type for all rows:
            df_local["Type"] = type
            df_local["Class_type"] = df_local["Class"].apply(lambda x: f"{x} {type}")
            return df_local

        custom_class_order = ["losupf", "nodriftupf", "QCQP 20s", "QCQP 10s", "losupf_per", "NLS"]
        start_time = 250
        var = ["error_x_relative"]
        folder = "./Results_review/Sim_to_real_gap_analysis"
        df_advanced_uwb_model = filter_data(f"{folder}/results_uwb_advanced_sim_model.pkl", var, custom_class_order, start_time, "Ad. UWB")
        df_exp = filter_data(f"{folder}/results_exp.pkl", var, custom_class_order, start_time, "Exp")
        df_or_sim = filter_data(f"{folder}/results_original_sim.pkl", var, custom_class_order, start_time, "Or. sim")
        df_uwb_sim_real_odom = filter_data(f"{folder}/results_uwb_sim_real_odom.pkl", var, custom_class_order, start_time, "sim UWB , real odom")
        df_uwb_real_sim_odom = filter_data(f"{folder}/results_real_uwb_sim_odom.pkl", var, custom_class_order, start_time, "real UWB , sim odom")
        df_uwb_outlier_rejection = filter_data(f"{folder}/results_exp_uwb_outlier_rejection.pkl", var, custom_class_order, start_time, "UWB outlier rejection")
        df_uwb_exp_uwb_model = filter_data(f"{folder}/results_exp_uwb_model.pkl", var, custom_class_order, start_time, "UWB exp ad UWB")
        df_uwb_exp_uwb_model_out = filter_data(f"{folder}/results_exp_uwb_model_outlier.pkl", var, custom_class_order, start_time, "UWB exp ad UWB out")



        df = pd.concat([df_exp, df_uwb_exp_uwb_model, df_uwb_outlier_rejection, df_uwb_exp_uwb_model_out, df_or_sim, df_advanced_uwb_model, df_uwb_sim_real_odom, df_uwb_real_sim_odom  ])

        custom_class_order =["losupf", "nodriftupf", "QCQP 20s",  "QCQP 10s", "losupf_per", "NLS"]
        g = sns.catplot(data=df, kind='box', col="Variable", y='value', x="Type", hue='Class',
                        dodge=True, aspect=1.33,  height=8,
                        # order = custom_class_order,
                        hue_order=custom_class_order,
                        # dodge=True, aspect=1.33, palette=methods_color, hue_order=hue_order, height=8,
                        legend=True, sharey=False)
        g = sns.catplot(data=df, kind='box', col="Variable", y='value', x="Class", hue='Type',
                        dodge=True, aspect=1.33, height=8,
                        order = custom_class_order,
                        # hue_order=custom_class_order,
                        # dodge=True, aspect=1.33, palette=methods_color, hue_order=hue_order, height=8,
                        legend=True, sharey=False)
        plt.show()


    def test_plot_plot_individual_results(self):
        def filter_data(resutsFile, var, classes, start_time, type):
            taa = TAA.TwoAgentAnalysis(result_folders=None)
            taa.load_df(resutsFile)
            df = taa.df
            df_local = df[(df["Variable"].isin(var)) & (df["Class"].isin(classes)) & (df["Time"] > start_time)]
            # Change type for all rows:
            df_local["Type"] = type
            df_local["Class_type"] = df_local["Class"].apply(lambda x: f"{x} {type}")
            return df_local

        custom_class_order = ["losupf", "nodriftupf", "QCQP 20s", "QCQP 10s", "losupf_per", "NLS"]
        custom_class_order = ["losupf", "nodriftupf", "QCQP 10s", "losupf_per", "NLS"]
        start_time = 0
        var = ["error_x_relative"]


        folder = "./Results_review/Sim_to_real_gap_analysis"
        # df_exp = filter_data(f"{folder}/results_exp.pkl", var, custom_class_order, start_time, "Exp")
        df_exp = filter_data(f"{folder}/results_exp_uwb_outlier_rejection.pkl", var, custom_class_order, start_time, "UWB outlier rejection")

        runs = df_exp["Run"].unique()
        # runs = ["exp4_los_sampled"]
        for run in runs:
            df_run = df_exp[df_exp["Run"].isin([run])]
            #lineplot:
            plt.figure()
            sns.lineplot(data=df_run, x="Time", y="value", hue="Class",  markers=False, dashes=False, hue_order=custom_class_order)
            # g = sns.catplot(data=df_run, kind='box', col="Variable", y='value', x="Class", hue='Type',
            #                 dodge=True, aspect=1.33, height=8,
            #                 order=custom_class_order,
            #                 # hue_order=custom_class_order,
            #                 # dodge=True, aspect=1.33, palette=methods_color, hue_order=hue_order, height=8,
            #                 legend=True, sharey=False)
            plt.suptitle(run)
        plt.show()


