import os

import rosbags.rosbag2 as rb2
import unittest

from matplotlib.lines import Line2D
from rosbags.serde import deserialize_cdr

import numpy as np
import pickle as pkl
from UtilityCode.Measurement import Measurement, create_experiment, create_experimental_data, create_experimental_sim_data
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Analysis import TwoAgentAnalysis as TAA
import pandas as pd
import seaborn as sns

from pathlib import Path
from rosbags.typesys import get_types_from_msg, register_types

class MyTestCase(unittest.TestCase):
    def test_create_sim_data_from_real(self):
        sig_v = 0.15
        sig_w = 0.05
        sig_uwb = 0.25

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/sim2real/"
        data_folder = main_folder + "Measurements/"

        experiment_data, measurements = create_experimental_sim_data(data_folder, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = False
        tas.uwb_rate = 1.
        # tas.run_experiment(methods=["losupf", "nodriftupf", "algebraic", "NLS"], redo_bool=False, experiment_data=experiment_data)
        tas.run_experiment(methods=[ "nodriftupf"], redo_bool=True, experiment_data=experiment_data)
        # plt.show()
        return tas, measurements

    def test_fuse_sim_real_pkl(self):
        folder = "./Experiments/LOS_exp/Results/sim2real/"
        sim_data_file = "s_number_of_agents_2_sigma_dv_0c15_sigma_dw_0c05_sigma_uwb_0c25_alpha_1c0_kappa_neg1c0_beta_2c0.pkl"
        real_data_file = "r_number_of_agents_2_sigma_dv_0c15_sigma_dw_0c05_sigma_uwb_0c25_alpha_1_kappa_neg1c0_beta_2c0.pkl"
        merged_data = {}
        with open(folder + sim_data_file, "rb") as f:
            sim_data = pkl.load(f)
        with open(folder + real_data_file, "rb") as f:
            real_data = pkl.load(f)

        for exp in real_data:
            merged_data[exp] = real_data[exp]
            for method in sim_data[exp]:
                if exp != "parameters":
                    merged_data[exp]["sim_"+method] = sim_data[exp][method]

        with open(folder + "merged/merged_data.pkl", "wb") as f:
            pkl.dump(merged_data, f)

    def test_sim2real_analysis(self):
        result_folder = "./Experiments/LOS_exp/Results/sim2real/merged"
        taa = TAA.TwoAgentAnalysis(result_folder=result_folder)
        taa.delete_data()
        taa.create_panda_dataframe()

        # Filter the DataFrame for the desired variables and methods
        df = taa.df.loc[
            (~taa.df["Method"].isin(["slam", "WLS", "upf", "sim_slam", "sim_WLS"])) &
            (taa.df["Variable"].isin(["error_x_relative", "error_h_relative"]))
            ]

        order = ["losupf", "sim_losupf", "nodriftupf", "sim_nodriftupf", "algebraic", "sim_algebraic", "NLS", "sim_NLS"]
        method_colors = {"NLS": "tab:blue", "sim_NLS":"lightblue",
                         "losupf": "tab:green", "sim_losupf": "lightgreen",
                         "nodriftupf": "tab:red",  "sim_nodriftupf": "lightcoral",
                         "algebraic": "tab:orange", "sim_algebraic": "bisque"}
        names = {"NLS": "NLS", "sim_NLS":"NLS sim", "losupf": "Ours, proposed",
                 "sim_losupf": "Ours, proposed sim",
                 "nodriftupf": "Ours, without pseudo-state",  "sim_nodriftupf": "Ours, without pseudo-state sim",
                 "algebraic": "Algebraic", "sim_algebraic": "Algebraic sim"}

        # Create custom subplots with different y-axis scales
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(r"Experiments: $\sigma_v= 0.15 \frac{m}{s}$, $\sigma_w=0.05 \frac{(rad)}{s}$ and $\sigma_d = 0.25 m$", fontsize=12)
        for i, variable in enumerate(["error_x_relative", "error_h_relative"]):
            df_c = df[df["Variable"] == variable]
            methods = df_c["Method"].unique()
            for method in methods:
                print(method, variable, df_c[df_c["Method"] == method]["value"].mean(), " pm ", df_c[df_c["Method"] == method]["value"].std(), "; median: " , df_c[df_c["Method"] == method]["value"].median())

            g = sns.boxplot(data=df[df["Variable"] == variable], x='Method', y='value', hue='Method', dodge=False,
                        ax=axes[i], order = order, palette=method_colors, hue_order=order,) #, hue_order=order)


            axes[i].set_title("")
            axes[i].set_ylabel("")
            # axes[i].set_xlabel("Method")
            axes[i].set_xlabel(taa.y_label[variable], fontsize=12)
            axes[i].tick_params(bottom=False)
            axes[i].set_xticklabels([])
            if variable == "error_x_relative":
                axes[i].set_yscale("log")
            else:
                axes[i].set_yscale("linear")
            # legend_data = g._legend_data
        # Remove the legend from the first subplot
        axes[0].get_legend().remove()
        # axes[1].get_legend().remove()

        legend_labels = ['Ours, proposed', 'Ours, proposed sim',
                         "Ours, without pseudo-state", "Ours, without pseudo-state sim",
                         'Algebraic', 'Algebraic sim',
                         "NLS", "NLS sim"]
        # # Customize the legend for the second subplot and move it outside to the right
        legend_handles = axes[1].get_legend().legendHandles
        # print(legend_data)
        # new_legend_data = {}
        # for line, text in zip(legend_data, df["Method"].unique()):
        #     new_legend_data[names[text]] = line
        # legend_data = new_legend_data
        # new_legend_data = {}
        # for name in order:
        #     new_legend_data[names[name]] = legend_data[names[name]]
        fig.legend( handles=legend_handles, labels=legend_labels, ncol=4, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 0.92))

        # fig.legend( handles=new_legend_data.values(), labels=new_legend_data.keys(), ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 0.92))


        # Position the legend to the right outside of the second subplot
        # axes[1].legend(new_legend_data.values(), new_legend_data.keys(), loc='upper left', bbox_to_anchor=(1.05, 0.75))

        # Adjust the spacing between subplots to create more room for the legend
        plt.subplots_adjust(wspace=0.2, right=0.7)  # Adjust the 'wspace' value as needed
        # You can save the figure to a file if needed


        # fig.suptitle("Average error over all simulated conditions")
        axes[1].get_legend().remove()
        plt.subplots_adjust(top=0.70, bottom=0.12, left=0.07, right=0.93)


        plt.show()

if __name__=="__main__":
    t = MyTestCase()
    sim_data = t.test_fuse_sim_real_pkl()


