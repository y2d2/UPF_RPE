import unittest
from Code.Analysis import TwoAgentAnalysis as TAA
import os
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import itertools

import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_create_results_pkl(self):
        results = "./Results_2D/base"
        taa = TAA.TwoAgentAnalysis(result_folders=results)
        taa.laod_directly_to_df()
        taa.save_df("./Results_2D/base.pkl")


    def test_2D_results(self):
        """
        This data changed the UWB simulated data to have non zero mean gaussian noise
        """
        def filter_data(resutsFile, var, classes, start_time, cus_type):
            taa = TAA.TwoAgentAnalysis(result_folders=None)
            taa.load_df(resutsFile)
            df = taa.df
            df_local = df[(df["Variable"].isin(var)) & (df["Class"].isin(classes)) & (df["Time"] > start_time)]
            # Change type for all rows:
            df_local["Type"] = cus_type
            # df_local["Class_type"] = df_local["Class"].apply(lambda x: f"{x} {type}")
            return df_local

        custom_class_order = ["losupf", "nodriftupf", "QCQP 20s", "QCQP 10s", "losupf_per", "nodriftupf_per" , "NLS"]
        start_time = 250
        var = ["error_x_relative"]
        folder = "./Results_2D/"
        df_2D_base = filter_data(f"{folder}/base.pkl", var, custom_class_order, start_time, "2D")
        # df_exp = filter_data(f"{folder}/results_exp.pkl", var, custom_class_order, start_time, "Exp")
        # df_or_sim = filter_data(f"{folder}/results_original_sim.pkl", var, custom_class_order, start_time, "Or. sim")
        # df_uwb_sim_real_odom = filter_data(f"{folder}/results_uwb_sim_real_odom.pkl", var, custom_class_order, start_time, "sim UWB , real odom")
        # df_uwb_real_sim_odom = filter_data(f"{folder}/results_real_uwb_sim_odom.pkl", var, custom_class_order, start_time, "real UWB , sim odom")
        # df_uwb_outlier_rejection = filter_data(f"{folder}/results_exp_uwb_outlier_rejection.pkl", var, custom_class_order, start_time, "UWB outlier rejection")
        # df_uwb_exp_uwb_model = filter_data(f"{folder}/results_exp_uwb_model.pkl", var, custom_class_order, start_time, "UWB exp ad UWB")
        # df_uwb_exp_uwb_model_out = filter_data(f"{folder}/results_exp_uwb_model_outlier.pkl", var, custom_class_order, start_time, "UWB exp ad UWB out")



        df = pd.concat([df_2D_base ])

        custom_class_order =["losupf", "nodriftupf", "QCQP 20s",  "QCQP 10s", "losupf_per", "NLS"]
        custom_class_order =["losupf", "nodriftupf", "losupf_per","nodriftupf_per"]
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

        custom_class_order = ["losupf", "nodriftupf", "QCQP 20s", "QCQP 10s", "losupf_per", "nodriftupf_per" , "NLS"]
        start_time = 0
        var = ["error_x_relative"]
        folder = "./Results_2D/"
        df_2D_base = filter_data(f"{folder}/base.pkl", var, custom_class_order, start_time, "2D")

        runs = df_2D_base["Run"].unique()
        # runs = ["exp4_los_sampled"]
        for run in runs:
            df_run = df_2D_base[df_2D_base["Run"].isin([run])]
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




if __name__ == '__main__':
    unittest.main()

