import os

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
import seaborn as sns

class MyTestCase(unittest.TestCase):


    def test_single_exp(self):
        sig_v = 0.08
        sig_w = 0.08
        sig_uwb = 0.25


        result_folder = "Real_Exp_test"
        data_file = "Experiments/Exp3_SemiNLOS/Measurements/exp3_sec1_los_sampled.pkl"
        experiment_data, _ = create_experimental_data(data_file, sig_v, sig_w, sig_uwb)
        tas = create_experiment(result_folder, sig_v, sig_w, sig_uwb)
        # tas.run_experiment(methods=[ "upf"], redo_bool=True, experiment_data=experiment_data)

        tas.run_experiment(methods=[ "NLS", "algebraic", "upf", "losupf", "nodriftupf", "upfnaive"], redo_bool=False, experiment_data=experiment_data)

        return tas
        # taa = TAA.TwoAgentAnalysis(result_folder=result_folder)
        # taa.delete_data()
        # taa.create_panda_dataframe()
        # taa.single_settings_boxplot(save_fig=False)
        # plt.show()

    def test_run_LOS_exp(self):
        # From the data sig_v =0.1, sig_w=0.1 and sig_uwb = 0.35 (dependable on the set... ) are the best values.
        sig_v = 0.08
        sig_w = 0.12
        sig_uwb = 0.15

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/exp_cor_new6/exp"
        data_folder = "corrections3/"

        experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)

        methods = ["losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "algebraic|frequency=1.0|horizon=10",
                   "algebraic|frequency=10.0|horizon=100",
                   # # "algebraic|frequency=10.0|horizon=1000",
                   "QCQP|frequency=10.0|horizon=100",
                   # # "QCQP|frequency=10.0|horizon=1000",
                   "NLS|frequency=1.0|horizon=10",
                   ]

        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = False
        tas.run_experiment(methods=methods, redo_bool=False, experiment_data=experiment_data)
        plt.show()
        # return tas, measurements

    def test_rename_experiments(self):
        main_folder = "./Experiments/LOS_exp/"
        load_dir = main_folder + "Results/experiment_outlier_rejection_3"
        save_dir =  main_folder + "Results/experiment_outlier_rejection_4"

        n_files = len(os.listdir(load_dir))
        n_file = 0
        for file in os.listdir(load_dir):
            n_file += 1
            print(str(int(n_file/n_files*100)) + "%: " +  file)
            if os.path.isfile(load_dir + "/"+file) and not os.path.exists(save_dir + "/exp_" + file):
                os.rename(load_dir + "/" + file, save_dir + "/exp_"+ file)
                with open(save_dir + "/exp_" + file, "rb") as f:
                    data = pkl.load(f)
                f.close()
                with open(save_dir + "/exp_" + file, "wb") as f:
                    data["parameters"]["type"] = "experiment"
                    pkl.dump(data, f)
                f.close()

    def test_rename_methods(self):
        main_folder = "./Experiments/LOS_exp/"
        load_dir = main_folder + "Results/experiment_outlier_rejection_3"
        save_dir =  main_folder + "Results/experiment_outlier_rejection_4"

        n_files = len(os.listdir(load_dir))
        n_file = 0
        for file in os.listdir(load_dir):
            n_file += 1
            print(str(int(n_file/n_files*100)) + "%: " +  file)
            if os.path.isfile(load_dir + "/"+file) and not os.path.exists(save_dir + "/exp_" + file):
                os.rename(load_dir + "/" + file, save_dir + "/exp_"+ file)
                with open(save_dir + "/exp_" + file, "rb") as f:
                    data = pkl.load(f)
                f.close()
                with open(save_dir + "/exp_" + file, "wb") as f:
                    data["parameters"]["type"] = "experiment"
                    pkl.dump(data, f)
                f.close()

    def test_plot_LOS_error_time(self):
        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/new_experiment/"
        tas = MRC.TwoAgentSystem(trajectory_folder="./", result_folder=results_folder)
        data = tas.get_data_from_file(results_folder + "number_of_agents_2_sigma_dv_0c15_sigma_dw_0c05_sigma_uwb_0c25_alpha_1c0_kappa_neg1c0_beta_2c0.pkl")
        exp = 2
        for i in range(2):
            agent_name = "drone_"+str(i)

            exp_name = "exp"+str(exp)+"_los_sampled"
            upf_x_error_0 = data[exp_name]["nodriftupf"][agent_name]["error_x_relative"]
            NLS_x_error_0 = data[exp_name]["NLS"][agent_name]["error_x_relative"]
            losupf_x_error_0 = data[exp_name]["losupf"][agent_name]["error_x_relative"]
            slam_x_error_0 = data[exp_name]["slam"][agent_name]["error_x_relative"]

            upf_h_error_0 = data[exp_name]["nodriftupf"][agent_name]["error_h_relative"]
            NLS_h_error_0 = data[exp_name]["NLS"][agent_name]["error_h_relative"]
            losupf_h_error_0 = data[exp_name]["losupf"][agent_name]["error_h_relative"]
            slam_h_error_0 = data[exp_name]["slam"][agent_name]["error_h_relative"]

            _, ax = plt.subplots(2, 1)
            ax[0].plot(NLS_x_error_0, label="NLS [7]", color="tab:blue", linewidth=2)
            ax[0].plot(losupf_x_error_0, label="upf ours", color="tab:green", linewidth=2)
            ax[0].plot(upf_x_error_0, label="No drift UPF", color="tab:red", linewidth=2)
            ax[0].plot(slam_x_error_0, label="slam", color="tab:orange", linewidth=2)
            ax[0].set_xlabel("Time [s]", fontsize=12)
            ax[0].set_ylabel(r"$\epsilon_{\hat{p}^t}$ [m]", fontsize=12)
            ax[0].grid()
            ax[0].legend()

            ax[1].plot(NLS_h_error_0, label="NLS [7]", color="tab:blue", linewidth=2)
            ax[1].plot(losupf_h_error_0, label=r"upf ours", color="tab:green", linewidth=2)
            ax[1].plot(upf_h_error_0, label="No drift UPF", color="tab:red", linewidth=2)
            ax[1].plot(slam_h_error_0, label="slam", color="tab:orange", linewidth=2)
            ax[1].set_xlabel("Time [s]", fontsize=12)
            ax[1].set_ylabel(r"$\epsilon_{\hat{\theta}^t}$ [(rad))]", fontsize=12)
            ax[1].grid()
            ax[1].set_ylim([0., 0.5])

            ax[1].legend()



        plt.show()


    def test_exp_analysis(self):
        result_folders = [
                "./Experiments/LOS_exp/Results/exp_cor_new1/exp",
                "./Experiments/LOS_exp/Results/exp_cor_new1/sim",
                # "./Experiments/LOS_exp/Results/experiments_paper/Sim",
            ]
        sigma_dv = [0.08]
        sigma_uwb = [0.25]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        upf_sim = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "Variables": {
                            "Type": ["simulation"],
                             "Variable":["error_x_relative", "error_h_relative"],
                             "Sigma_dv": [0.08],
                             "Sigma_uwb": sigma_uwb,
                             # "Sigma_dw": [],
                             "Frequency": [10.0],
                         },
                           "Color": "lightgreen",
                           "Legend": "Ours (sim)",
                           }
        nodriftupf_sim = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "Variables": {
                             "Type": ["simulation"],
                             "Variable": ["error_x_relative", "error_h_relative"],
                             "Sigma_dv": [0.08],
                             "Sigma_uwb": sigma_uwb,
                             # "Sigma_dw": [],
                             "Frequency": [10.0],
                         },
                         "Color": "salmon",
                         "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state (sim)",
                         }
        alg_sim = {"Method": "algebraic|frequency=10.0|horizon=100",
                         "Variables": {
                             "Type": ["simulation"],
                             "Variable": ["error_x_relative", "error_h_relative"],
                             "Sigma_dv": [0.08],
                             "Sigma_uwb": sigma_uwb,
                             # "Sigma_dw": [],
                             "Frequency": [10.0],
                         },
                         "Color": "bisque",
                         "Legend": "Algebraic (sim)",
                         }
        qcqp_sim = {"Method":"QCQP|frequency=10.0|horizon=100",
                         "Variables": {
                             "Type": ["simulation"],
                             "Variable": ["error_x_relative", "error_h_relative"],
                             "Sigma_dv": [0.08],
                             "Sigma_uwb": sigma_uwb,
                             # "Sigma_dw": [],
                             "Frequency": [10.0],
                         },
                         "Color": "cornflowerblue",
                         "Legend": "QCQP (sim)",
                         }
        nls_sim = {"Method": "NLS|frequency=1.0|horizon=10",
                         "Variables": {
                             "Type": ["simulation"],
                             "Variable": ["error_x_relative", "error_h_relative"],
                             "Sigma_dv": [0.08],
                             "Sigma_uwb": sigma_uwb,
                             "Frequency": [1.0],
                         },
                         "Color": "thistle",
                         "Legend": "NLS (sim)",
                         }

        upf_exp = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": sigma_dv,
                       "Sigma_uwb": sigma_uwb,
                       # "Sigma_dw": [],
                       "Frequency": [10.0],
                   },
                   "Color": "tab:green",
                   "Legend": "Ours",
                   }
        nodriftupf_exp = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                          "Variables": {
                              "Type": ["experiment"],
                              "Variable": ["error_x_relative", "error_h_relative"],
                              "Sigma_dv": sigma_dv,
                              "Sigma_uwb": sigma_uwb,
                              # "Sigma_dw": [],
                              "Frequency": [10.0],
                          },
                          "Color": "tab:red",
                          "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state",
                          }
        alg_exp = {"Method": "algebraic|frequency=10.0|horizon=100",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": sigma_dv,
                       "Sigma_uwb": sigma_uwb,
                       # "Sigma_dw": [],
                       "Frequency": [10.0],
                   },
                   "Color": "tab:orange",
                   "Legend": "Algebraic",
                   }
        qcqp_exp = {"Method": "QCQP|frequency=10.0|horizon=100",
                    "Variables": {
                        "Type": ["experiment"],
                        "Variable": ["error_x_relative", "error_h_relative"],
                        "Sigma_dv": sigma_dv,
                        "Sigma_uwb": sigma_uwb,
                        # "Sigma_dw": [],
                        "Frequency": [10.0],
                    },
                    "Color": "tab:blue",
                    "Legend": "QCQP",
                    }
        nls_exp = {"Method": "NLS|frequency=1.0|horizon=10",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": sigma_dv,
                       "Sigma_uwb": sigma_uwb,
                       # "Sigma_dw": [0.05],
                       "Frequency": [1.0],
                   },
                   "Color": "tab:purple",
                   "Legend": "NLS",
                   }

        methods_order = [
                        upf_exp, upf_sim,
                          nodriftupf_exp, nodriftupf_sim,
                         alg_exp, alg_sim,
                          qcqp_exp, qcqp_sim,
                          nls_exp, nls_sim,
                        ]

        df, methods_names, methods_colors, methods_legends = taa.filter_methods_new(methods_order)
        taa.print_statistics(methods_names, ["error_x_relative", "error_h_relative"], df)
        g = taa.boxplot_exp(df, methods_color=methods_colors, methods_legend=methods_legends,
                        hue_variable="Name", hue_order=methods_names,
                        col_variable="Variable",
                        row_variable=None,
                        x_variable="Sigma_dv", 
                        )

        g.axes_dict["error_x_relative"].set_yscale("log")
        g.axes_dict["error_h_relative"].set_ylabel(taa.y_label["error_h_relative"])
        g.axes_dict["error_x_relative"].set_ylabel(taa.y_label["error_x_relative"])
        sns.move_legend(g, loc="upper center", bbox_to_anchor= (0.5, 0.98), ncol=5)
        plt.subplots_adjust(top=0.8, bottom=0.12, left=0.1, right=0.99)
        # plt.suptitle("Experiments")
        plt.show()



    def test_exp_individual_try_analysis(self):
        #Todo: Plot resutls per drone per simulation run.
        result_folders = [
                "./Experiments/LOS_exp/Results/experiments_paper/Experiments",
                # "./Experiments/LOS_exp/Results/experiments_paper/Sim",
                          ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)


        upf_exp = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.08],
                       "Sigma_uwb": [0.25],
                       # "Sigma_dw": [],
                       "Frequency": [10.0],
                   },
                   "Color": "tab:green",
                   "Legend": "Ours, proposed",
                   }
        nodriftupf_exp = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                          "Variables": {
                              "Type": ["experiment"],
                              "Variable": ["error_x_relative", "error_h_relative"],
                              "Sigma_dv": [0.08],
                              "Sigma_uwb": [0.25],
                              # "Sigma_dw": [],
                              "Frequency": [10.0],
                          },
                          "Color": "tab:red",
                          "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state",
                          }
        alg_exp = {"Method": "algebraic|frequency=10.0|horizon=1000",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.08],
                       "Sigma_uwb": [0.25],
                       # "Sigma_dw": [],
                       "Frequency": [10.0],
                   },
                   "Color": "tab:orange",
                   "Legend": "Algebraic",
                   }
        qcqp_exp = {"Method": "QCQP|frequency=10.0|horizon=1000",
                    "Variables": {
                        "Type": ["experiment"],
                        "Variable": ["error_x_relative", "error_h_relative"],
                        "Sigma_dv": [0.08],
                        "Sigma_uwb": [0.25],
                        # "Sigma_dw": [],
                        "Frequency": [10.0],
                    },
                    "Color": "tab:blue",
                    "Legend": "QCQP",
                    }
        nls_exp = {"Method": "NLS|frequency=1.0|horizon=10|perfect_guess=0",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.08],
                       "Sigma_uwb": [0.35],
                       # "Sigma_dw": [],
                       "Frequency": [1.0],
                   },
                   "Color": "tab:purple",
                   "Legend": "NLS",
                   }

        methods_order = [ upf_exp,
                          nodriftupf_exp,
                         alg_exp,
                          qcqp_exp,
                          nls_exp,
                        ]

        df, methods_names, methods_colors, methods_legends = taa.filter_methods_new(methods_order)
        g = taa.boxplot_exp(df, methods_color=methods_colors, methods_legend=methods_legends,
                        hue_variable="Name", hue_order=methods_names,
                        col_variable="Variable",
                        row_variable=None,
                        x_variable="Sigma_dv",
                        )

        g.axes_dict["error_x_relative"].set_yscale("log")
        g.axes_dict["error_h_relative"].set_xlabel(taa.y_label["error_h_relative"])
        g.axes_dict["error_x_relative"].set_xlabel(taa.y_label["error_x_relative"])
        sns.move_legend(g, loc="upper center", bbox_to_anchor= (0.5, 0.92), ncol=5)
        plt.subplots_adjust(top=0.8, bottom=0.12, left=0.06, right=0.99)
        plt.suptitle("Experiments")
        plt.show()

    def test_exp_time_analysis(self):
        # result_folder = "./Experiments/LOS_exp/Results/new_nls_correct_init_test/"
        result_folders = [
                            # "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/10hz",
                            # "./Experiments/LOS_exp/Results/experiments_paper/exp5"
                            "./Experiments/LOS_exp/Results/exp_cor_new1/exp"
                            ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        methods_order = [
                        #"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        #  # "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",

                        #  # "NLS|horizon=10",
                        #  # "algebraic|horizon=10",
                        #  # "algebraic|frequency=1.0|horizon=10",
                        #  # "algebraic|frequency=1.0|horizon=100",
                         "algebraic|frequency=10.0|horizon=100",
                        #  # "QCQP|horizon=10",
                         "QCQP|frequency=10.0|horizon=100",
                         # "QCQP|frequency=1.0|horizon=100",
                        #  "QCQP|frequency=10.0|horizon=1000",
                        # "NLS|frequency=1.0|horizon=10",
                        # "NLS|frequency=1.0|horizon=100",
                        "NLS|frequency=1.0|horizon=10",
                        "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        ]

        methods_color = {
                        "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                        "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         # "NLS|horizon=10": "tab:red",
                         # "algebraic|horizon=10": "tab:green",
                         "algebraic|frequency=1.0|horizon=100": "tab:orange",
                         "algebraic|frequency=10.0|horizon=100": "tab:orange",
                         # "QCQP|horizon=10": "tab:purple",
                         "QCQP|frequency=1.0|horizon=100": "tab:blue",
                         "QCQP|frequency=10.0|horizon=100": "tab:blue",
                        "NLS|frequency=1.0|horizon=100": "tab:purple",
                        "NLS|frequency=1.0|horizon=10": "tab:purple",
                        "NLS|frequency=1.0|horizon=10|perfect_guess=0": "tab:purple",
                        "Sigma": "tab:olive"
                        }

        methods_legend = {
                            "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours",
                            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours",
                          "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": r"Ours, $\tilde{\text{w}}$ pseudo-state",
                          "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": r"Ours, $\tilde{\text{w}}$ pseudo-state",
                          # "NLS|horizon=10": "NLS_10",
                          # "algebraic|horizon=10": "Algebraic_10",
                          "algebraic|frequency=1.0|horizon=10": "Algebraic 10s",
                          "algebraic|frequency=1.0|horizon=100": "Algebraic",
                          "algebraic|frequency=10.0|horizon=100": "Algebraic",
                          # "QCQP|horizon=10": "QCQP_10",
                          "QCQP|frequency=1.0|horizon=10": "QCQP",
                          "QCQP|frequency=10.0|horizon=1000": "QCQP",
                          "QCQP|frequency=10.0|horizon=100": "QCQP",
                            "NLS|frequency=1.0|horizon=100": "NLS",
                            "NLS|frequency=1.0|horizon=10": "NLS",
                        "NLS|frequency=1.0|horizon=10|perfect_guess=0": "NLS",

            "Sigma": r" Ours, $1 \sigma$-bound"}
        # taa.delete_data()
        taa.create_panda_dataframe()
        taa.time_analysis(sigma_uwbs=[0.15, 0.25], sigma_vs=[0.03, 0.08], frequencies = [1.0,10.0], start_time=0.,
                          methods_order=methods_order, methods_color=methods_color, methods_legend=methods_legend,
                          sigma_bound=True, save_fig=False)
        # taa.boxplot_LOS_comp_time(save_fig=False)
        # taa.calculation_time(save_fig=False)
        plt.show()



    def test_run_NLOS_exp(self):
        sig_v = 0.15 # issue with VIO being disturbed by the closets initial had to increase this to make it work.
        sig_w = 0.05
        sig_uwb = 0.25 # tried 0.3, 0.25 and 0.2

        main_folder = "./Experiments/NLOS_exp/"
        results_folder = main_folder + "Results_changed/"
        data_folder = main_folder + "Measurements/exp4_nlos_a_sampled.pkl"

        experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool= True
        # tas.plot_bool = True

        # tas.run_experiment(methods=["upf", "losupf"], redo_bool=True, experiment_data=experiment_data)
        # tas.run_experiment(methods=["NLS"], redo_bool=True, experiment_data=experiment_data)
        tas.run_experiment(methods=["upf", "NLS", "losupf"], redo_bool=False, experiment_data=experiment_data)
        plt.show()
        return tas, measurements

    def test_nlos_detection_plot(self):
        tas, measurements = self.test_run_NLOS_exp()

        sig_uwb = 0.25
        sig2_los = measurements[0].get_uwb_LOS(2*sig_uwb)
        sig3_los = measurements[0].get_uwb_LOS(3 * sig_uwb)
        sigma2_los_t = [i/10 for i in range(len(sig2_los)) if (i%10==0 and sig2_los[i]==0)]
        sigma3_los_t = [i/10 for i in range(len(sig3_los)) if (i%10==0 and sig3_los[i]==0)]

        drone_0_los = tas.data["exp4_nlos_a_sampled"]["upf"]["drone_0"]["los_state"]
        drone_0_los_t = [i for i in range(len(drone_0_los)) if (drone_0_los[i]==0)]
        drone_0_los_x = [0 for i in range(len(drone_0_los_t))]
        #
        drone_1_los = tas.data["exp4_nlos_a_sampled"]["upf"]["drone_1"]["los_state"]
        drone_1_los_t = [i for i in range(len(drone_1_los)) if (drone_1_los[i]==0)]
        drone_1_los_x = [1 for i in range(len(drone_1_los_t))]

        _, ax = plt.subplots(2,1)
        # plt.figure()
        for sigma2_los in sigma2_los_t:
            ax[1].axvline(x=sigma2_los, color="tab:blue", linestyle="--", linewidth=3)
        for sigma3_los in sigma3_los_t:
            ax[1].axvline(x=sigma3_los, color="tab:red", linestyle="--", linewidth=3)

        plt.text(0.25,0.92, "agent 1 NLOS", fontsize=12)
        plt.text(0.25, 0, "agent 0 NLOS", fontsize=12)
        ax[1].plot(drone_0_los_t, drone_0_los_x, ".k")
        ax[1].plot(drone_1_los_t, drone_1_los_x, ".k")
        ax[1].plot(0,0, color = "tab:blue", linestyle="--", label=r"$\epsilon_{d} > 2\sigma_d$", linewidth=3)
        ax[1].plot(0, 0, color="tab:red", linestyle="--", label=r"$\epsilon_{d} > 3\sigma_d$",linewidth=3)
        ax[1].set_xlabel("Time [s]")
        plt.yticks([])

        # ax[1].set_ylabel("Robot")
        ax[1].legend(loc='center left', fontsize=12)


        uwb_1 = measurements[0].uwb
        uwb_1.plot_real(factor=10, ax =ax[0])
        ax[0].legend(loc='upper left', fontsize=12)

        ax[0].set_xlim([-11, 259])
        ax[1].set_xlim([-11, 259])
        print(drone_0_los_t)
        print(drone_1_los_t)
        print(sigma2_los_t)
        print(sigma3_los_t)



        plt.show()
        return tas

    def test_plot_NLOS_error(self):
        tas, measurements = self.test_run_NLOS_exp()
        agent_nr = 0
        agent_name = "drone_"+str(agent_nr)

        exp_name = "exp4_nlos_a_sampled"
        upf_x_error_0 = tas.data[exp_name]["upf"][agent_name]["error_x_relative"]
        NLS_x_error_0 = tas.data[exp_name]["NLS"][agent_name]["error_x_relative"]
        losupf_x_error_0 = tas.data[exp_name]["losupf"][agent_name]["error_x_relative"]
        slam_x_error_0 = tas.data[exp_name]["slam"][agent_name]["error_x_relative"]

        upf_h_error_0 = tas.data[exp_name]["upf"][agent_name]["error_h_relative"]
        NLS_h_error_0 = tas.data[exp_name]["NLS"][agent_name]["error_h_relative"]
        losupf_h_error_0 = tas.data[exp_name]["losupf"][agent_name]["error_h_relative"]
        slam_h_error_0 = tas.data[exp_name]["slam"][agent_name]["error_h_relative"]


        agent_nr = 1
        agent_name = "drone_"+str(agent_nr)
        upf_x_error_1 = tas.data[exp_name]["upf"][agent_name]["error_x_relative"]
        NLS_x_error_1 = tas.data[exp_name]["NLS"][agent_name]["error_x_relative"]
        losupf_x_error_1 = tas.data[exp_name]["losupf"][agent_name]["error_x_relative"]
        slam_x_error_1 = tas.data[exp_name]["slam"][agent_name]["error_x_relative"]

        _, ax  = plt.subplots(1, 2)
        ax[0].plot(NLS_x_error_0, label="NLS [7]", color="tab:blue", linewidth=3)
        ax[0].plot(losupf_x_error_0, label="", color="tab:red", linewidth=3)
        ax[0].plot(upf_x_error_0, label="UPF (ours)", color="tab:green", linewidth=3)
        # ax[0].plot(slam_x_error_0, label="slam")
        ax[0].set_xlabel("Time [s]", fontsize=12)
        ax[0].set_ylabel(r"$\epsilon_{\hat{p}^t}$ [m]", fontsize=12)
        ax[0].grid()
        # ax[0].legend()


        ax[1].plot(NLS_h_error_0, label="NLS [7]", color="tab:blue", linewidth=3)
        ax[1].plot(losupf_h_error_0, label=r"UPF $\tilde{w}$ $s_{LOS}$ (ours)",color="tab:red", linewidth=3)
        ax[1].plot(upf_h_error_0, label="UPF (ours)", color="tab:green",  linewidth=3)
        ax[1].set_xlabel("Time [s]", fontsize=12)
        ax[1].set_ylabel(r"$\epsilon_{\hat{\theta}^t}$ [(rad))]", fontsize=12)
        ax[1].grid()
        ax[1].set_ylim([0., 0.5])
        # ax[1].plot(slam_h_error_0, label="slam")
        ax[1].legend()

        plt.show()

        # ax[1].plot(upf_x_error_1, label="upf")
        # ax[1].plot(NLS_x_error_1, label="NLS")
        # ax[1].plot(losupf_x_error_1, label="losupf")
        # # ax[1].plot(slam_x_error_1, label="slam")
        # ax[1].legend()



    def test_plot_augmented_NLOS_Error_Graph(self):
        tas, measurements = self.test_run_NLOS_exp()
        agent_nr = 0
        agent_name = "drone_" + str(agent_nr)
        labels = ["Original", "$d_{NLOS}=2m$", "$d_{NLOS}=10m$"]

        exp = ["exp4_nlos_a_sampled","exp4_nlos_a_changed_2_sampled" ,"exp4_nlos_a_changed_10_sampled"]
        line_style = ["-", "--", ":"]
        alpha = [0.3, 0.5, 1]

        plt.figure()

        plt.plot(0, 0, color="tab:blue", linestyle="-", alpha=1,  linewidth=3, label="NLS [7]")
        plt.plot(0, 0, color="tab:red", linestyle="-", alpha=1,  linewidth=3, label= r"UPF $\tilde{w}$ $s_{LOS}$ (ours)")
        plt.plot(0, 0, color="tab:green", linestyle="-", alpha=1,  linewidth=3, label="UPF (ours)")
        for i, exp_name in enumerate(exp):
            upf_error = tas.data[exp_name]["upf"][agent_name]["error_x_relative"]
            NLS_error = tas.data[exp_name]["NLS"][agent_name]["error_x_relative"]
            losupf_error= tas.data[exp_name]["losupf"][agent_name]["error_x_relative"]

            plt.plot(0, 0, color="k", linestyle=line_style[i], alpha=alpha[i], label=labels[i], linewidth=3)
            plt.plot(NLS_error, color="tab:blue", linestyle=line_style[i], alpha=alpha[i], linewidth=3)
            plt.plot(losupf_error, color="tab:red", linestyle=line_style[i], alpha=alpha[i], linewidth=3)
            plt.plot(upf_error, color="tab:green", linestyle=line_style[i], alpha=alpha[i], linewidth=3)
        plt.ylim([0, 2.])
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel(r"$\epsilon_{\hat{p}^t}$ [m]", fontsize=12)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid()
        plt.show()

    def test_unobservable_motion(self):
        sig_v = 0.15
        sig_w = 0.06
        sig_uwb = 0.3

        main_folder = "./Experiments/Unob_exp/Measurements/"
        results_folder = main_folder + "Results/"
        data_folder = main_folder + "Measurements/"
        data_folder = main_folder + "exp2_unobservable_sampled.pkl"
        print(data_folder)
        experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = False
        # tas.run_experiment(methods=["NLS", "algebraic", "upf", "losupf", "nodriftupf"], redo_bool=False, experiment_data=experiment_data)
        tas.run_experiment(methods=["losupf|frequency=1.0|resample_factor=0.5|sigma_uwb_factor=1.0"], redo_bool=True, experiment_data=experiment_data)
        # upf_x_error = tas.data["exp1_unobservable_sampled"]["losupf"]["drone_1"]["error_x_relative"]
        # slam_x_error = tas.data["exp1_unobservable_sampled"]["slam"]["drone_1"]["error_x_relative"]
        ax = plt.figure().add_subplot(projection='3d')

        tas.agents["drone_0"]["log"].plot_poses(ax, color_ha="darkblue", color_ca="red", name_ha="$a_0$", name_ca="$a_1$")
        # ax = plt.figure().add_subplot(projection='3d')
        tas.agents["drone_1"]["log"].plot_poses(ax, color_ha="maroon", color_ca="dodgerblue",
                                                                           name_ha="$a_1$", name_ca="$a_0$")
        ax.plot(10,10, color="darkblue", label="Real pose $i$" )
        ax.plot(10,10, color='red', alpha=1, linestyle="-", label="Active particles for $j$ by $i$")  # for estimation of "+ name)
        ax.plot(10,10, color='red', alpha=0.4, linestyle=":", label="Killed particles for $j$ by $i$")  # for estimation of "+ name)
        ax.plot(10,10, color="maroon", label="Real pose $j$")
        ax.plot(10,10, color='dodgerblue', alpha=1, linestyle="-", label="Active particles for $i$  by $j$")  # for estimation of "+ name)
        ax.plot(10,10, color='dodgerblue', alpha=0.4, linestyle=":", label="Killed particles for $i$ by $j$")  # for estimation of "+ name)
        plt.plot(10, 10, color="black", linestyle="", marker="o", label="Start of a trajectory")
        plt.plot(10, 10, color="black", linestyle="", marker="x", label="End of a trajectory")

        ax.legend(fontsize=10, loc="upper left")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        # plt.figure()
        # tas.agents["drone_1"]["upf"].upf_connected_agent_logger
        plt.show()
        return tas, measurements

    def test_quickfix_legend_unobservable(self):
        plt.figure()
        plt.plot(0, 0, color="darkblue", label="Real pose $i$")
        plt.plot(0, 0, color='red', alpha=1, linestyle="--",
                label="Active particles for $j$ by $i$")  # for estimation of "+ name)
        plt.plot(0, 0, color='red', alpha=0.1, linestyle=":",
                label="Killed particles for $j$ by $i$")  # for estimation of "+ name)
        plt.plot(0, 0, color="maroon", label="Real pose $j$")
        plt.plot(0, 0, color='dodgerblue', alpha=1, linestyle="--",
                label="Active particles for $j$  by $j$")  # for estimation of "+ name)
        plt.plot(0, 0, color='dodgerblue', alpha=0.2, linestyle=":",
                label="Killed particles for $i$ by $j$")  # for estimation of "+ name)
        plt.plot(0, 0, color="black", linestyle="", marker="o", label="Start of a trajectory")
        plt.plot(0, 0, color="black", linestyle="", marker="x", label="End of a trajectory")
        plt.legend(fontsize=10)
        plt.show()

    def test_particle_generation(self):
        sig_v = 0.15
        sig_w = 0.03
        sig_uwb = 0.25

        result_folder = "Analysis/Particle_count"
        data_file = "Experiments/Exp3_SemiNLOS/Measurements/exp3_sec1_los_sampled.pkl"
        experiment_data, _ = create_experimental_data(data_file, sig_v, sig_w, sig_uwb)
        tas = create_experiment(result_folder, sig_v, sig_w, sig_uwb)
        # tas.run_experiment(methods=[ "upf"], redo_bool=True, experiment_data=experiment_data)

        tas.run_experiment(methods=[ "upf", "upfnaive"], redo_bool=False,
                           experiment_data=experiment_data)

        plt.figure(figsize=(6, 3))
        plt.plot(tas.data["exp3_sec1_los_sampled"]["upf"]["drone_0"]["number_of_particles"], label="NLOS UPF (ours)",
                 color="tab:orange", linewidth=3)
        plt.plot(tas.data["exp3_sec1_los_sampled"]["upfnaive"]["drone_0"]["number_of_particles"],
                 label="Naive sampling UPF", color="k", linewidth=3)
        plt.xlim([0, 60])
        plt.yscale("log")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("Number of particles", fontsize=12)
        plt.legend(fontsize=12)

        # plt.figure(figsize=(6, 3))
        # plt.plot(tas.data["exp3_sec1_los_sampled"]["upf"]["drone_0"]["calculation_time"], label="UPF (ours)",
        #          color="tab:green", linewidth=3)
        # plt.plot(tas.data["exp3_sec1_los_sampled"]["upfnaive"]["drone_0"]["calculation_time"],
        #          label="UPF naive sampling", color="tab:red", linewidth=3)
        # plt.xlim([0, 60])
        # plt.yticks( fontsize=12)
        # plt.xticks(fontsize=12)
        # plt.xlabel("Time [s]", fontsize=12)
        # plt.ylabel("Calculation time [s]", fontsize=12)
        # plt.legend(fontsize=12)

        plt.show()

        return tas

if __name__ == '__main__':
    t = MyTestCase()
    tas = t.test_particle_generation()

    # tas, exp = t.test_run_NLOS_exp()



    # plt.figure()
    # tas.agents["drone_0"]["upf"].upf_connected_agent_logger.
    # exp = t.create_experimental_data("./Experiments/LOS_exp/Measurements/")
    # tb2, tb3 = t.test_new_robot_population()

    # tas = t.test_run_exp()

