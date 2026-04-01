import unittest
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import Code.Simulation.MultiRobotClass as MRC
import Code.Analysis.TwoAgentAnalysis as TAA
from Code.DataLoggers.ConnectedAgent_DataLogger import UPFConnectedAgentDataLogger as UPFDL
from Code.DataLoggers.TargetTrackingUKF_DataLogger import UKFDatalogger as UKFDL
import pickle as pkl
import os
import seaborn as sns
class MyTestCase(unittest.TestCase):

    def test_TAS_RPE(self):
        test = "test_new_system"
        result_folder = "Results/" + test
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        # Parameters
        alpha = 1
        kappa = -1.
        beta = 2.
        n_azimuth = 4
        n_altitude = 3
        n_heading = 4
        sigma_dv = 0.01
        sigma_dw = 0.1 * sigma_dv
        sigma_uwb = 0.1

        # for i in range(4):
        #     mrss  = MRC.MultiRobotSingleSimulation(folder = "robot_trajectories/"+test_na_5_na_8_nh_8+"/sim_"+str(i))
        #     mrss.delete_sim(sigma_dv, sigma_dw, sigma_uwb)

        TAS = MRC.TwoAgentSystem(trajectory_folder="small_robot_trajectories/",
                                 result_folder=result_folder)

        TAS.debug_bool = True
        TAS.plot_bool = False
        TAS.save_folder = ("./save_data_test")
        TAS.save_bool = True
        TAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)
        TAS.set_ukf_properties(alpha, beta, kappa, n_azimuth, n_altitude, n_heading)
        # TAS.run_simulations(methods=["losupf", "nodriftupf", "algebraic", "NLS", "QCQP"], redo_bool=True)
        methods = ["losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                   "algebraic|frequency=1.0|horizon=10",
                   "algebraic|frequency=10.0|horizon=100",
                   "QCQP|frequency=1.0|horizon=10",
                   "QCQP|frequency=10.0|horizon=100"]
        TAS.run_simulations(methods=methods, redo_bool=False)

    def test_UPF_detail(self):
        upfs = []
        for sim in range(11):
            upfs.append(pkl.load(open("save_data/sim_"+str(sim)+"/drone_0_losupf.pkl", "rb")))
            upfs.append(pkl.load(open("save_data/sim_"+str(sim)+"/drone_1_losupf.pkl", "rb")))
        return upfs
        # plt.show()

    def test_rename_simulations(self):
        save_dir = "../../../Data/Results/Sim_LOS_06_2024"
        load_dir = "../../../Data/Results/Standard_LOS_06_2024"
        n_files = len(os.listdir(load_dir))
        n_file = 0
        for file in os.listdir(load_dir):
            n_file += 1
            print(str(int(n_file/n_files*100)) + "%: " +  file)
            if os.path.isfile(load_dir + "/"+file) and not os.path.exists(save_dir + "/sim_" + file):
                os.rename(load_dir + "/" + file, save_dir + "/sim_"+ file)
                with open(save_dir + "/sim_" + file, "rb") as f:
                    data = pkl.load(f)
                f.close()
                with open(save_dir + "/sim_" + file, "wb") as f:
                    data["parameters"]["type"] = "simulation"
                    pkl.dump(data, f)
                f.close()

    def load_data(self, variables=["error_x_relative", "error_h_relative", "calculation_time"]):

        sigma_dv = [0.1, 0.01]
        sigma_dw = [0.1, 0.01]
        sigma_uwb = [1., 0.1]
        frequency =[1.0]

        upf_sim_full = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "Variables": {
                            "Type": ["simulation"],
                            "Variable": variables,
                            "Sigma_dv": sigma_dv,
                            "Sigma_dw": sigma_dw,
                            "Sigma_uwb": sigma_uwb,
                            "Frequency": frequency,
                        },
                        "Color": "tab:green",
                        "Legend": "Ours, proposed",
                        }
        upf_sim_full_per = {
            "Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
            "Variables": {
                "Type": ["simulation"],
                "Variable": variables,
                "Sigma_dv": sigma_dv,
                "Sigma_dw": sigma_dw,
                "Sigma_uwb": sigma_uwb,
                "Frequency": [1],
            },
            "Color": "tab:orange",
            "Legend": "Ours*",
            }
        nodriftupf_sim_full = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                               "Variables": {
                                   "Type": ["simulation"],
                                   "Variable": variables,
                                   "Sigma_dv": sigma_dv,
                                   "Sigma_dw": sigma_dw,
                                   "Sigma_uwb": sigma_uwb,
                                   "Frequency": [10.0],
                               },
                               "Color": "tab:red",
                               "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state",
                               }
        alg_sim_full = {"Method": "algebraic|frequency=10.0|horizon=100",
                        "Variables": {
                            "Type": ["simulation"],
                            "Variable": variables,
                            "Sigma_dv": sigma_dv,
                            "Sigma_dw": sigma_dw,
                            "Sigma_uwb": sigma_uwb,
                            "Frequency": [10.0],
                        },
                        "Color": "tab:orange",
                        "Legend": "Algebraic",
                        }
        qcqp_sim_full = {"Method": "QCQP|frequency=10.0|horizon=100",
                         "Variables": {
                             "Type": ["simulation"],
                             "Variable": variables,
                             "Sigma_dv": sigma_dv,
                             "Sigma_dw": sigma_dw,
                             "Sigma_uwb": sigma_uwb,
                             "Frequency": [10.0],
                         },
                         "Color": "tab:blue",
                         "Legend": "QCQP",
                         }
        nls_sim_full = {
            "Method": "NLS|frequency=1.0|horizon=10",
            "Variables": {
                "Type": ["simulation"],
                "Variable": variables,
                "Sigma_dv": sigma_dv,
                "Sigma_dw": sigma_dw,
                "Sigma_uwb": sigma_uwb,
                "Frequency": [1.0],
            },
            "Color": "tab:purple",
            "Legend": "NLS*",
        }

        methods_order_sim_full = [upf_sim_full,
                                  nodriftupf_sim_full,
                                  # alg_sim_full,
                                  qcqp_sim_full,
                                  upf_sim_full_per,
                                  nls_sim_full,
                                  ]
        return methods_order_sim_full

    def test_print_statistics(self):
        result_folders = [
            "../../../Results/simulations_1hz",
        ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        variables = ["error_x_relative", "error_h_relative", "calculation_time"]
        methods_order = self.load_data(variables=variables)

        df, methods_names, methods_colors, methods_legends = taa.filter_methods_new(methods_order)
        taa.print_statistics(methods_names, variables, df)

    def test_sim_analysis(self):
        result_folders = [
            "../../../Results/simulations",
        ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        variables = ["error_x_relative", "error_h_relative"]
        methods_order = self.load_data(variables)

        df_sim_full, methods_names_sim_full, methods_colors_sim_full, methods_legends_sim_full = taa.filter_methods_new(
            methods_order)

        g = taa.boxplot_exp(df_sim_full, methods_color=methods_colors_sim_full,
                                    methods_legend=methods_legends_sim_full,
                                    hue_variable="Name", hue_order=methods_names_sim_full,
                                    col_variable="Variable", col_order=["error_x_relative", "error_h_relative"],
                                    row_variable="Sigma_dv", row_order=[0.01, 0.1],
                                    x_variable="Sigma_uwb", x_order=[0.1, 1.],
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
        sns.move_legend(g, loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=5)
        plt.subplots_adjust(top=0.8, bottom=0.12, left=0.12, right=0.99)
        # plt.suptitle("Monte Carlo simulation")
        plt.show()

    @DeprecationWarning
    def test_time_analysis(self):
        result_folder =  "../../../Data/Results/Standard_LOS_05_2024/alfa_1_434_server_02_06_24/1hz"
        # result_folder =  "../../../Data/Results/Standard_LOS_05_2024/alfa_1_434/10hz"
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        taa.create_panda_dataframe()
        taa.boxplot_LOS_comp_time(save_fig=False)
        plt.show()

    @DeprecationWarning
    def test_exp_time_analysis(self):
        # result_folder = "./Experiments/LOS_exp/Results/new_nls_correct_init_test/"
        result_folders = [
            "./Results/test_new_system",
            # "../../../Data/Results/Sim_LOS_06_2024/1_sim",
            # "../../../Data/Results/Sim_LOS_06_2024/final_methods_RPE_paper",
                            ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        methods_order = [
                        #"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        #  # "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        #  # "NLS|horizon=10",
                        #  # "algebraic|horizon=10",
                        #  # "algebraic|frequency=1.0|horizon=10",
                        #  # "algebraic|frequency=1.0|horizon=100",
                        #  "algebraic|frequency=10.0|horizon=1000",
                        #  # "QCQP|horizon=10",
                        #  # "QCQP|frequency=10.0|horizon=100",
                        #  # "QCQP|frequency=1.0|horizon=100",
                        #  "QCQP|frequency=10.0|horizon=1000",
                        # "NLS|frequency=1.0|horizon=10",
                        # "NLS|frequency=1.0|horizon=100",
                        "NLS|frequency=1.0|horizon=10",
        ]

        methods_color = {
                        "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                        "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         # "NLS|horizon=10": "tab:red",
                         # "algebraic|horizon=10": "tab:green",
                         "algebraic|frequency=1.0|horizon=100": "tab:orange",
                         "algebraic|frequency=10.0|horizon=1000": "tab:orange",
                         # "QCQP|horizon=10": "tab:purple",
                         "QCQP|frequency=1.0|horizon=100": "tab:blue",
                         "QCQP|frequency=10.0|horizon=1000": "tab:blue",
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
                          "algebraic|frequency=10.0|horizon=1000": "Algebraic",
                          # "QCQP|horizon=10": "QCQP_10",
                          "QCQP|frequency=1.0|horizon=10": "QCQP",
                          "QCQP|frequency=10.0|horizon=1000": "QCQP",
                          "QCQP|frequency=1.0|horizon=100": "QCQP",
                            "NLS|frequency=1.0|horizon=100": "NLS",
                            "NLS|frequency=1.0|horizon=10": "NLS",
                        "NLS|frequency=1.0|horizon=10|perfect_guess=0": "NLS",

                             "Sigma": r" Ours, $1 \sigma$-bound"}
        # taa.delete_data()
        taa.create_panda_dataframe()
        taa.time_analysis(sigma_uwbs=[1.0], sigma_vs=[0.1], frequencies = [1.0,10.0], start_time=0.,
                          methods_order=methods_order, methods_color=methods_color, methods_legend=methods_legend,
                          sigma_bound=False, save_fig=False)
        # taa.boxplot_LOS_comp_time(save_fig=False)
        # taa.calculation_time(save_fig=False)
        plt.show()

    def create_result_dict(self, method, variables, sig_dv, sig_duwb, exp_type, frequencies, legend, color):
        result_dict = {"Method": method,
                       "Variables": {
                           "Type": exp_type,
                           "Variable": variables,
                           "Sigma_dv": sig_dv,
                           "Sigma_uwb": sig_duwb,
                           "Frequency": frequencies,
                       },
                       "Color": color,
                       "Legend": legend,
                       }
        return result_dict

    def test_time_analysis_new(self):
        result_folder = [
                        # "./Results/UPF_perfect_guess",
                        # "../../../Data/Results/Sim_LOS_06_2024/1_sim",
                        # "../../../Data/Results/Sim_LOS_06_2024/final_methods_RPE_paper",
                        "./Results/small_trajectories",
                         # "../../../test_cases/RPE_2_agents_LOS/Experiments/Experiments/LOS_exp/Results/experiments_paper/Experiments",
                         ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)

        variables = ["error_x_relative", "error_h_relative"]
        upf_sim = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                          variables=variables,
                                          sig_dv=[0.1], sig_duwb=[0.1], exp_type=["simulation"], frequencies=[10.0],
                                          legend="UPF sim, $\sigma_{uwb} = 0.1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="red")
        upf_sim_2 = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                            variables=variables,
                                            sig_dv=[0.1], sig_duwb=[1.], exp_type=["simulation"],
                                            frequencies=[10.0],
                                            legend="UPF sim, $\sigma_{uwb} = 1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="blue")
        upf_exp = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                          variables=variables,
                                          sig_dv=[0.08], sig_duwb=[0.25], exp_type=["experiment"],
                                          frequencies=[10.0],
                                          legend="UPF exp, $\sigma_{uwb} = 0.25$, $\sigma_{v} = 0.08$, $\sigma_{w} = 0.12$", color="tab:green")
        nls_sim = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                          sig_dv=[0.1], sig_duwb=[0.1], exp_type=["simulation"], frequencies=[1.0],
                                          legend="NLS sim, $\sigma_{uwb} = 0.1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$",
                                          color="salmon")
        nls_sim2 = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                           sig_dv=[0.1], sig_duwb=[1.], exp_type=["simulation"], frequencies=[1.0],
                                           legend="NLS sim, $\sigma_{uwb} = 1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$",
                                           color="lightblue")
        nls_exp = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                          sig_dv=[0.08], sig_duwb=[0.35], exp_type=["experiment"],
                                          frequencies=[1.0],
                                          legend="NLS exp, $\sigma_{uwb} = 0.25$, $\sigma_{v} = 0.08$, $\sigma_{w} = 0.12$", color="limegreen")
        upf_p_exp = self.create_result_dict(
            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0", variables=variables,
            sig_dv=[0.08], sig_duwb=[0.25], exp_type=["experiment"], frequencies=[10.0],
            legend="UPF per exp, $\sigma_{uwb} = 0.25$, $\sigma_{v} = 0.08$, $\sigma_{w} = 0.12$", color="tab:green")
        upf_p_sim = self.create_result_dict(
            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0", variables=variables,
            sig_dv=[0.1], sig_duwb=[0.1], exp_type=["simulation"], frequencies=[10.0],
            legend="UPF per sim, $\sigma_{uwb} = 0.1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="red")
        upf_p_sim2 = self.create_result_dict(
            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0", variables=variables,
            sig_dv=[0.1], sig_duwb=[1.0], exp_type=["simulation"], frequencies=[10.0],
            legend="UPF per sim, $\sigma_{uwb} = 1.0$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="blue")

        methods_order = [
                        upf_exp, nls_exp, upf_sim, nls_sim, upf_sim_2, nls_sim2,
                         # upf_p_sim, upf_p_sim2,
                         # alg_exp,
                         #  qcqp_exp,
                         #  nls_exp,
                         ]

        df, methods_names, methods_colors,  methods_styles, methods_legends = taa.filter_methods_new(methods_order)
        g = taa.lineplot(df, methods_names, methods_colors, methods_styles=methods_styles, methods_legends=methods_legends)
        # plt.legend(loc="upper left")
        plt.show()

    def test_time_analysis_new_perfect_guess(self):
        result_folder = [
            # "./Results/UPF_perfect_guess",
            # "../../../test_cases/RPE_2_agents_LOS/Experiments/Experiments/LOS_exp/Results/UPF_perfect_guess",
            # # "../../../Data/Results/Sim_LOS_06_2024/1_sim",
            # "../../../Data/Results/Sim_LOS_06_2024/final_methods_RPE_paper",
            "./Results/small_trajectories",
            "../../../test_cases/RPE_2_agents_LOS/Experiments/Experiments/LOS_exp/Results/experiments_paper/Experiments",
        ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)

        variables = ["error_x_relative", "error_h_relative"]

        upf_sim = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                          variables=variables,
                                          sig_dv=[0.1], sig_duwb=[0.1], exp_type=["simulation"], frequencies=[10.0],
                                          legend="UPF sim, $\sigma_{uwb} = 0.1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="red")
        upf_sim_2 = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                            variables=variables,
                                            sig_dv=[0.1], sig_duwb=[1.], exp_type=["simulation"],
                                            frequencies=[10.0],
                                            legend="UPF sim, $\sigma_{uwb} = 1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="blue")
        upf_exp = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                          variables=variables,
                                          sig_dv=[0.08], sig_duwb=[0.25], exp_type=["experiment"],
                                          frequencies=[10.0],
                                          legend="UPF exp, $\sigma_{uwb} = 0.25$, $\sigma_{v} = 0.08$, $\sigma_{w} = 0.12$", color="tab:green")
        nls_sim = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                          sig_dv=[0.1], sig_duwb=[0.1], exp_type=["simulation"], frequencies=[1.0],
                                          legend="NLS sim, $\sigma_{uwb} = 0.1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$",
                                          color="salmon")
        nls_sim2 = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                           sig_dv=[0.1], sig_duwb=[1.], exp_type=["simulation"], frequencies=[1.0],
                                           legend="NLS sim, $\sigma_{uwb} = 1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$",
                                           color="lightblue")
        nls_exp = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                          sig_dv=[0.08], sig_duwb=[0.35], exp_type=["experiment"],
                                          frequencies=[1.0],
                                          legend="NLS exp, $\sigma_{uwb} = 0.25$, $\sigma_{v} = 0.08$, $\sigma_{w} = 0.12$", color="limegreen")
        upf_p_exp = self.create_result_dict(
            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0", variables=variables,
            sig_dv=[0.08], sig_duwb=[0.25], exp_type=["experiment"], frequencies=[10.0],
            legend="UPF per exp, $\sigma_{uwb} = 0.25$, $\sigma_{v} = 0.08$, $\sigma_{w} = 0.12$", color="tab:green")
        upf_p_sim = self.create_result_dict(
            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0", variables=variables,
            sig_dv=[0.1], sig_duwb=[0.1], exp_type=["simulation"], frequencies=[10.0],
            legend="UPF per sim, $\sigma_{uwb} = 0.1$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="red")
        upf_p_sim2 = self.create_result_dict(
            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0", variables=variables,
            sig_dv=[0.1], sig_duwb=[1.0], exp_type=["simulation"], frequencies=[10.0],
            legend="UPF per sim, $\sigma_{uwb} = 1.0$, $\sigma_{v} = 0.1$, $\sigma_{w} = 0.1$", color="blue")

        methods_order = [
            upf_p_exp, nls_exp, upf_p_sim, nls_sim, upf_p_sim2, nls_sim2,
            # upf_p_sim, upf_p_sim2,
            # alg_exp,
            #  qcqp_exp,
            #  nls_exp,
        ]

        df, methods_names, methods_colors, methods_styles, methods_legends = taa.filter_methods_new(methods_order)
        g = taa.lineplot(df, methods_names, methods_colors, methods_styles=methods_styles,
                         methods_legends=methods_legends)
        # plt.legend(loc="upper left")
        plt.show()


    def test_10hz_vs_1h(self):
        result_folder = [
            # "./Results/UPF_perfect_guess",
            # "../../../test_cases/RPE_2_agents_LOS/Experiments/Experiments/LOS_exp/Results/UPF_perfect_guess",
            # # "../../../Data/Results/Sim_LOS_06_2024/1_sim",
            # "../../../Data/Results/Sim_LOS_06_2024/final_methods_RPE_paper",
            "../../../Results/simulations_1hz",
            "../../../Results/simulations",
        ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        variables = ["error_x_relative", "error_h_relative"]
        sigma_dv = [0.1, 0.01]
        sigma_dw = [0.1, 0.01]
        sigma_uwb = [1., 0.1]
        frequency = [1.0]  # [10.0]
        # create 1hz
        upf_sim_1 = self.create_result_dict("losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                          variables=variables,
                                          sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"], frequencies=frequency,
                                          legend=r"UPF 1hz",
                                          color="lightgreen")
        upf_sim_per_1 = self.create_result_dict("losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                                          variables=variables,
                                          sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"], frequencies=frequency,
                                          legend=r"UPF * 1hz",
                                          color="bisque")
        upfnodrift_sim_1 = self.create_result_dict("nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                            variables=variables,
                                            sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"],
                                            frequencies=frequency,
                                            legend=r"UPF  $\tilde{\text{w}}$ pseudo-state  1hz",
                                            color="salmon")
        nls_sim_1 = self.create_result_dict("NLS|frequency=0.1|horizon=1", variables=variables,
                                          sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"], frequencies=[0.1],
                                          legend=r"NLS * 1hz",
                                          color="thistle")
        qcqp_sim_1 = self.create_result_dict("QCQP|frequency=1.0|horizon=10", variables=variables,
                                            sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"],
                                            frequencies=frequency,
                                            legend=r"QCQP 1hz",
                                            color="cornflowerblue")
        alg_sim_1 = self.create_result_dict("algebraic|frequency=1.0|horizon=10", variables=variables,
                                             sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"],
                                             frequencies=frequency,
                                             legend=r"Algebraic 1hz",
                                             color="chocolate")

        frequency = [10.0]  # [10.0]
        # create 1hz
        upf_sim_10 = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                          variables=variables,
                                          sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"], frequencies=frequency,
                                          legend=r"UPF 10hz",
                                          color="tab:green")
        upf_sim_per_10 = self.create_result_dict("losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                                          variables=variables,
                                          sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"], frequencies=frequency,
                                          legend=r"UPF * 10hz",
                                          color="tab:brown")
        upfnodrift_sim_10 = self.create_result_dict("nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                                                   variables=variables,
                                                   sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"],
                                                   frequencies=frequency,
                                                   legend=r"UPF  $\tilde{\text{w}}$ pseudo-state  10hz",
                                                   color="tab:red")
        nls_sim_10 = self.create_result_dict("NLS|frequency=1.0|horizon=10", variables=variables,
                                          sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"], frequencies=[1.0],
                                          legend=r"NLS * 10hz",
                                          color="tab:purple")
        qcqp_sim_10 = self.create_result_dict("QCQP|frequency=10.0|horizon=100", variables=variables,
                                            sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"],
                                            frequencies=frequency,
                                            legend=r"QCQP 10hz",
                                            color="tab:blue")
        alg_sim_10 = self.create_result_dict("algebraic|frequency=10.0|horizon=100", variables=variables,
                                             sig_dv=sigma_dv, sig_duwb=sigma_uwb, exp_type=["simulation"],
                                             frequencies=frequency,
                                             legend=r"Algebraic 10hz",
                                             color="tab:orange")

        methods_order_sim_full = [ upf_sim_10,
                                    upf_sim_1,
                                   upfnodrift_sim_10,
                                  upfnodrift_sim_1,
                                   qcqp_sim_10,
                                  qcqp_sim_1,
                                   alg_sim_10,
                                  alg_sim_1,
                                   upf_sim_per_10,
                                  upf_sim_per_1,
                                      nls_sim_10,
                                  nls_sim_1
                                  ]

        df_sim_full, methods_names_sim_full, methods_colors_sim_full, methods_legends_sim_full = taa.filter_methods_new(
            methods_order_sim_full)

        g = taa.boxplot_exp(df_sim_full, methods_color=methods_colors_sim_full,
                                    methods_legend=methods_legends_sim_full,
                                    hue_variable="Name", hue_order=methods_names_sim_full,
                                    col_variable="Variable", col_order=["error_x_relative", "error_h_relative"],
                                    row_variable="Sigma_dv", row_order=[0.01, 0.1],
                                    x_variable="Sigma_uwb", x_order=[0.1, 1.],
                                    )
        g.fig.set_size_inches(24, 6)
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
        sns.move_legend(g, loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=6)
        plt.subplots_adjust(top=0.8, bottom=0.12, left=0.12, right=0.99)
        # plt.suptitle("Monte Carlo simulation")
        plt.show()

if __name__ == '__main__':
    # unittest.main()
    t = MyTestCase()
    upfs = t.test_UPF_detail()



