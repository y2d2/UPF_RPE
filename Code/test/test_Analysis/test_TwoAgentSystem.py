import unittest



import matplotlib
from tensorflow.python.ops.initializers_ns import variables

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
                   "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
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

        upf_sim_full = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "Variables": {
                            "Type": ["simulation"],
                            "Variable": variables,
                            "Sigma_dv": sigma_dv,
                            "Sigma_dw": sigma_dw,
                            "Sigma_uwb": sigma_uwb,
                            "Frequency": [10.0],
                        },
                        "Color": "tab:green",
                        "Legend": "Ours",
                        }
        upf_sim_full_per = {
            "Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
            "Variables": {
                "Type": ["simulation"],
                "Variable": variables,
                "Sigma_dv": sigma_dv,
                "Sigma_dw": sigma_dw,
                "Sigma_uwb": sigma_uwb,
                "Frequency": [10.0],
            },
            "Color": "tab:orange",
            "Legend": "Ours *",
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
            "Legend": "NLS *",
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
            "../../../Results/simulations",
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


    def test_time_analysis(self):
        result_folder =  "../../../Data/Results/Standard_LOS_05_2024/alfa_1_434_server_02_06_24/1hz"
        # result_folder =  "../../../Data/Results/Standard_LOS_05_2024/alfa_1_434/10hz"
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        taa.create_panda_dataframe()
        taa.boxplot_LOS_comp_time(save_fig=False)
        plt.show()


if __name__ == '__main__':
    # unittest.main()
    t = MyTestCase()
    upfs = t.test_UPF_detail()



