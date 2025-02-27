
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


    def test_TAS_RPE(self, sigma_dv = 0.1, sigma_uwb = 0.1):
        test = "small_trajectories"
        result_folder = "Results/" + test
        trajectory_folder = "../../../Data/small_trajectories"
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        # Parameters
        alpha = 1
        kappa = -1.
        beta = 2.
        n_azimuth = 4
        n_altitude = 3
        n_heading = 4
        sigma_dw = sigma_dv

        # for i in range(4):
        #     mrss  = MRC.MultiRobotSingleSimulation(folder = "robot_trajectories/"+test_na_5_na_8_nh_8+"/sim_"+str(i))
        #     mrss.delete_sim(sigma_dv, sigma_dw, sigma_uwb)

        TAS = MRC.TwoAgentSystem(trajectory_folder=trajectory_folder,
                                 result_folder=result_folder)

        TAS.debug_bool = False
        TAS.plot_bool = False
        TAS.save_folder = ("./save_data_test")
        TAS.save_bool = False
        TAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)
        TAS.set_ukf_properties(alpha, beta, kappa, n_azimuth, n_altitude, n_heading)
        # TAS.run_simulations(methods=["losupf", "nodriftupf", "algebraic", "NLS", "QCQP"], redo_bool=True)
        methods = [
                    "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                     "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                    "NLS|frequency=1.0|horizon=10.",
                   #  "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "algebraic|frequency=1.0|horizon=10",
                   # "algebraic|frequency=10.0|horizon=100",
                   # "QCQP|frequency=1.0|horizon=10",
                   # "QCQP|frequency=10.0|horizon=100"
                   ]
        TAS.run_simulations(methods=methods, redo_bool=False)

    def test_sig_dv_0c1_duwb_0c1(self):
        self.test_TAS_RPE(sigma_dv=0.1, sigma_uwb=0.1)

    def test_sig_dv_0c1_duwb_1(self):
        self.test_TAS_RPE(sigma_dv=0.1, sigma_uwb=1.0)

    def test_sig_dv_0c01_duwb_0c1(self):
        self.test_TAS_RPE(sigma_dv=0.01, sigma_uwb=0.1)

    def test_sig_dv_0c01_duwb_1(self):
        self.test_TAS_RPE(sigma_dv=0.01, sigma_uwb=1.0)

    def test_time_analysis_new(self):
        result_folder = [
                        "./Results/simulations_1hz",
                        # "../../../Data/Results/Sim_LOS_06_2024/1_sim",
                        # "../../../Data/Results/Sim_LOS_06_2024/final_methods_RPE_paper",
                        #  "../../../test_cases/RPE_2_agents_LOS/Experiments/Experiments/LOS_exp/Results/experiments_paper/Experiments",
                         ]

        # result_folder = "./Results/test_new_system"

        # result_folder = ("../../../Data/Results/test_files")
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)

        upf_sim = {"Method": "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                   "Variables": {
                       "Type": ["simulation"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.01, 0.1],
                       "Sigma_uwb": [0.1, 1.],
                       # "Sigma_dw": [],
                       "Frequency": [10.0],
                   },
                   "Color": "red",
                   "Legend": "UPF sim, $\sigma_{uwb} = 0.1$, $\sigma_{dv} = 0.1$",
                   }

        upf_sim_2 = {"Method": "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "Variables": {
                       "Type": ["simulation"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.01, 0.1],
                       "Sigma_uwb": [0.1, 1.],
                       # "Sigma_dw": [],
                       "Frequency": [10.0],
                   },
                   "Color": "blue",
                   "Legend": "UPF sim, $\sigma_{uwb} = 1$, $\sigma_{dv} = 0.1$",
                   }
        #
        # upf_exp = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
        #            "Variables": {
        #                "Type": ["experiment"],
        #                "Variable": ["error_x_relative", "error_h_relative"],
        #                "Sigma_dv": [0.08],
        #                "Sigma_uwb": [0.25],
        #                # "Sigma_dw": [],
        #                "Frequency": [10.0],
        #            },
        #            "Color": "tab:green",
        #            "Legend": "UPF exp",
        #            }

        # nodriftupf_exp = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
        #                   "Variables": {
        #                       "Type": ["simulation"],
        #                       "Variable": ["error_x_relative", "error_h_relative"],
        #                       "Sigma_dv": [0.01, 0.1],
        #                       "Sigma_uwb": [0.1, 1.],
        #                       # "Sigma_dw": [],
        #                       "Frequency": [10.0],
        #                   },
        #                   "Color": "tab:red",
        #                   "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state",
        #                   }
        alg_exp = {"Method": "algebraic|frequency=1.0|horizon=100",
                   "Variables": {
                       "Type": ["simulation"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.01, 0.1],
                       "Sigma_uwb": [0.1, 1.],
                       # "Sigma_dw": [],
                       "Frequency": [1.0],
                   },
                   "Color": "tab:orange",
                   "Legend": "Algebraic",
                   "Style" : "-",
                   }
        qcqp_exp = {"Method": "QCQP|frequency=10.0|horizon=1000",
                    "Variables": {
                        "Type": ["simulation"],
                        "Variable": ["error_x_relative", "error_h_relative"],
                        "Sigma_dv": [0.01, 0.1],
                        "Sigma_uwb": [0.1, 1.],
                        # "Sigma_dw": [],
                        "Frequency": [10.0],
                    },
                    "Color": "tab:blue",
                    "Legend": "QCQP",
                    }
        nls_sim = {"Method": "NLS|frequency=1.0|horizon=10",
                   "Variables": {
                       "Type": ["simulation"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.1],
                       "Sigma_uwb": [0.1],
                       # "Sigma_dw": [],
                       "Frequency": [1.],
                   },
                   "Color": "salmon",
                   "Legend": "NLS sim, $\sigma_{uwb} = 0.1$, $\sigma_{dv} = 0.1$",
                   }

        nls_sim2 = {"Method": "NLS|frequency=1.0|horizon=10",
                   "Variables": {
                       "Type": ["simulation"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.1],
                       "Sigma_uwb": [1.],
                       # "Sigma_dw": [],
                       "Frequency": [1.],
                   },
                   "Color": "lightblue",
                   "Legend": "NLS sim, $\sigma_{uwb} = 1$, $\sigma_{dv} = 0.1$",
                   }

        nls_exp = {"Method": "NLS|frequency=1.0|horizon=10",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": ["error_x_relative", "error_h_relative"],
                       "Sigma_dv": [0.08],
                       "Sigma_uwb": [0.35],
                       # "Sigma_dw": [],
                       "Frequency": [1.],
                   },
                   "Color": "limegreen",
                   "Legend": "NLS exp",
                   }

        methods_order = [ upf_sim,
                         # alg_exp,
                         #  qcqp_exp,
                         #  nls_exp,
                         ]

        df, methods_names, methods_colors,  methods_styles, methods_legends = taa.filter_methods_new(methods_order)
        g = taa.lineplot(df, methods_names, methods_colors, methods_styles=methods_styles, methods_legends=methods_legends)
        # plt.legend(loc="upper left")
        plt.show()



if __name__ == '__main__':
    unittest.main()
