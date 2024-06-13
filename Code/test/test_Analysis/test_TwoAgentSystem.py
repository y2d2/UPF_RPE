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



    def test_analysis_LOS_simulation(self):
        result_folder = "../../../Data/Results/Sim_LOS_06_2024/final_methods_RPE_paper"
        # result_folder = "../../../Data/Results/Broken"
        # result_folder = "./Results/test/1hz"

        methods_order = [
                        # "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
#                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        # "NLS|horizon=10",
                        # "algebraic|horizon=10",
                        # "algebraic|frequency=1.0|horizon=10",
                        # "algebraic|frequency=10.0|horizon=100",
#                          "algebraic|frequency=1.0|horizon=100",
                        "algebraic|frequency=10.0|horizon=1000",
                        # "QCQP|horizon=10",
                        # "QCQP|frequency=1.0|horizon=10",
                        # "QCQP|frequency=10.0|horizon=100",
#                         "QCQP|frequency=1.0|horizon=100",
                        "QCQP|frequency=10.0|horizon=1000",
                        "NLS|frequency=1.0|horizon=10",
        ]

        methods_color = {"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                        "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         # "NLS|horizon=10": "tab:red",
                         # "algebraic|horizon=10": "tab:green",
                         "algebraic|frequency=1.0|horizon=10": "tab:orange",
                            "algebraic|frequency=10.0|horizon=100": "tab:orange",
                         "algebraic|frequency=1.0|horizon=100": "tab:orange",
                         "algebraic|frequency=10.0|horizon=1000": "tab:orange",
                         # "QCQP|horizon=10": "tab:purple",
                         "QCQP|frequency=1.0|horizon=10": "tab:blue",
                         "QCQP|frequency=10.0|horizon=100":  "tab:blue",
                         "QCQP|frequency=1.0|horizon=100": "tab:blue",
                         "QCQP|frequency=10.0|horizon=1000": "tab:blue",
                         "NLS|frequency=1.0|horizon=10": "tab:purple",
                         }

        methods_legend = {
                        # "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, proposed",
                          "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, proposed",
                          # "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction",
                          "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction",
                        # "NLS|horizon=10": "NLS_10",
                         # "algebraic|horizon=10": "Algebraic_10",
                         #  "algebraic|frequency=1.0|horizon=10": "Algebraic (10s)",
                          "algebraic|frequency=10.0|horizon=1000": "Algebraic",
                         # "QCQP|horizon=10": "QCQP_10",
                         #  "QCQP|frequency=10.0|horizon=100": "QCQP (10s)",
                          "QCQP|frequency=10.0|horizon=1000": "QCQP",
                          "NLS|frequency=1.0|horizon=10": "NLS",
        }

        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        # taa.delete_data()
        # taa.create_panda_dataframe()
        # taa.percent_to_load = 5
        taa.boxplots(sigma_uwb=[0.1, 1.0], sigma_v=[0.1, 0.01], frequencies=[1.0, 10.0],
                             methods_order = methods_order, methods_color= methods_color,
                            variables=["error_x_relative", "error_h_relative", "calculation_time"],
                             methods_legend=methods_legend, start_time=10, save_fig=False)
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



