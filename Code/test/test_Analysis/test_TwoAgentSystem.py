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
        TAS.run_simulations(methods=methods, redo_bool=False )

    def test_UPF_detail(self):
        upfs = []
        for sim in range(11):
            upfs.append(pkl.load(open("save_data/sim_"+str(sim)+"/drone_0_losupf.pkl", "rb")))
            upfs.append(pkl.load(open("save_data/sim_"+str(sim)+"/drone_1_losupf.pkl", "rb")))
        return upfs
        # plt.show()

    def test_analysis_freq_simulation(self):
        result_folder = "Results/test"
        taa = TAA.TwoAgentAnalysis(result_folder=result_folder)
        taa.create_panda_dataframe()
        taa.boxplot_freq_comp(save_fig=False)
        plt.show()

    def test_analysis_LOS_simulation(self):
        result_folder = "../../../Data/Results/Standard_LOS_06_2024"
        # result_folder = "../../../Data/Results/Broken"
        # result_folder = "./Results/test/1hz"

        methods_order = ["losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                            "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                            "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         # "NLS|horizon=10",
                         # "algebraic|horizon=10",
                         "algebraic|frequency=1.0|horizon=10",
                            "algebraic|frequency=10.0|horizon=100",
                         "algebraic|frequency=1.0|horizon=100",
                            "algebraic|frequency=10.0|horizon=1000",
                        # "QCQP|horizon=10",
                      "QCQP|frequency=1.0|horizon=10",
                         "QCQP|frequency=10.0|horizon=100",
                         "QCQP|frequency=1.0|horizon=100",
                         "QCQP|frequency=10.0|horizon=1000"]

        methods_color = {"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                        "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         # "NLS|horizon=10": "tab:red",
                         # "algebraic|horizon=10": "tab:green",
                         "algebraic|frequency=1.0|horizon=10": "tab:orange",
                            "algebraic|frequency=10.0|horizon=100": "tab:orange",
                         "algebraic|frequency=1.0|horizon=100": "tab:brown",
                         "algebraic|frequency=10.0|horizon=1000": "tab:brown",
                         # "QCQP|horizon=10": "tab:purple",
                         "QCQP|frequency=1.0|horizon=10": "tab:blue",
                         "QCQP|frequency=10.0|horizon=100":  "tab:blue",
                         "QCQP|frequency=1.0|horizon=100": "tab:cyan",
                         "QCQP|frequency=10.0|horizon=1000": "tab:cyan"}

        methods_legend = {"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, proposed",
                          # "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Proposed, ours (10hz)",
                          "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction",
                          # "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction (10hz)",
                        # "NLS|horizon=10": "NLS_10",
                         # "algebraic|horizon=10": "Algebraic_10",
                          "algebraic|frequency=1.0|horizon=10": "Algebraic (10s)",
                          "algebraic|frequency=10.0|horizon=1000": "Algebraic (100s)",
                         # "QCQP|horizon=10": "QCQP_10",
                          "QCQP|frequency=1.0|horizon=10": "QCQP (10s)",
                          "QCQP|frequency=1.0|horizon=100": "QCQP (100s)"}

        taa = TAA.TwoAgentAnalysis(result_folder=result_folder)
        # taa.delete_data()
        # taa.create_panda_dataframe()
        taa.boxplot_LOS_comp(sigma_uwb=[0.1, 1.0], sigma_v=[0.1, 0.01], frequencies=[1.0],
                             methods_order = methods_order, methods_color= methods_color,
                             methods_legend=methods_legend, start_time_index=1000, save_fig=False)
        plt.show()

    def test_time_analysis(self):
        result_folder =  "../../../Data/Results/Standard_LOS_05_2024/alfa_1_434_server_02_06_24/1hz"
        # result_folder =  "../../../Data/Results/Standard_LOS_05_2024/alfa_1_434/10hz"
        taa = TAA.TwoAgentAnalysis(result_folder=result_folder)
        taa.create_panda_dataframe()
        taa.boxplot_LOS_comp_time(save_fig=False)
        plt.show()


if __name__ == '__main__':
    # unittest.main()
    t = MyTestCase()
    upfs = t.test_UPF_detail()



