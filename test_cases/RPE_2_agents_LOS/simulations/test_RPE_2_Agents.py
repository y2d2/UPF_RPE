import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import unittest
import Code.Simulation.MultiRobotClass as MRC
import Code.Analysis.TwoAgentAnalysis as TAA



class MyTestCase(unittest.TestCase):
    def test_generate_trajectories(self):
        test = "test"
        # Create a trajectory
        folder_name = 'robot_trajectories'
        MRS = MRC.MultiRobotSimulation()
        MRS.set_simulation_parameters(max_v=1, max_w=0.05, slowrate_v=0.1, slowrate_w=0.005,
                                      simulation_time_step=0.05, simulation_time=5*60,
                                      number_of_drones=2, max_range=25, range_origin_bool=True,
                                      trajectory_folder_name=folder_name, reset_trajectories=True)
        MRS.create_trajectories(50)

    def test_generate_sim_data(self):
        folder_name = 'robot_trajectories'
        MRS = MRC.MultiRobotSimulation(trajectory_folder_name=folder_name)
        sigma_vs = [0.1, 0.01, 0.001]
        sigma_ds = [1, 0.1, 0.01]
        sigma_v_long = []
        sigma_w_long = []
        sigma_d_long = []
        for sigma_v in sigma_vs:
            for sigma_d in sigma_ds:
                sigma_v_long.append(sigma_v)
                sigma_w_long.append(sigma_v*0.1)
                sigma_d_long.append(sigma_d)

        MRS.run_simulations(sigma_v_long, sigma_w_long, sigma_d_long)
        plt.show()

    def test_simulation(self):
        test = "test"
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
        sigma_dv = 0.1
        sigma_dw = 0.1 * sigma_dv
        sigma_uwb = 0.1

        # for i in range(4):
        #     mrss  = MRC.MultiRobotSingleSimulation(folder = "robot_trajectories/"+test_na_5_na_8_nh_8+"/sim_"+str(i))
        #     mrss.delete_sim(sigma_dv, sigma_dw, sigma_uwb)

        TAS = MRC.TwoAgentSystem(trajectory_folder="robot_trajectories/" + test, result_folder="Results/" + test+ "/new_method_test")
        TAS.debug_bool = False
        TAS.plot_bool = True
        TAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)
        TAS.set_ukf_properties(alpha, beta, kappa, n_azimuth, n_altitude, n_heading)
        TAS.run_simulations(methods=["upf", "losupf", "nodriftupf", "NLS", "algebraic"])

        plt.show()
        # TODO Launch mnassive NLOS simulation on Ares
        # TODO Work on the presentation of results.
        if TAS.plot_bool:
            for agent in TAS.agents:
                TAS.agents[agent]["upf"].upf_connected_agent_logger.plot_self(TAS.los_state)
                TAS.agents[agent]["algebraic"].logger.plot_self()
                TAS.agents["drone_0"]["NLS"].nls_logger.plot_self()

            plt.show()

    def test_run_all_simulations(self):
        trajectory_folder = "robot_trajectories/test"
        result_folder = "Results/test/NLOS_test"
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        # Parameters
        alpha = 1
        kappa = -1.
        beta = 2.
        n_azimuth = 4
        n_altitude = 3
        n_heading = 4

        sigma_vs = [0.1, 0.01]
        sigma_ds = [1, 0.1]
        for sigma_v in sigma_vs:
            sigma_w = 0.1* sigma_v
            for sigma_d in sigma_ds:
                print("sigma_v: ", sigma_v, "sigma_w: ", sigma_w, "sigma_d: ", sigma_d)
                TAS = MRC.TwoAgentSystem(trajectory_folder=trajectory_folder, result_folder=result_folder)
                TAS.debug_bool = False
                TAS.plot_bool = False
                TAS.set_uncertainties(sigma_v, sigma_w, sigma_d)
                TAS.set_ukf_properties(alpha=alpha, beta=beta, kappa= kappa,
                                       n_azimuth=n_azimuth, n_altitude=n_altitude, n_heading = n_heading)
                TAS.run_simulations(methods=["upf", "losupf"], nlos_function=TAS.nlos_man.nlos_2, redo_bool=True)
        # plt.show()
        # plt.close("all")

    def test_analysis_LOS_simulation(self):
        result_folder = "Results/Standard_LOS/alfa_1_434"
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        taa.create_panda_dataframe()
        taa.boxplot_LOS_comp(save_fig=False)
        plt.show()

    def test_time_analysis(self):
        result_folder = "Results/Standard_LOS/alfa_1_434"
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        taa.create_panda_dataframe()
        taa.boxplot_LOS_comp_time(save_fig=False)
        plt.show()

    def test_NLOS_analysis(self):
        result_folder = "Results/test/NLOS_test"

        # result_folder = "Results/Standard_LOS/alfa_1"
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        # taa.delete_data()
        taa.create_panda_dataframe()
        # taa.boxplot_var()
        # taa.boxplot_LOS_comp()
        taa.boxplot_NLOS_comp()
        plt.show()


    def test_exp_analysis(self):
        result_folder = "ros_tests/Real_Exp_Results"
        taa = TAA.TwoAgentAnalysis(result_folders=result_folder)
        taa.delete_data()
        taa.create_panda_dataframe()
        taa.boxplot_LOS_comp(save_fig=False)
        plt.show()




if __name__ == '__main__':
    unittest.main()
