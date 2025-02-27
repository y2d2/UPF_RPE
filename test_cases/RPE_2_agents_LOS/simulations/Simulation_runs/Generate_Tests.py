import math
import os
import shutil

if __name__ == "__main__":
    result_folder = "Data/Results/Sim_1hz_2024"
    trajectory_folder = "Data/Simulations"
    generated_tests_folder = "generated_tests"
    if generated_tests_folder not in os.listdir("./"):
        os.mkdir("./"+generated_tests_folder)
    # if generated_tests_folder in os.listdir("./"):
    #     shutil.rmtree("./" + generated_tests_folder)
    # os.mkdir("./"+generated_tests_folder)

    redo_bool = False
    methods = [
                "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particels=0",
                # "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                # "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                "NLS|frequency=0.1|horizon=10",
                # "NLS|frequency=1.0|horizon=10",
                # "NLS|frequency=1.0|horizon=100"
                # "NLS|frequency=1.0|horizon=100",
                # "NLS|frequency=10.0|horizon=100",
                # "NLS|frequency=10.0|horizon=1000"
                # "algebraic|frequency=10.0|horizon=100",
                "algebraic|frequency=1.0|horizon=10",
                # "algebraic|frequency=10.0|horizon=1000",
                # "algebraic|frequency=1.0|horizon=100"
                # "QCQP|frequency=10.0|horizon=100",
                "QCQP|frequency=1.0|horizon=10",
                # "QCQP|frequency=10.0|horizon=1000",
                # "QCQP|frequency=1.0|horizon=100"
                ]
    dvs = [0.1, 0.01]
    sigma_dw_factor = 1.0
    d_uwbs = [1.0, 0.1]
    # uwb_rates = [1.0, 10.0]

    file_content_start = "import os \n"
    file_content_start += "os.environ[\"OPENBLAS_NUM_THREADS\"]= \"2\"\n"
    file_content_start += "import Code.Simulation.MultiRobotClass as MRC\n"
    file_content_start += "\n"
    file_content_start += "if __name__ == \"__main__\":\n"
    file_content_start += "\tresult_folder = \"" + result_folder + "\" \n"
    file_content_start += "\ttrajectory_folder = \"" + trajectory_folder + "\" \n"
    file_content_start += "\t\n"
    file_content_start += "\talpha = 1\n"
    file_content_start += "\tkappa = -1.\n"
    file_content_start += "\tbeta = 2.\n"
    file_content_start += "\tn_azimuth = 4\n"
    file_content_start += "\tn_altitude = 3\n"
    file_content_start += "\tn_heading = 4\n"
    file_content_start += "\t\n"
    file_content_start += "\tmethods = []\n"
    for method in methods:
        file_content_start += "\tmethods.append(\"" + method + "\")\n"




    file_content_start += "\tdvs = "+str(dvs) + "\n"
    file_content_start += "\tsigma_dw_factor =  "+str(sigma_dw_factor) + "\n"
    file_content_start += "\td_uwbs = "+str(d_uwbs) + "\n"

    file_content_start += "\tfor dv in dvs:\n"
    file_content_start += "\t\tsigma_dv = dv \n"
    file_content_start += "\t\tsigma_dw = sigma_dw_factor * dv \n"
    file_content_start += "\t\tfor duwb in d_uwbs:\n"
    file_content_start += "\t\t\tsigma_uwb = duwb \n"
    file_content_start += "\t\t\tTAS = MRC.TwoAgentSystem(trajectory_folder=trajectory_folder, result_folder=result_folder)\n"
    file_content_start += "\t\t\tTAS.debug_bool = False\n"
    file_content_start += "\t\t\tTAS.plot_bool = False\n"
    file_content_start += "\t\t\tTAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)\n"
    file_content_start += "\t\t\tTAS.set_ukf_properties(alpha = alpha, beta =  beta, kappa = kappa,\n"
    file_content_start += "\t\t\t                       n_azimuth = n_azimuth, n_altitude=n_altitude, n_heading=n_heading)\n"

    parallel_processes = 50
    total_simulations = len(os.listdir(trajectory_folder))
    number_of_sim_per_process = int(math.ceil(total_simulations / parallel_processes))

    for i in range(parallel_processes):
        try:
            sim_list = os.listdir(trajectory_folder)[i*number_of_sim_per_process:(i+1)*number_of_sim_per_process]
        except IndexError:
            sim_list = os.listdir(trajectory_folder)[i*number_of_sim_per_process:]
        file_content_middle = "\t\t\tTAS.run_simulations(methods=methods, redo_bool="+str(redo_bool)+", sim_list="+str(sim_list)+")\n"
        file_name = "Test_RPE_2_agents_"+str(i)+".py"
        if file_name in os.listdir("./generated_tests"):
            os.remove("./generated_tests/"+file_name)
        with open("./generated_tests/"+file_name, "w") as f:
            f.write(file_content_start + file_content_middle)
            f.close()
