import os
import shutil

if __name__ == "__main__":
    result_folder = "Data/Results/Standard_LOS_05_2024/alfa_1_434"
    trajectory_folder = "Data/Simulations"
    generated_tests_folder = "generated_tests"
    if generated_tests_folder in os.listdir("./"):
        shutil.rmtree("./" + generated_tests_folder)
    os.mkdir("./"+generated_tests_folder)


    methods = ["losupf|resample_factor=0.1|sigma_uwb_factor=2.0",
               "nodriftupf|resample_factor=0.1|sigma_uwb_factor=2.0",
               # "losupf|resample_factor=0.1|sigma_uwb_factor=1.0",
               # "losupf|resample_factor=0.5|sigma_uwb_factor=2.0",
               # "NLS|horizon=10",  # "NLS|horizon=100",
               # "algebraic|horizon=10",
               "algebraic|horizon=100"
               # "QCQP|horizon=10", 
               "QCQP|horizon=100"
                ]
    dvs = [0.1, 0.01, 0.001]
    sigma_dw_factor = 0.1
    d_uwbs = [1.0, 0.1, 0.01]
    uwb_rates = [1.0, 10.0]

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


    file_content_end =  "\tTAS = MRC.TwoAgentSystem(trajectory_folder=trajectory_folder, result_folder=result_folder)\n"
    file_content_end += "\tTAS.uwb_rate = uwb_rate\n"
    file_content_end += "\tTAS.debug_bool = False\n"
    file_content_end += "\tTAS.plot_bool = False\n"
    file_content_end += "\tTAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)\n"
    file_content_end += "\tTAS.set_ukf_properties(alpha = alpha, beta =  beta, kappa = kappa,\n"
    file_content_end += "                           n_azimuth = n_azimuth, n_altitude=n_altitude, n_heading=n_heading)\n"
    file_content_end += "\tTAS.run_simulations(methods=methods, redo_bool=True)\n"

    for dv in dvs:
        for duwb in d_uwbs:
            for uwb_rate in uwb_rates:
                file_content_middle = "\tsigma_dv = " + str(dv) + "\n"
                file_content_middle += "\tsigma_dw = " + str(sigma_dw_factor * dv) + "\n"
                file_content_middle += "\tsigma_uwb = " + str(duwb) + "\n"
                file_content_middle += "\tuwb_rate = " + str(1./uwb_rate) + "\n"

                file_name = "Test_RPE_2_agents_Sigma_dv_"+str(dv).replace(".", "c")+"_Sigma_uwb_"+str(duwb).replace(".", "c")+"_"+str(int(uwb_rate))+"hz.py"
                if file_name in os.listdir("./generated_tests"):
                    os.remove("./generated_tests/"+file_name)
                with open("./generated_tests/"+file_name, "w") as f:
                    f.write(file_content_start + file_content_middle + file_content_end)
                    f.close()
