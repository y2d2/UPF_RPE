import os
os.environ["OPENBLAS_NUM_THREADS"]= "2"

import Code.Simulation.MultiRobotClass as MRC

if __name__ == "__main__":
    result_folder = "Data/Results/Standard_LOS_05_2024/alfa_1_434"
    trajectory_folder = ("Data/Simulations")

    alpha = 1
    kappa = -1.
    beta = 2.
    n_azimuth = 4
    n_altitude = 3
    n_heading = 4

    sigma_dv = 0.1
    sigma_dw_factor = 0.1
    sigma_dw = sigma_dw_factor * sigma_dv
    sigma_uwb = 0.1

    uwb_rate = 1.0
    methods = ["losupf|resample_factor=0.1|sigma_uwb_factor=2.0",
               "losupf|resample_factor=0.1|sigma_uwb_factor=1.0",
               "losupf|resample_factor=0.5|sigma_uwb_factor=2.0",
               "nodriftupf|resample_factor=0.1|sigma_uwb_factor=2.0",
               "NLS|horizon=10", #"NLS|horizon=100",
               "algebraic|horizon=10", "algebraic|horizon=100",
               "QCQP|horizon=10", "QCQP|horizon=100"]



    TAS = MRC.TwoAgentSystem(trajectory_folder=trajectory_folder, result_folder=result_folder)
    TAS.uwb_rate = uwb_rate
    TAS.debug_bool = False
    TAS.plot_bool = False
    TAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)
    TAS.set_ukf_properties(alpha = alpha, beta =  beta, kappa = kappa,
                           n_azimuth = n_azimuth, n_altitude=n_altitude, n_heading=n_heading)
    TAS.run_simulations(methods=methods, redo_bool=False)