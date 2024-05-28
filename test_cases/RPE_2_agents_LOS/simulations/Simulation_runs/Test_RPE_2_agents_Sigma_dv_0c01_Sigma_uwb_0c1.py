import os
os.environ["OPENBLAS_NUM_THREADS"]= "2"

import matplotlib
matplotlib.use('Qt5Agg')

import Code.Simulation.MultiRobotClass as MRC

if __name__ == "__main__":
    result_folder = "../Results/Standard_LOS/alfa_1_434"
    trajectory_folder = "../robot_trajectories/Standard"

    alpha = 1
    kappa = -1.
    beta = 2.
    n_azimuth = 4
    n_altitude = 3
    n_heading = 4

    sigma_dv = 0.01
    sigma_dw_factor = 0.1
    sigma_uwb = 0.1

    sigma_dw = sigma_dw_factor * sigma_dv
    TAS = MRC.TwoAgentSystem(trajectory_folder=trajectory_folder, result_folder=result_folder)
    TAS.debug_bool = False
    TAS.plot_bool = False
    TAS.set_uncertainties(sigma_dv, sigma_dw, sigma_uwb)
    TAS.set_ukf_properties(alpha = alpha, beta =  beta, kappa = kappa,
                           n_azimuth = n_azimuth, n_altitude=n_altitude, n_heading=n_heading)
    TAS.run_simulations(methods=["upf", "losupf", "nodriftupf"], redo_bool=True)