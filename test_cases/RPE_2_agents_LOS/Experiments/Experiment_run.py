import os

from Code.UtilityCode.Measurement import Measurement, create_experiment, create_experimental_data


def test_run_LOS_exp(self):
    # From the data sig_v =0.1, sig_w=0.1 and sig_uwb = 0.35 (dependable on the set... ) are the best values.
    sig_v = 0.08
    sig_w = 0.12
    sig_uwb = 0.15

    results_folder =  "Results_exp"
    data_folder = "corrections3/"

    os.mkdir(results_folder)


    experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)

    methods = ["losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particels=0",
               "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
               # "algebraic|frequency=1.0|horizon=10",
               "algebraic|frequency=10.0|horizon=100",
               # # "algebraic|frequency=10.0|horizon=1000",
               "QCQP|frequency=10.0|horizon=100",
               # # "QCQP|frequency=10.0|horizon=1000",
               "NLS|frequency=1.0|horizon=10",
               ]

    tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
    tas.debug_bool = True
    tas.plot_bool = False
    tas.run_experiment(methods=methods, redo_bool=False, experiment_data=experiment_data)


if __name__ == '__main__':
    test_run_LOS_exp()