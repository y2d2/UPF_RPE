import unittest
import os

import rosbags.rosbag2 as rb2
import unittest
from rosbags.serde import deserialize_cdr

import Code.Simulation.MultiRobotClass as MRC
from Code.UtilityCode.turtlebot4 import Turtlebot4
import numpy as np

from Code.UtilityCode.Measurement import Measurement, create_experiment, create_experimental_data
from Code.Analysis import TwoAgentAnalysis as TAA

from Code.Simulation.RobotClass import NewRobot
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle as pkl

class MyTestCase(unittest.TestCase):
    def test_run_LOS_exp(self):
        # From the data sig_v =0.1, sig_w=0.1 and sig_uwb = 0.35 (dependable on the set... ) are the best values.
        sig_v = 0.08
        sig_w = 0.12
        sig_uwbs = [0.1, 0.25, 0.35, 0.5]

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/QCQP"
        data_folder = "Measurements_correction/"


        methods = [ "QCQP|frequency=1.0|horizon=10",
                   "QCQP|frequency=1.0|horizon=100"
                   ]
        for sig_uwb in sig_uwbs:
            experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
            tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
            tas.debug_bool = True
            tas.plot_bool = False
            tas.run_experiment(methods=methods, redo_bool=False, experiment_data=experiment_data)


if __name__ == '__main__':
    unittest.main()
