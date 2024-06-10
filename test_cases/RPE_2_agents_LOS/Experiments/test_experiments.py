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

    def set_test_case(self):
        self.exp_folder = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/Experiments/LOS_exp/"
        self.rosbag = self.exp_folder+"exp4"
        self.name = self.rosbag.split("/")[-1]
        self.sampled_pkl = "LOS_exp1_sampled.pkl"

        self.measurment_folder = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/CodeBase/yd_MA_Inter_Robot_Range/tests/test_cases/RPE_2_agents/ros_tests/Measurements/"

        self.uwb_topic = "/yd_uwb/dev_0x7603_0x683a"
        self.tb2_topic = "/vicon/tb2/tb2"
        self.tb3_topic = "/vicon/tb3/tb3"
        self.tb2_odom_topic = "/tb2/odom"
        self.tb3_odom_topic = "/tb3/odom"
        self.tb2 = Turtlebot4("tb2")
        self.tb3 = Turtlebot4("tb3")

    def test_vio_sanity(self):
        rosbag = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/Experiments/LOS_exp/Exp2/tb3_2023_10_05T15_18_21"
        ns = "/tb3"
        imu_topic = "/oakd/imu/data"
        image_l_topic = "/oakd/left/image_rect/compressed"
        image_r_topic = "/oakd/right/image_rect/compressed"

        image_l_Dt = []
        image_l_ts = []
        image_r_Dt = []
        image_r_ts = []
        imu_Dt = []
        imu_ts = []

        prev_imu_t = None
        prev_image_l_t = None
        prev_image_r_t = None

        with rb2.Reader(rosbag) as ros2_reader:
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                # print(timestamp, connection.topic)

                if ns+imu_topic == connection.topic:


                    data = deserialize_cdr(rawdata, connection.msgtype)
                    imu_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
                    if prev_imu_t is None:
                        prev_imu_t = imu_t
                    imu_Dt.append(imu_t - prev_imu_t)
                    imu_ts.append(imu_t)
                    prev_imu_t = imu_t

                if ns+image_l_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    image_l_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
                    if prev_image_l_t is None:
                        prev_image_l_t = image_l_t
                    image_l_Dt.append(image_l_t - prev_image_l_t)
                    prev_image_l_t = image_l_t
                    image_l_ts.append(image_l_t)

                if ns+image_r_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    image_r_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
                    if prev_image_r_t is None:
                        prev_image_r_t = image_r_t
                    image_r_Dt.append(image_r_t - prev_image_r_t)
                    prev_image_r_t = image_r_t
                    image_r_ts.append(image_r_t)

        plt.figure()
        plt.plot(imu_ts, imu_Dt, label="imu")
        plt.plot(image_l_ts, image_l_Dt, label="image_l")
        plt.plot(image_r_ts, image_r_Dt, label="image_r")
        plt.legend()
        plt.show()

    def test_split_rosbag(self):
        rosbag = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/Experiments/LOS_exp/Exp3/tb3_2023_10_05T16_35_24"
        ns = "/tb3"
        imu_topic = "/oakd/imu/data"
        image_l_topic = "/oakd/left/image_rect/compressed"
        image_r_topic = "/oakd/right/image_rect/compressed"

        # writer = rb2.SequentialWriter()
        # rb2.Writer.

        image_l_Dt = []
        image_r_Dt = []
        imu_Dt = []

        prev_imu_t = None
        prev_image_l_t = None
        prev_image_r_t = None

        with rb2.Reader(rosbag) as ros2_reader:
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                # print(timestamp, connection.topic)

                if ns + imu_topic == connection.topic:

                    data = deserialize_cdr(rawdata, connection.msgtype)
                    imu_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9

                if ns + image_l_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    image_l_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9


                if ns + image_r_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    image_r_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9



    def test_read_bag(self):
        self.set_test_case()
        measurement = Measurement(self.rosbag)
        measurement.read_bag()
        measurement.save_raw_data()
        print(len(measurement.tb3.vio_frame.t))
        print(len(measurement.tb2.vio_frame.t))
        print(len(measurement.tb3.vicon_frame.t))
        print(len(measurement.tb2.vicon_frame.t))
        print(len(measurement.uwb.t))
        plt.figure()
        measurement.tb3.plot_trajectory(plt)
        measurement.tb2.plot_trajectory(plt)
        plt.show()

    def test_read_raw_data_pkl(self):
        self.set_test_case()
        pikle_file = self.name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.tb2.plot_trajectory(plt)
        print(len(measurement.tb3.vio_frame.t))
        print(len(measurement.tb2.vio_frame.t))
        print(len(measurement.tb3.vicon_frame.t))
        print(len(measurement.tb2.vicon_frame.t))
        print(len(measurement.uwb.t))
        plt.show()

    def test_raw_data(self):
        pickle_file = "../../RPE_2_agents/ros_tests/exp4_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pickle_file)
        measurement.sample(10)
        measurement.plot_sampled()
        measurement.print_sampled_lengths()

        measurement.get_VIO_error(plot=True)
        measurement.tb2.plot_vio_error()
        measurement.tb3.plot_vio_error()
        measurement.get_uwb_distances()
        measurement.uwb.plot_real()
        measurement.uwb.plot_indices()
        plt.show()
        return measurement



    def test_split_data_exp1(self):
        # self.set_test_case()
        name = "exp1"
        pikle_file = name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.sample(10)
        mesList = measurement.split_data([4805, 4900, 4900+2200, 4825+2280, + 4825+2280+2200])
        # mesList[0].name = "exp1_unobservable"
        # mesList[0].save_sampled_data()
        # mesList[2].name = "exp1_sec1_los"
        # mesList[2].save_sampled_data()
        # mesList[4].name = "exp1_sec2_los"
        # mesList[4].save_sampled_data()

        mesList = measurement.split_data([4900])
        # mesList[1].name = "exp1_los"
        # mesList[1].save_sampled_data()

        mesList = measurement.split_data([900, 900+600])
        mesList[1].name = "exp1_unobservable"
        mesList[1].save_sampled_data()

    def test_split_data_exp3(self):
        name = "exp3"
        pikle_file = name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.sample(10)
        mesList = measurement.split_data([2200])  # , 4900, 4900 + 2200, 4825 + 2280, + 4825 + 2280 + 2200])
        mesList[0].name = "exp3_sec1_los"
        mesList[0].save_sampled_data()
        mesList = measurement.split_data([1200, 1200+2200])  # , 4900, 4900 + 2200, 4825 + 2280, + 4825 + 2280 + 2200])
        mesList[1].name = "exp3_sec2_los"
        mesList[1].save_sampled_data()
        mesList = measurement.split_data([2400])
        mesList[1].name = "exp3_nlos"
        mesList[1].save_sampled_data()

    def test_split_data_exp4(self):
        name = "exp4"
        pikle_file = name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.sample(10)
        mesList = measurement.split_data([500,500+2200]) #, 4900, 4900 + 2200, 4825 + 2280, + 4825 + 2280 + 2200])
        mesList[1].name = "exp4_sec1_los"
        # mesList[1].save_sampled_data()
        mesList = measurement.split_data([2700-300, 2700-300+2400 ])
        mesList[1].name = "exp4_nlos_b"
        mesList[1].save_sampled_data()


    def test_add_errors_to_exp4(self):
        pikle_file = "../../RPE_2_agents/ros_tests/exp4_nlos_a_sampled.pkl"
        measurement = Measurement()
        measurement.load_sampled_data(pikle_file)
        measurement.get_uwb_distances()
        measurement.uwb.plot_real()
        sigma_uwb = 0.25
        measurement.uwb.change_data(1200, 10., sigma_uwb)
        measurement.get_uwb_distances()
        measurement.uwb.plot_real()
        measurement.name = "exp4_nlos_a_changed_10"
        measurement.save_sampled_data()
        plt.show()


    def test_check_sampling(self):
        self.set_test_case()
        sampled_pkl = "Measurements/exp2_los_sampled.pkl"

        pikle_file = self.name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_sampled_data(sampled_pkl)
        # measurement.sample(10)
        # measurement.save_sampled_data()
        measurement.uwb.plot_sampled()
        measurement.tb2.vio_frame.plot_sampled()
        measurement.tb3.vio_frame.plot_sampled()
        measurement.tb2.vicon_frame.plot_sampled()
        measurement.tb3.vicon_frame.plot_sampled()
        plt.show()

    def test_uwb_optimisation_T(self):
        self.set_test_case()
        measurement = Measurement()
        measurement.load_sampled_data(self.sampled_pkl)
        measurement.get_uwb_distances()
        measurement.uwb.plot_real()
        measurement.uwb.plot_indices()
        # measurement.optimise_uwb_T()
        plt.show()

    def test_uwb_Transforms(self):
        # self.set_test_case()
        for i in range(1, 6):
            sampled_pkl = "Measurements/exp"+str(i)+"_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.get_uwb_distances()
            # measurement.correct_orb_transformation()
            # measurement.get_rpe_transformation()
            measurement.uwb.plot_real()
        plt.show()

    def test_vio_rejection(self):
        # TODO: make rejection strategy for VIO. (Maybe can help.)
        self.set_test_case()
        sampled_pkl = "Measurements/exp1_los_sampled.pkl"
        measurement = Measurement()
        measurement.load_sampled_data(sampled_pkl)
        measurement.tb2.vio_frame.outlier_rejection()

    def test_resave_the_sample_data(self):
        self.set_test_case()
        for i in range(1, 6):
            sampled_pkl = "Measurements/exp"+str(i)+"_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.save_folder = "Meas_new/"
            measurement.name = "exp"+str(i)+"_los"
            measurement.save_sampled_data()
        # sampled_pkl = "Measurements/exp4_los_sampled.pkl"
        # measurement = Measurement()
        # measurement.load_sampled_data(sampled_pkl)
        # measurement.save_sampled_data()

    def test_vio_error(self):
        self.set_test_case()
        for i in range(1, 6):
            sampled_pkl = "Measurements/exp"+str(i)+"_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.tb2.vio_frame.outlier_rejection(max_a=0.5)
            measurement.tb3.vio_frame.outlier_rejection(max_a = 0.5)
            measurement.get_VIO_error(plot=True)
            measurement.tb2.plot_vio_error()
            measurement.tb3.plot_vio_error()
        plt.show()
        # sampled_pkl = "Measurements/exp1_los_sampled.pkl"
        # # sampled_pkl = "Meas_new/exp1_sec1_los_sampled.pkl"
        # measurement = Measurement()
        # measurement.load_sampled_data(sampled_pkl)
        # measurement.tb2.vio_frame.outlier_rejection(max_a = 0.5)
        # measurement.tb3.vio_frame.outlier_rejection(max_a = 0.5)
        #
        # measurement.get_VIO_error(plot=True)
        # measurement.tb2.plot_vio_error()
        # measurement.tb3.plot_vio_error()
        # plt.show()

    def test_set_vio_correction(self):
        self.set_test_case()
        for i in range(1, 6):
            sampled_pkl = "Measurements/exp" + str(i) + "_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.tb2.vio_frame.outlier_rejection(max_a=0.5)
            measurement.tb3.vio_frame.outlier_rejection(max_a=0.5)
            measurement.tb2.vio_frame.sampled_v = measurement.tb2.vio_frame.v_cor
            measurement.tb3.vio_frame.sampled_v = measurement.tb3.vio_frame.v_cor
            measurement.name = "exp" + str(i) + "_los"
            measurement.save_folder = "Measurements_correction/"
            measurement.save_sampled_data()




    def test_new_robot_population(self):
        # self.set_test_case()
        sampled_pkl = "./Experiments/LOS_exp/Measurements/exp1_sec1_los_sampled.pkl"
        measurement = Measurement()
        measurement.load_sampled_data(sampled_pkl)
        sample_freq=measurement.sample_frequency

        # measurement.correct_orb_transformation()

        sig_v = 0.1
        sig_w = 0.1
        sig_uwb = 0.2
        sig_d = sig_v / sample_freq
        sig_phi = sig_w / sample_freq
        Q_vio = np.diag([sig_d ** 2, sig_d ** 2, sig_d ** 2, sig_phi ** 2])

        DT_vio_tb2 = measurement.tb2.vio_frame.get_relative_motion_in_T()
        DT_vio_tb3 = measurement.tb3.vio_frame.get_relative_motion_in_T()
        T_vicon_tb2 = measurement.tb2.vicon_frame.sampled_T
        T_vicon_tb3 = measurement.tb3.vicon_frame.sampled_T

        tb2 = NewRobot()
        tb2.from_experimental_data(T_vicon_tb2, DT_vio_tb2, Q_vio, sample_freq)
        tb3 = NewRobot()
        tb3.from_experimental_data(T_vicon_tb3, DT_vio_tb3, Q_vio, sample_freq)

        ax = plt.axes(projection="3d")
        tb2.set_plotting_settings(color="r")
        tb2.plot_real_position(ax)
        tb2.plot_slam_position(ax,  linestyle=":", alpha=0.6 )
        tb3.set_plotting_settings(color="b")
        tb3.plot_real_position(ax)
        tb3.plot_slam_position(ax, linestyle=":", alpha=0.6)
        plt.legend()
        plt.show()
        return tb2, tb3

    # @DeprecationWarning
    # def create_experimental_data(self, data_folder, sig_v, sig_w, sig_uwb):
    #     experiments=[]
    #     measurements = []
    #     # check wether data_folder is a file or a folder
    #
    #     if os.path.isfile(data_folder):
    #         list_of_files = [data_folder]
    #     else:
    #         list_of_files = os.listdir(data_folder)
    #     for sampled_data in list_of_files:
    #         name = sampled_data.split(".")[-2].split("/")[-1]
    #         measurement = Measurement()
    #         measurement.load_sampled_data(sampled_data)
    #         sample_freq = measurement.sample_frequency
    #
    #         #
    #         sig_d = sig_v / sample_freq
    #         sig_phi = sig_w / sample_freq
    #         Q_vio = np.diag([sig_d ** 2, sig_d ** 2, sig_d ** 2, sig_phi ** 2])
    #
    #         # measurement.get_uwb_distances()
    #         uwb = measurement.uwb.sampled_d
    #         uwb_los = measurement.get_uwb_LOS(sig_uwb)
    #         DT_vio_tb2 = measurement.tb2.vio_frame.get_relative_motion_in_T()
    #         DT_vio_tb3 = measurement.tb3.vio_frame.get_relative_motion_in_T()
    #         T_vicon_tb2 = measurement.tb2.vicon_frame.sampled_T
    #         T_vicon_tb3 = measurement.tb3.vicon_frame.sampled_T
    #
    #         experiment_data = {}
    #         experiment_data["name"] = name
    #         experiment_data["sample_freq"] = sample_freq
    #         experiment_data["drones"] = {}
    #         experiment_data["drones"]["drone_0"] = {"DT_slam": DT_vio_tb2, "T_real": T_vicon_tb2, "Q_slam": Q_vio}
    #         experiment_data["drones"]["drone_1"] = {"DT_slam": DT_vio_tb3, "T_real": T_vicon_tb3, "Q_slam": Q_vio}
    #         experiment_data["uwb"] = uwb
    #         experiment_data["los_state"] = uwb_los
    #
    #
    #         measurements.append(measurement)
    #         # experiment_data["eps_d"] = np.abs(measurement.uwb.real_d - measurement.uwb.sampled_d)
    #
    #         experiments.append(experiment_data)
    #     return experiments, measurements

    # @DeprecationWarning
    # def create_experiment(self, results_folder, sig_v, sig_w, sig_uwb, alpha = 1., kappa = -1., beta = 2. , n_azimuth = 4, n_altitude = 3, n_heading = 4):
    #
    #
    #     tas = MRC.TwoAgentSystem(trajectory_folder="./", result_folder=results_folder)
    #     tas.debug_bool = True
    #     tas.plot_bool = True
    #     tas.set_ukf_properties(kappa=kappa, alpha=alpha, beta=beta, n_azimuth=n_azimuth, n_altitude=n_altitude,
    #                            n_heading=n_heading)
    #     tas.set_uncertainties(sig_v, sig_w, sig_uwb)
    #     return tas

    def test_single_exp(self):
        sig_v = 0.10
        sig_w = 0.03
        sig_uwb = 0.25


        result_folder = "Real_Exp_test"
        data_file = "Experiments/Exp3_SemiNLOS/Measurements/exp3_sec1_los_sampled.pkl"
        experiment_data, _ = create_experimental_data(data_file, sig_v, sig_w, sig_uwb)
        tas = create_experiment(result_folder, sig_v, sig_w, sig_uwb)
        # tas.run_experiment(methods=[ "upf"], redo_bool=True, experiment_data=experiment_data)

        tas.run_experiment(methods=[ "NLS", "algebraic", "upf", "losupf", "nodriftupf", "upfnaive"], redo_bool=False, experiment_data=experiment_data)

        return tas
        # taa = TAA.TwoAgentAnalysis(result_folder=result_folder)
        # taa.delete_data()
        # taa.create_panda_dataframe()
        # taa.single_settings_boxplot(save_fig=False)
        # plt.show()

    def test_run_LOS_exp(self):
        # From the data sig_v =0.1, sig_w=0.1 and sig_uwb = 0.35 (dependable on the set... ) are the best values.
        sig_v = 0.08
        sig_w = 0.12
        sig_uwb = 0.25

        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/experiment_outlier_rejection_3/1hz"
        data_folder = "Measurements_correction/"

        experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)

        methods = ["losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   "algebraic|frequency=1.0|horizon=10",
                   "algebraic|frequency=10.0|horizon=100",
                   "algebraic|frequency=10.0|horizon=1000",
                   "QCQP|frequency=10.0|horizon=100",
                   "QCQP|frequency=10.0|horizon=1000"
                   ]
        methods = ["losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                           "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                           "algebraic|frequency=1.0|horizon=10",
                           "algebraic|frequency=1.0|horizon=100",
                           "QCQP|frequency=1.0|horizon=10",
                           "QCQP|frequency=1.0|horizon=100"
                           ]

        methods = [
                   # "algebraic|frequency=1.0|horizon=10",
                   # "algebraic|frequency=1.0|horizon=100",
                   # "algebraic|frequency=10.0|horizon=100",
                   # "algebraic|frequency=10.0|horizon=1000",
                    "QCQP|frequency=1.0|horizon=10",
                    "QCQP|frequency=1.0|horizon=100",
                    "QCQP|frequency=10.0|horizon=100",
                    "QCQP|frequency=10.0|horizon=1000"
        ]

        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = False
        tas.run_experiment(methods=methods, redo_bool=True, experiment_data=experiment_data)
        plt.show()
        # return tas, measurements

    def test_rename_experiments(self):
        main_folder = "./Experiments/LOS_exp/"
        load_dir = main_folder + "Results/experiment_outlier_rejection_3"
        save_dir =  main_folder + "Results/experiment_outlier_rejection_4"

        n_files = len(os.listdir(load_dir))
        n_file = 0
        for file in os.listdir(load_dir):
            n_file += 1
            print(str(int(n_file/n_files*100)) + "%: " +  file)
            if os.path.isfile(load_dir + "/"+file) and not os.path.exists(save_dir + "/exp_" + file):
                os.rename(load_dir + "/" + file, save_dir + "/exp_"+ file)
                with open(save_dir + "/exp_" + file, "rb") as f:
                    data = pkl.load(f)
                f.close()
                with open(save_dir + "/exp_" + file, "wb") as f:
                    data["parameters"]["type"] = "experiment"
                    pkl.dump(data, f)
                f.close()


    def test_plot_LOS_error_time(self):
        main_folder = "./Experiments/LOS_exp/"
        results_folder = main_folder + "Results/new_experiment/"
        tas = MRC.TwoAgentSystem(trajectory_folder="./", result_folder=results_folder)
        data = tas.get_data_from_file(results_folder + "number_of_agents_2_sigma_dv_0c15_sigma_dw_0c05_sigma_uwb_0c25_alpha_1c0_kappa_neg1c0_beta_2c0.pkl")
        exp = 2
        for i in range(2):
            agent_name = "drone_"+str(i)

            exp_name = "exp"+str(exp)+"_los_sampled"
            upf_x_error_0 = data[exp_name]["nodriftupf"][agent_name]["error_x_relative"]
            NLS_x_error_0 = data[exp_name]["NLS"][agent_name]["error_x_relative"]
            losupf_x_error_0 = data[exp_name]["losupf"][agent_name]["error_x_relative"]
            slam_x_error_0 = data[exp_name]["slam"][agent_name]["error_x_relative"]

            upf_h_error_0 = data[exp_name]["nodriftupf"][agent_name]["error_h_relative"]
            NLS_h_error_0 = data[exp_name]["NLS"][agent_name]["error_h_relative"]
            losupf_h_error_0 = data[exp_name]["losupf"][agent_name]["error_h_relative"]
            slam_h_error_0 = data[exp_name]["slam"][agent_name]["error_h_relative"]

            _, ax = plt.subplots(2, 1)
            ax[0].plot(NLS_x_error_0, label="NLS [7]", color="tab:blue", linewidth=2)
            ax[0].plot(losupf_x_error_0, label="upf ours", color="tab:green", linewidth=2)
            ax[0].plot(upf_x_error_0, label="No drift UPF", color="tab:red", linewidth=2)
            ax[0].plot(slam_x_error_0, label="slam", color="tab:orange", linewidth=2)
            ax[0].set_xlabel("Time [s]", fontsize=12)
            ax[0].set_ylabel(r"$\epsilon_{\hat{p}^t}$ [m]", fontsize=12)
            ax[0].grid()
            ax[0].legend()

            ax[1].plot(NLS_h_error_0, label="NLS [7]", color="tab:blue", linewidth=2)
            ax[1].plot(losupf_h_error_0, label=r"upf ours", color="tab:green", linewidth=2)
            ax[1].plot(upf_h_error_0, label="No drift UPF", color="tab:red", linewidth=2)
            ax[1].plot(slam_h_error_0, label="slam", color="tab:orange", linewidth=2)
            ax[1].set_xlabel("Time [s]", fontsize=12)
            ax[1].set_ylabel(r"$\epsilon_{\hat{\theta}^t}$ [(rad))]", fontsize=12)
            ax[1].grid()
            ax[1].set_ylim([0., 0.5])

            ax[1].legend()



        plt.show()


    def test_exp_analysis(self):
        result_folders = ["./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/1hz",
                          "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/10hz"]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        methods_order = [
                         "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                        "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",

                        # "algebraic|frequency=1.0|horizon=10",
                        # "algebraic|frequency=10.0|horizon=100",
                         "algebraic|frequency=1.0|horizon=100",
                        "algebraic|frequency=10.0|horizon=1000",

                        # "QCQP|frequency=1.0|horizon=10",
                        # "QCQP|frequency=10.0|horizon=100",
                        "QCQP|frequency=1.0|horizon=100",
                        "QCQP|frequency=10.0|horizon=1000"
                        ]

        methods_color = {"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         # "NLS|horizon=10": "tab:red",
                         # "algebraic|horizon=10": "tab:green",
                         # "algebraic|frequency=1.0|horizon=10": "tab:orange",
                         "algebraic|frequency=1.0|horizon=100": "tab:orange",
                         "algebraic|frequency=10.0|horizon=1000": "tab:orange",
                         # "QCQP|horizon=10": "tab:purple",
                         # "QCQP|frequency=1.0|horizon=10": "tab:blue",
                         "QCQP|frequency=1.0|horizon=100": "tab:blue",
                         "QCQP|frequency=10.0|horizon=1000": "tab:blue"}

        methods_legend = {"losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Proposed, ours",
                          "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction",
                          # "NLS|horizon=10": "NLS_10",
                          # "algebraic|horizon=10": "Algebraic_10",
                          # "algebraic|frequency=1.0|horizon=10": "Algebraic 10s",
                          "algebraic|frequency=1.0|horizon=100": "Algebraic",
                          # "QCQP|horizon=10": "QCQP_10",
                          # "QCQP|frequency=1.0|horizon=10": "QCQP",
                          "QCQP|frequency=1.0|horizon=100": "QCQP"}


        # taa.delete_data()
        # taa.create_panda_dataframe()
        taa.boxplots(sigma_uwb=[0.25], sigma_v=[0.08], frequencies=[1.0, 10.0],
                             methods_order=methods_order, methods_color=methods_color,
                             methods_legend=methods_legend, start_time=10, save_fig=False)
        plt.show()

    def test_exp_time_analysis(self):
        # result_folder = "./Experiments/LOS_exp/Results/new_nls_correct_init_test/"
        result_folders = [
                            # "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/1hz",
                            "./Experiments/LOS_exp/Results/experiment_outlier_rejection_3/10hz"
                            ]
        taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
        methods_order = [
                        # "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         # "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                         # "NLS|horizon=10",
                         # "algebraic|horizon=10",
                         # "algebraic|frequency=10.0|horizon=100",
                         # "algebraic|frequency=1.0|horizon=100",
                         "algebraic|frequency=10.0|horizon=1000",
                         # "QCQP|horizon=10",
                         # "QCQP|frequency=10.0|horizon=100",
                         # "QCQP|frequency=1.0|horizon=100",
                         "QCQP|frequency=10.0|horizon=1000"]

        methods_color = {
                        "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                        "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:green",
                         "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "tab:red",
                         # "NLS|horizon=10": "tab:red",
                         # "algebraic|horizon=10": "tab:green",
                         "algebraic|frequency=1.0|horizon=100": "tab:orange",
                         "algebraic|frequency=10.0|horizon=1000": "tab:orange",
                         # "QCQP|horizon=10": "tab:purple",
                         "QCQP|frequency=1.0|horizon=100": "tab:blue",
                         "QCQP|frequency=10.0|horizon=1000": "tab:blue"}

        methods_legend = {
                            "losupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Proposed, ours",
                            "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Proposed, ours",
                          "nodriftupf|frequency=1.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction",
                          "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0": "Ours, without drift correction",
                          # "NLS|horizon=10": "NLS_10",
                          # "algebraic|horizon=10": "Algebraic_10",
                          "algebraic|frequency=1.0|horizon=10": "Algebraic 10s",
                          "algebraic|frequency=1.0|horizon=100": "Algebraic",
                          "algebraic|frequency=10.0|horizon=1000": "Algebraic",
                          # "QCQP|horizon=10": "QCQP_10",
                          "QCQP|frequency=1.0|horizon=10": "QCQP",
                          "QCQP|frequency=10.0|horizon=1000": "QCQP",
                          "QCQP|frequency=1.0|horizon=100": "QCQP"}
        # taa.delete_data()
        taa.create_panda_dataframe()
        taa.time_analysis(sigma_uwb=0.25, sigma_v=0.08, frequency = 10.0, start_time=10,
                          methods_order=methods_order, methods_color=methods_color, methods_legend=methods_legend, save_fig=False)
        # taa.boxplot_LOS_comp_time(save_fig=False)
        # taa.calculation_time(save_fig=False)
        plt.show()



    def test_run_NLOS_exp(self):
        sig_v = 0.15 # issue with VIO being disturbed by the closets initial had to increase this to make it work.
        sig_w = 0.05
        sig_uwb = 0.25 # tried 0.3, 0.25 and 0.2

        main_folder = "./Experiments/NLOS_exp/"
        results_folder = main_folder + "Results_changed/"
        data_folder = main_folder + "Measurements/exp4_nlos_a_sampled.pkl"

        experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool= True
        # tas.plot_bool = True

        # tas.run_experiment(methods=["upf", "losupf"], redo_bool=True, experiment_data=experiment_data)
        # tas.run_experiment(methods=["NLS"], redo_bool=True, experiment_data=experiment_data)
        tas.run_experiment(methods=["upf", "NLS", "losupf"], redo_bool=False, experiment_data=experiment_data)
        plt.show()
        return tas, measurements

    def test_nlos_detection_plot(self):
        tas, measurements = self.test_run_NLOS_exp()

        sig_uwb = 0.25
        sig2_los = measurements[0].get_uwb_LOS(2*sig_uwb)
        sig3_los = measurements[0].get_uwb_LOS(3 * sig_uwb)
        sigma2_los_t = [i/10 for i in range(len(sig2_los)) if (i%10==0 and sig2_los[i]==0)]
        sigma3_los_t = [i/10 for i in range(len(sig3_los)) if (i%10==0 and sig3_los[i]==0)]

        drone_0_los = tas.data["exp4_nlos_a_sampled"]["upf"]["drone_0"]["los_state"]
        drone_0_los_t = [i for i in range(len(drone_0_los)) if (drone_0_los[i]==0)]
        drone_0_los_x = [0 for i in range(len(drone_0_los_t))]
        #
        drone_1_los = tas.data["exp4_nlos_a_sampled"]["upf"]["drone_1"]["los_state"]
        drone_1_los_t = [i for i in range(len(drone_1_los)) if (drone_1_los[i]==0)]
        drone_1_los_x = [1 for i in range(len(drone_1_los_t))]

        _, ax = plt.subplots(2,1)
        # plt.figure()
        for sigma2_los in sigma2_los_t:
            ax[1].axvline(x=sigma2_los, color="tab:blue", linestyle="--", linewidth=3)
        for sigma3_los in sigma3_los_t:
            ax[1].axvline(x=sigma3_los, color="tab:red", linestyle="--", linewidth=3)

        plt.text(0.25,0.92, "agent 1 NLOS", fontsize=12)
        plt.text(0.25, 0, "agent 0 NLOS", fontsize=12)
        ax[1].plot(drone_0_los_t, drone_0_los_x, ".k")
        ax[1].plot(drone_1_los_t, drone_1_los_x, ".k")
        ax[1].plot(0,0, color = "tab:blue", linestyle="--", label=r"$\epsilon_{d} > 2\sigma_d$", linewidth=3)
        ax[1].plot(0, 0, color="tab:red", linestyle="--", label=r"$\epsilon_{d} > 3\sigma_d$",linewidth=3)
        ax[1].set_xlabel("Time [s]")
        plt.yticks([])

        # ax[1].set_ylabel("Robot")
        ax[1].legend(loc='center left', fontsize=12)


        uwb_1 = measurements[0].uwb
        uwb_1.plot_real(factor=10, ax =ax[0])
        ax[0].legend(loc='upper left', fontsize=12)

        ax[0].set_xlim([-11, 259])
        ax[1].set_xlim([-11, 259])
        print(drone_0_los_t)
        print(drone_1_los_t)
        print(sigma2_los_t)
        print(sigma3_los_t)



        plt.show()
        return tas

    def test_plot_NLOS_error(self):
        tas, measurements = self.test_run_NLOS_exp()
        agent_nr = 0
        agent_name = "drone_"+str(agent_nr)

        exp_name = "exp4_nlos_a_sampled"
        upf_x_error_0 = tas.data[exp_name]["upf"][agent_name]["error_x_relative"]
        NLS_x_error_0 = tas.data[exp_name]["NLS"][agent_name]["error_x_relative"]
        losupf_x_error_0 = tas.data[exp_name]["losupf"][agent_name]["error_x_relative"]
        slam_x_error_0 = tas.data[exp_name]["slam"][agent_name]["error_x_relative"]

        upf_h_error_0 = tas.data[exp_name]["upf"][agent_name]["error_h_relative"]
        NLS_h_error_0 = tas.data[exp_name]["NLS"][agent_name]["error_h_relative"]
        losupf_h_error_0 = tas.data[exp_name]["losupf"][agent_name]["error_h_relative"]
        slam_h_error_0 = tas.data[exp_name]["slam"][agent_name]["error_h_relative"]


        agent_nr = 1
        agent_name = "drone_"+str(agent_nr)
        upf_x_error_1 = tas.data[exp_name]["upf"][agent_name]["error_x_relative"]
        NLS_x_error_1 = tas.data[exp_name]["NLS"][agent_name]["error_x_relative"]
        losupf_x_error_1 = tas.data[exp_name]["losupf"][agent_name]["error_x_relative"]
        slam_x_error_1 = tas.data[exp_name]["slam"][agent_name]["error_x_relative"]

        _, ax  = plt.subplots(1, 2)
        ax[0].plot(NLS_x_error_0, label="NLS [7]", color="tab:blue", linewidth=3)
        ax[0].plot(losupf_x_error_0, label="", color="tab:red", linewidth=3)
        ax[0].plot(upf_x_error_0, label="UPF (ours)", color="tab:green", linewidth=3)
        # ax[0].plot(slam_x_error_0, label="slam")
        ax[0].set_xlabel("Time [s]", fontsize=12)
        ax[0].set_ylabel(r"$\epsilon_{\hat{p}^t}$ [m]", fontsize=12)
        ax[0].grid()
        # ax[0].legend()


        ax[1].plot(NLS_h_error_0, label="NLS [7]", color="tab:blue", linewidth=3)
        ax[1].plot(losupf_h_error_0, label=r"UPF $\tilde{w}$ $s_{LOS}$ (ours)",color="tab:red", linewidth=3)
        ax[1].plot(upf_h_error_0, label="UPF (ours)", color="tab:green",  linewidth=3)
        ax[1].set_xlabel("Time [s]", fontsize=12)
        ax[1].set_ylabel(r"$\epsilon_{\hat{\theta}^t}$ [(rad))]", fontsize=12)
        ax[1].grid()
        ax[1].set_ylim([0., 0.5])
        # ax[1].plot(slam_h_error_0, label="slam")
        ax[1].legend()

        plt.show()

        # ax[1].plot(upf_x_error_1, label="upf")
        # ax[1].plot(NLS_x_error_1, label="NLS")
        # ax[1].plot(losupf_x_error_1, label="losupf")
        # # ax[1].plot(slam_x_error_1, label="slam")
        # ax[1].legend()



    def test_plot_augmented_NLOS_Error_Graph(self):
        tas, measurements = self.test_run_NLOS_exp()
        agent_nr = 0
        agent_name = "drone_" + str(agent_nr)
        labels = ["Original", "$d_{NLOS}=2m$", "$d_{NLOS}=10m$"]

        exp = ["exp4_nlos_a_sampled","exp4_nlos_a_changed_2_sampled" ,"exp4_nlos_a_changed_10_sampled"]
        line_style = ["-", "--", ":"]
        alpha = [0.3, 0.5, 1]

        plt.figure()

        plt.plot(0, 0, color="tab:blue", linestyle="-", alpha=1,  linewidth=3, label="NLS [7]")
        plt.plot(0, 0, color="tab:red", linestyle="-", alpha=1,  linewidth=3, label= r"UPF $\tilde{w}$ $s_{LOS}$ (ours)")
        plt.plot(0, 0, color="tab:green", linestyle="-", alpha=1,  linewidth=3, label="UPF (ours)")
        for i, exp_name in enumerate(exp):
            upf_error = tas.data[exp_name]["upf"][agent_name]["error_x_relative"]
            NLS_error = tas.data[exp_name]["NLS"][agent_name]["error_x_relative"]
            losupf_error= tas.data[exp_name]["losupf"][agent_name]["error_x_relative"]

            plt.plot(0, 0, color="k", linestyle=line_style[i], alpha=alpha[i], label=labels[i], linewidth=3)
            plt.plot(NLS_error, color="tab:blue", linestyle=line_style[i], alpha=alpha[i], linewidth=3)
            plt.plot(losupf_error, color="tab:red", linestyle=line_style[i], alpha=alpha[i], linewidth=3)
            plt.plot(upf_error, color="tab:green", linestyle=line_style[i], alpha=alpha[i], linewidth=3)
        plt.ylim([0, 2.])
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel(r"$\epsilon_{\hat{p}^t}$ [m]", fontsize=12)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid()
        plt.show()

    def test_unobservable_motion(self):
        sig_v = 0.15
        sig_w = 0.05
        sig_uwb = 0.3

        main_folder = "./Experiments/Unob_exp/Measurements/"
        results_folder = main_folder + "Results/"
        data_folder = main_folder + "Measurements/"
        data_folder = main_folder + "exp1_unobservable_sampled.pkl"
        print(data_folder)
        experiment_data, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = False
        tas.plot_bool = False
        # tas.run_experiment(methods=["NLS", "algebraic", "upf", "losupf", "nodriftupf"], redo_bool=False, experiment_data=experiment_data)
        tas.run_experiment(methods=["losupf"], redo_bool=True, experiment_data=experiment_data)
        # upf_x_error = tas.data["exp1_unobservable_sampled"]["losupf"]["drone_1"]["error_x_relative"]
        # slam_x_error = tas.data["exp1_unobservable_sampled"]["slam"]["drone_1"]["error_x_relative"]
        ax = plt.figure().add_subplot(projection='3d')

        tas.agents["drone_0"]["losupf"].upf_connected_agent_logger.plot_poses(ax, color_ha="darkblue", color_ca="red",
                                                                           name_ha="$a_0$", name_ca="$a_1$")
        # ax = plt.figure().add_subplot(projection='3d')
        tas.agents["drone_1"]["losupf"].upf_connected_agent_logger.plot_poses(ax, color_ha="maroon", color_ca="dodgerblue",
                                                                           name_ha="$a_1$", name_ca="$a_0$")
        ax.plot(0,0, color="darkblue", label=r"$T_{\mathcal{W}, \mathcal{S}_0}$" )
        ax.plot(0, 0, color='red', alpha=1, linestyle="--",
                label="Active particles of $a_0$ for $T_{\mathcal{W}, \mathcal{S}_0}\hat{T}_{\mathcal{S}_0, \mathcal{S}_1}$")  # for estimation of "+ name)
        ax.plot(0, 0, color='red', alpha=0.1, linestyle=":",
                label="Killed particles of $a_0$ for $T_{\mathcal{W}, \mathcal{S}_0}\hat{T}_{\mathcal{S}_0, \mathcal{S}_1}$")  # for estimation of "+ name)
        ax.plot(0, 0, color="maroon", label=r"$T_{\mathcal{W}, \mathcal{S}_1}$")
        ax.plot(0, 0, color='dodgerblue', alpha=1, linestyle="--",
                label="Active particles of $a_1$ for $T_{\mathcal{W}, \mathcal{S}_1}\hat{T}_{\mathcal{S}_1, \mathcal{S}_0}$")  # for estimation of "+ name)
        ax.plot(0, 0, color='dodgerblue', alpha=0.2, linestyle=":",
                label="Killed particles of $a_1$ for $T_{\mathcal{W}, \mathcal{S}_1}\hat{T}_{\mathcal{S}_1, \mathcal{S}_0}$")  # for estimation of "+ name)


        # ax.legend(fontsize=10)

        # plt.figure()
        # tas.agents["drone_1"]["upf"].upf_connected_agent_logger
        plt.show()
        return tas, measurements

    def test_quickfix_legend_unobservable(self):
        plt.figure()
        plt.plot(0, 0, color="darkblue", label="Real pose $i$")
        plt.plot(0, 0, color='red', alpha=1, linestyle="--",
                label="Active particles for $j$ by $i$")  # for estimation of "+ name)
        plt.plot(0, 0, color='red', alpha=0.1, linestyle=":",
                label="Killed particles for $j$ by $i$")  # for estimation of "+ name)
        plt.plot(0, 0, color="maroon", label="Real pose $j$")
        plt.plot(0, 0, color='dodgerblue', alpha=1, linestyle="--",
                label="Active particles for $j$  by $j$")  # for estimation of "+ name)
        plt.plot(0, 0, color='dodgerblue', alpha=0.2, linestyle=":",
                label="Killed particles for $i$ by $j$")  # for estimation of "+ name)
        plt.plot(0, 0, color="black", linestyle="", marker="o", label="Start of a trajectory")
        plt.plot(0, 0, color="black", linestyle="", marker="x", label="End of a trajectory")
        plt.legend(fontsize=10)
        plt.show()

    def test_particle_generation(self):
        sig_v = 0.15
        sig_w = 0.03
        sig_uwb = 0.25

        result_folder = "Analysis/Particle_count"
        data_file = "Experiments/Exp3_SemiNLOS/Measurements/exp3_sec1_los_sampled.pkl"
        experiment_data, _ = create_experimental_data(data_file, sig_v, sig_w, sig_uwb)
        tas = create_experiment(result_folder, sig_v, sig_w, sig_uwb)
        # tas.run_experiment(methods=[ "upf"], redo_bool=True, experiment_data=experiment_data)

        tas.run_experiment(methods=[ "upf", "upfnaive"], redo_bool=False,
                           experiment_data=experiment_data)

        plt.figure(figsize=(6, 3))
        plt.plot(tas.data["exp3_sec1_los_sampled"]["upf"]["drone_0"]["number_of_particles"], label="NLOS UPF (ours)",
                 color="tab:orange", linewidth=3)
        plt.plot(tas.data["exp3_sec1_los_sampled"]["upfnaive"]["drone_0"]["number_of_particles"],
                 label="Naive sampling UPF", color="k", linewidth=3)
        plt.xlim([0, 60])
        plt.yscale("log")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("Number of particles", fontsize=12)
        plt.legend(fontsize=12)

        # plt.figure(figsize=(6, 3))
        # plt.plot(tas.data["exp3_sec1_los_sampled"]["upf"]["drone_0"]["calculation_time"], label="UPF (ours)",
        #          color="tab:green", linewidth=3)
        # plt.plot(tas.data["exp3_sec1_los_sampled"]["upfnaive"]["drone_0"]["calculation_time"],
        #          label="UPF naive sampling", color="tab:red", linewidth=3)
        # plt.xlim([0, 60])
        # plt.yticks( fontsize=12)
        # plt.xticks(fontsize=12)
        # plt.xlabel("Time [s]", fontsize=12)
        # plt.ylabel("Calculation time [s]", fontsize=12)
        # plt.legend(fontsize=12)

        plt.show()

        return tas

if __name__ == '__main__':
    t = MyTestCase()
    tas = t.test_particle_generation()

    # tas, exp = t.test_run_NLOS_exp()



    # plt.figure()
    # tas.agents["drone_0"]["upf"].upf_connected_agent_logger.
    # exp = t.create_experimental_data("./Experiments/LOS_exp/Measurements/")
    # tb2, tb3 = t.test_new_robot_population()

    # tas = t.test_run_exp()

