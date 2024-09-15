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
import seaborn as sns

class MyTestCase(unittest.TestCase):

    def set_test_case(self):
        self.exp_folder = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/experiments/"
        self.rosbag = self.exp_folder+"exp4"
        self.name = self.rosbag.split("/")[-1]
        self.sampled_pkl = "LOS_exp1_sampled.pkl"

        self.measurment_folder = "Experiments/Measurements/Unob_exp"

        self.uwb_topic = "/yd_uwb/dev_0x7603_0x683a"
        self.tb2_topic = "/vicon/tb2/tb2"
        self.tb3_topic = "/vicon/tb3/tb3"
        self.tb2_odom_topic = "/tb2/odom"
        self.tb3_odom_topic = "/tb3/odom"
        self.tb2 = Turtlebot4("tb2")
        self.tb3 = Turtlebot4("tb3")

    def test_vio_sanity(self):
        rosbag = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/experiments/exp1"
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
        pickle_file = ("./exp1_raw.pkl")
        measurement = Measurement()
        measurement.load_raw_data(pickle_file)
        measurement.sample(10)
        measurement.plot_sampled()
        measurement.print_sampled_lengths()
        measurement.get_VIO_error(plot=True)
        # measurement.tb2.plot_vio_error()
        # measurement.tb3.plot_vio_error()
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
        # mesList = measurement.split_data([4805, 4900, 4900+2200, 4825+2280, + 4825+2280+2200])
        mesList = measurement.split_data([900, 900+600,5060, 5060 + 3000])
        for mes in mesList:
            mes.plot_sampled()
            # mes.get_VIO_error(plot=True)
            # mes.get_uwb_distances()



        mesList[0].name = "exp1_unobservable"
        mesList[0].save_folder = "New_measurements/"
        mesList[0].save_sampled_data()
        mesList[1].name = "exp2_unobservable"
        mesList[1].save_folder = "New_measurements/"
        mesList[1].save_sampled_data()

        mesList[3].name = "exp1_los_sampled.pkl"
        mesList[3].save_folder = "New_measurements/LOS/"
        mesList[3].save_sampled_data()

    def test_new_measurments(self):
        name = "exp1"
        pikle_file = name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.sample(10)
        mesList = measurement.split_data([ 5060, 8060])

        mesList[1].name = "exp1_los"
        mesList[1].save_folder = "New_measurements/LOS/"
        # mesList[1].save_sampled_data()

        mesList = measurement.split_data([9558-3000, 9558])
        mesList[1].name = "exp5_los"
        mesList[1].save_folder = "New_measurements/LOS/"
        mesList[1].save_sampled_data()


        name = "exp3"
        pikle_file = name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.sample(10)
        mesList = measurement.split_data([3000,6000])

        mesList[0].name = "exp2_los"
        mesList[0].save_folder = "New_measurements/LOS/"
        # mesList[0].save_sampled_data()

        mesList[1].name = "exp3_los"
        mesList[1].save_folder = "New_measurements/LOS/"
        # mesList[1].save_sampled_data()

        name = "exp4"
        pikle_file = name + "_raw.pkl"
        measurement = Measurement()
        measurement.load_raw_data(pikle_file)
        measurement.sample(10)
        mesList = measurement.split_data([3000, 3500, 6500])

        mesList[0].name = "exp4_los"
        mesList[0].save_folder = "New_measurements/LOS/"
        # mesList[0].save_sampled_data()

        mesList[2].name = "exp5_los"
        mesList[2].save_folder = "New_measurements/LOS/"
        # mesList[2].save_sampled_data()


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
        fig, ax = plt.subplots(5, 1)
        for i in range(1, 6):
            sampled_pkl = "New_measurements/LOS/exp"+str(i)+"_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.get_uwb_distances()
            # measurement.correct_orb_transformation()
            # measurement.get_rpe_transformation()
            measurement.uwb.plot_real(ax=ax[i-1])
            ax[i-1].set_title("exp"+str(i))
            ax[i-1].set_ylabel("Distance [m]")
        ax[-1].set_xlabel("Time [s]")
        ax[0].legend()

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
        w_errors = np.empty((0,3))
        v_errors = np.empty((0,3))
        v_cor_errors = np.empty((0,3))
        for i in range(1, 6):
            sampled_pkl = "New_measurements/LOS/exp"+str(i)+"_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.tb2.vio_frame.outlier_rejection(max_a=0.5)
            measurement.tb3.vio_frame.outlier_rejection(max_a = 0.5)
            measurement.get_VIO_error(plot=True)
            w_errors = np.concatenate((w_errors, measurement.tb2.vio_w_error))
            v_errors = np.concatenate((v_errors, measurement.tb2.vio_v_error))
            v_cor_errors = np.concatenate((v_cor_errors, measurement.tb2.vio_v_cor_error))
            w_errors = np.concatenate((w_errors, measurement.tb3.vio_w_error))
            v_errors = np.concatenate((v_errors, measurement.tb3.vio_v_error))
            v_cor_errors = np.concatenate((v_cor_errors, measurement.tb3.vio_v_cor_error))


        # Make np.array(w_erros) 1 dimentional

        w_mean = np.mean(w_errors, axis=0)
        w_std = np.std(w_errors, axis=0)
        v_mean = np.mean(v_errors, axis=0)
        v_std = np.std(v_errors, axis=0)
        v_cor_mean = np.mean(v_cor_errors, axis=0)
        v_cor_std = np.std(v_cor_errors, axis=0)
        print("W error mean: ", w_mean, " std: ", w_std)
        print("V error mean: ", v_mean, " std: ", v_std)
        print("V cor error mean: ", v_cor_mean, " std: ", v_cor_std)

        plt.show()


    def test_set_vio_correction(self):
        self.set_test_case()
        for i in range(1, 6):
            sampled_pkl = "New_measurements/LOS/exp" + str(i) + "_los_sampled.pkl"
            measurement = Measurement()
            measurement.load_sampled_data(sampled_pkl)
            measurement.tb2.vio_frame.outlier_rejection(max_a=0.5)
            measurement.tb3.vio_frame.outlier_rejection(max_a=0.5)
            measurement.tb2.vio_frame.sampled_v = measurement.tb2.vio_frame.v_cor
            measurement.tb3.vio_frame.sampled_v = measurement.tb3.vio_frame.v_cor
            measurement.name = "exp" + str(i) + "_los"
            measurement.save_folder = "New_measurements/corrections2/"
            measurement.save_sampled_data()

    def test_new_robot_population(self):
        # self.set_test_case()
        sampled_pkl = "./Measurements_correction/exp2_los_sampled.pkl"
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
        ax = tb3.plot_slam_error(annotation="TB1", color ="blue")
        ax = tb2.plot_slam_error(annotation="TB2", ax=ax, color="red")
        for axs in ax:
            axs.legend()
            axs.grid()
        plt.show()
        return tb2, tb3
if __name__ == '__main__':
    unittest.main()
