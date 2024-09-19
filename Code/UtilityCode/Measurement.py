import os
import rosbags.rosbag2 as rb2
from rosbags.rosbag2 import WriterError
from rosbags.typesys import Stores, get_typestore
from rosbags.serde import deserialize_cdr
from Code.UtilityCode.turtlebot4 import Turtlebot4
from Code.Simulation.MultiRobotClass import TwoAgentSystem
import numpy as np
import pickle as pkl
# import Experiments

from Code.UtilityCode.UWB import UWB
from Code.UtilityCode.Transformation_Matrix_Fucntions import inv_transformation_matrix,transformation_matrix_from_rot_vect

from pathlib import Path
from rosbags.typesys import get_types_from_msg, register_types

def load_custom_messages():
    def guess_msgtype(path: Path) -> str:
        """Guess message type name from path."""
        name = path.relative_to(path.parents[2]).with_suffix('')
        if 'msg' not in name.parts:
            name = name.parent / 'msg' / name.name
        return str(name)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    add_types = {}
    folder = '../../../Data/Custom_ros_messages/'
    for pathstr in ['yd_uwb_msgs/msg/UWBRange.msg',
                    'yd_uwb_msgs/msg/UWBRangeStamp.msg',
                    'vicon_receiver/msg/Position.msg',]:
        msgpath = Path(folder + pathstr)
        msgdef = msgpath.read_text(encoding='utf-8')
        add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))
    return add_types

    #######################
    # CUSTOM MESSAGS
    ######################
    # custom_msg_path = Path('../../../Data/Custom_ros_messages/UWBRangeStamp.msg')
    # msg_def = custom_msg_path.read_text(encoding='utf-8')
    # register_types(get_types_from_msg(
    #     msg_def, 'yd_uwb_msgs/msg/UWBRangeStamp'))
    #
    # custom_msg_path = Path(
    #     '/home/yuri/Documents/PhD/ROS_WS/humble/src/yd_uwb/yd_uwb_msgs/msg/UWBRange.msg')
    # msg_def = custom_msg_path.read_text(encoding='utf-8')
    # register_types(get_types_from_msg(
    #     msg_def, 'yd_uwb_msgs/msg/UWBRange'))
    #
    # custom_msg_path = Path(
    #     '/home/yuri/Documents/PhD/ROS_WS/humble/src/ros2-vicon-receiver/vicon_receiver/msg/Position.msg')
    # msg_def = custom_msg_path.read_text(encoding='utf-8')
    # register_types(get_types_from_msg(
    #     msg_def, 'vicon_receiver/msg/Position'))
    #
    # add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))
    #
    # typestore.register(add_types)
    #
    # add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

    #####################################################################################



class Measurement:
    def __init__(self, rosbag=None, save_folder="./", name = None):
        if name is None and rosbag is not None:
            self.name = rosbag.split("/")[-1]
        else:
            self.name = name

        self.rosbag = rosbag
        self.save_folder = save_folder
        #
        # self.times = []
        # self.distances = []
        # self.T_tb2_vio = []
        # self.T_tb3_vio = []
        # self.vw_tb2_vio = []
        # self.vw_tb3_vio = []
        # self.T_tb2 = []
        # self.T_tb3 = []

        self.sample_frequency = None

        # Subclasses
        self.uwb = UWB("name")
        self.tb2 = Turtlebot4("tb2")
        self.tb3 = Turtlebot4("tb3")

        # ROS2 topics
        self.uwb_topic = "/yd_uwb/dev_0x7603_0x683a"
        self.tb2_topic = "/vicon/tb2/tb2"
        self.tb3_topic = "/vicon/tb3/tb3"
        self.tb2_odom_topic = "/tb2/odom"
        self.tb3_odom_topic = "/tb3/odom"

        self.RPE_T_tb2_tb3 = np.empty((0,4,4))
        self.RPE_T_tb3_tb2 = np.empty((0,4,4))

    #--------------------
    # ROS and ROSBAG Reading
    #____________________
    def set_topics(self, tb2_vicon, tb2_vio, tb3_vicon, tb3_vio, uwb):
        self.tb2_topic = tb2_vicon
        self.tb2_odom_topic = tb2_vio
        self.tb3_topic = tb3_vicon
        self.tb3_odom_topic = tb3_vio
        self.uwb_topic = uwb

    def read_bag(self, VIO_source="orb"):
        # VIO_source = "orb" or "specVIO"
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        typestore.register(load_custom_messages())

        with rb2.Reader(self.rosbag) as ros2_reader:
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)
            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                if self.tb2_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    self.tb2.update_vicon(data, timestamp)

                if self.tb2_odom_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    if VIO_source == "specVIO":
                        self.tb2.update_specVIO(data)
                    if VIO_source == "orb":
                        self.tb2.update_orb(data)

                if self.tb3_odom_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    if VIO_source == "specVIO":
                        self.tb3.update_specVIO(data)
                    if VIO_source == "orb":
                        self.tb3.update_orb(data)

                if self.tb3_topic == connection.topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    self.tb3.update_vicon(data, timestamp)

                if self.uwb_topic == connection.topic:
                    msg_data = deserialize_cdr(rawdata, connection.msgtype)
                    self.uwb.get_measuremend(msg_data)

    def trim_bag(self, name, begin_time, end_time):
        # VIO_source = "orb" or "specVIO"
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        typestore.register(load_custom_messages())
        t0 = None
        with rb2.Writer(self.save_folder + "/" + name) as ros2_writer:
            tb2_topic_con = ros2_writer.add_connection(self.tb2_topic, 'vicon_receiver/msg/Position', typestore=typestore)
            tb2_odom_con = ros2_writer.add_connection(self.tb2_odom_topic, 'nav_msgs/msg/Odometry', typestore=typestore)
            tb3_topic_con = ros2_writer.add_connection(self.tb3_topic, 'vicon_receiver/msg/Position', typestore=typestore)
            tb3_odom_con = ros2_writer.add_connection(self.tb3_odom_topic, 'nav_msgs/msg/Odometry', typestore=typestore)
            uwb_con = ros2_writer.add_connection(self.uwb_topic, 'yd_uwb_msgs/msg/UWBRange', typestore=typestore)

            with rb2.Reader(self.rosbag) as ros2_reader:

                ros2_conns = [x for x in ros2_reader.connections]
                ros2_messages = ros2_reader.messages(connections=ros2_conns)
                for m, msg in enumerate(ros2_messages):
                    (connection, timestamp, rawdata) = msg
                    if t0 is None:
                        t0 = timestamp
                        print(t0)
                    if timestamp >= t0 + begin_time*1e9 and timestamp <= t0 + end_time*1e9:
                        print(timestamp)
                        if self.tb2_topic == connection.topic:
                            ros2_writer.write(tb2_topic_con, timestamp, rawdata)
                        if self.tb2_odom_topic == connection.topic:
                            ros2_writer.write(tb2_odom_con, timestamp, rawdata)
                        if self.tb3_topic == connection.topic:
                            ros2_writer.write(tb3_topic_con, timestamp, rawdata)
                        if self.tb3_odom_topic == connection.topic:
                            ros2_writer.write(tb3_odom_con, timestamp, rawdata)
                        if self.uwb_topic == connection.topic:
                            ros2_writer.write(uwb_con, timestamp, rawdata)


    def get_raw_dict(self):
        meas_dict = {}
        meas_dict["name"] = self.name
        meas_dict["save_folder"] = self.save_folder
        meas_dict["tb2"]: Turtlebot4 = self.tb2
        meas_dict["tb3"]: Turtlebot4 = self.tb3
        meas_dict["uwb"] = self.uwb
        return meas_dict

    def save_raw_data(self):
        meas_dict = self.get_raw_dict()
        with open(self.save_folder + self.name + "_raw.pkl", "wb") as f:
            pkl.dump(meas_dict, f)

    def load_raw_data(self, pkl_file):
        with open(pkl_file, "rb") as f:
            meas_dict = pkl.load(f)
        tb2 = meas_dict["tb2"]
        tb3 = meas_dict["tb3"]
        uwb = meas_dict["uwb"]
        self.tb2.load_raw_data(tb2)
        self.tb3.load_raw_data(tb3)
        self.uwb.load_raw_data(uwb)
        self.name = meas_dict["name"]
        self.save_folder = meas_dict["save_folder"]

    def save_sampled_data(self):
        meas_dict = self.get_raw_dict()
        meas_dict["sampled_frequency"] = self.sample_frequency

        with open(self.save_folder + self.name + "_sampled.pkl", "wb") as f:
            pkl.dump(meas_dict, f)

    def load_sampled_data(self, pkl_file):
        with open(pkl_file, "rb") as f:
            meas_dict = pkl.load(f)
        tb2 = meas_dict["tb2"]
        tb3 = meas_dict["tb3"]
        uwb = meas_dict["uwb"]
        self.tb2.load_sampled_data(tb2)
        self.tb3.load_sampled_data(tb3)
        self.uwb.load_sampled_data(uwb)
        self.name = meas_dict["name"]
        self.save_folder = meas_dict["save_folder"]
        self.sample_frequency = meas_dict["sampled_frequency"]


    #--------------------
    # Data splitting
    #____________________
    def split_data(self, split_ids):
        meas_list = []
        prev_id = 0
        split_ids.append(-1)
        for split_id in split_ids:
            meas = Measurement()
            meas.name = self.name
            meas.save_folder = self.save_folder
            meas.sample_frequency = self.sample_frequency
            meas.tb2 = self.tb2.split_data(prev_id, split_id)
            meas.tb3 = self.tb3.split_data(prev_id, split_id)
            meas.uwb = self.uwb.split_data(prev_id, split_id)
            meas_list.append(meas)
            prev_id = split_id
        return meas_list

    #--------------------
    # Sampling and processing data
    #____________________
    def sample(self, frequency, start_time=None, end_time=None):
        # Place holder to find start time
        self.sample_frequency = frequency
        if start_time is None:
            ts = np.array([self.tb2.vicon_frame.t[0], self.tb2.vio_frame.t[0], self.tb3.vicon_frame.t[0], self.tb3.vio_frame.t[0], self.uwb.t[0]])
            start_time = np.max(ts)
        if end_time is None:
            te = np.array([self.tb2.vicon_frame.t[-1], self.tb2.vio_frame.t[-1], self.tb3.vicon_frame.t[-1], self.tb3.vio_frame.t[-1], self.uwb.t[-1]])
            end_time = np.min(te)

        self.tb2.sample(frequency, start_time, end_time)
        self.tb3.sample(frequency, start_time, end_time)
        self.uwb.sample(frequency, start_time, end_time)

    #--------------------
    # UWB processing
    #____________________
    def optimise_uwb_T(self, i_0=0, i_e=-1):
        # Maybe not the best thing to do.
        # I rememeber having set the COG of the turtlebots in vicon at the marker that is almost at UWB chip.
        res = self.uwb.optimise_uwb_T(self.tb2, self.tb3 ,i_0 =i_0,  i_e = i_e)
        print(res)


    def get_uwb_distances(self):
        ds = np.linalg.norm(self.tb2.vicon_frame.sampled_p - self.tb3.vicon_frame.sampled_p, axis=1)
        self.uwb.real_d = ds

    def get_uwb_LOS(self, sigma_d):
        los_state = np.zeros(len(self.uwb.sampled_d))
        self.get_uwb_distances()
        for i in range(len(self.uwb.sampled_d)):
            if np.abs(self.uwb.sampled_d[i]  - self.uwb.real_d[i]) < 2*sigma_d:
                los_state[i] = 1
        return los_state

    #------------------
    # VIO processing
    #-----------------
    def get_VIO_error(self, plot=False):
        self.tb2.get_vio_error(plot)
        self.tb3.get_vio_error(plot)

    def correct_orb_transformation(self):
        T_cor = np.array([[0,-1,0,0],
                          [0,0,-1,0],
                            [1,0,0,0],
                            [0,0,0,1]])
        self.tb2.correct_orb_transformation(T_cor)
        self.tb3.correct_orb_transformation(T_cor)

    #------------------
    # RPE processing
    #-----------------
    def get_rpe_transformation(self):
        # RPE_T_tb2_tb3 = np.empty((0,4,4))
        # RPE_T_tb3_tb2 = np.empty((0,4,4))
        for i in range(len(self.tb2.vicon_frame.sampled_t)):
            T_w_tb2 = self.tb2.vicon_frame.sampled_T[i]
            T_w_tb3 = self.tb3.vicon_frame.sampled_T[i]
            T_tb2_tb3 = np.linalg.inv(T_w_tb2) @ T_w_tb3
            T_tb3_tb2 = np.linalg.inv(T_w_tb3) @ T_w_tb2
            self.RPE_T_tb2_tb3 = np.append(self.RPE_T_tb2_tb3, [T_tb2_tb3], axis=0)
            self.RPE_T_tb3_tb2 = np.append(self.RPE_T_tb3_tb2, [T_tb3_tb2], axis=0)


    #------------------
    # Data acquisition
    #-----------------

    def get_measurements(self, t, d, T_tb2_vio, T_tb3_vio, vw_tb2_vio, vw_tb3_vio, T_tb2, T_tb3):
        self.times.append(t)
        self.distances.append(d)
        self.T_tb2_vio.append(T_tb2_vio)
        self.T_tb3_vio.append(T_tb3_vio)
        self.T_tb2.append(T_tb2)
        self.T_tb3.append(T_tb3)
        self.vw_tb2_vio.append(vw_tb2_vio)
        self.vw_tb3_vio.append(vw_tb3_vio)



    def get_dict(self):
        traj_dict = {}
        traj_dict["times"] = self.times
        traj_dict["distances"] = self.distances
        traj_dict["DT_tb2"] = self.T_tb2_vio
        traj_dict["DT_tb3"] = self.T_tb3_vio
        traj_dict["T_tb2"] = self.T_tb2
        traj_dict["T_tb3"] = self.T_tb3
        return traj_dict

    def save_measurement_dict(self, file="./measurement.pkl"):
        meas_dict = self.get_dict()
        with open(file, "wb") as f:
            pkl.dump(meas_dict, f)

    def load_measurement_dict(self, file="./measurement.pkl"):
        with open(file, "rb") as f:
            meas_dict = pkl.load(f)
        self.times = meas_dict["times"]
        self.distances = meas_dict["distances"]
        self.T_tb2_vio = meas_dict["DT_tb2"]
        self.T_tb3_vio = meas_dict["DT_tb3"]
        self.T_tb2 = meas_dict["T_tb2"]
        self.T_tb3 = meas_dict["T_tb3"]

    #------------------
    # Plotting
    #-----------------
    def print_sampled_lengths(self):
        print("tb2 vicon: ", len(self.tb2.vicon_frame.sampled_t))
        print("tb2 vio: ", len(self.tb2.vio_frame.sampled_t))
        print("tb3 vicon: ", len(self.tb3.vicon_frame.sampled_t))
        print("tb3 vio: ", len(self.tb3.vio_frame.sampled_t))
        print("uwb: ", len(self.uwb.sampled_t))

    def plot_sampled(self):
        self.tb2.vio_frame.plot_sampled()
        self.tb3.vio_frame.plot_sampled()
        self.tb2.vicon_frame.plot_sampled()
        self.tb3.vicon_frame.plot_sampled()
        self.uwb.plot_sampled()




    #------------------
    # Experiment preprocessing
    #-----------------

def create_experiment(results_folder, sig_v, sig_w, sig_uwb, alpha=1., kappa=-1., beta=2., n_azimuth=4,
                      n_altitude=3, n_heading=4):
    tas = TwoAgentSystem(trajectory_folder="./", result_folder=results_folder)
    tas.set_ukf_properties(kappa=kappa, alpha=alpha, beta=beta, n_azimuth=n_azimuth, n_altitude=n_altitude,
                           n_heading=n_heading)
    tas.set_uncertainties(sig_v, sig_w, sig_uwb)
    return tas

def create_experimental_data(data_folder, sig_v, sig_w, sig_uwb):
    experiments=[]
    measurements = []
    # check wether data_folder is a file or a folder
    if os.path.isfile(data_folder):
        list_of_files = [data_folder.split("/")[-1]]
        data_folder = "/".join(data_folder.split("/")[0:-1])

    else:
        list_of_files = os.listdir(data_folder)
    print(list_of_files)
    for sampled_data in list_of_files:
        name = sampled_data.split(".")[-2].split("/")[-1]
        measurement = Measurement()
        measurement.load_sampled_data(data_folder+"/"+sampled_data)
        sample_freq = measurement.sample_frequency

        #
        sig_d = sig_v / sample_freq
        sig_phi = sig_w / sample_freq
        Q_vio = np.diag([sig_d ** 2, sig_d ** 2, sig_d ** 2, sig_phi ** 2])

        # measurement.get_uwb_distances()
        uwb = measurement.uwb.sampled_d
        uwb_los = measurement.get_uwb_LOS(sig_uwb)
        DT_vio_tb2 = measurement.tb2.vio_frame.get_relative_motion_in_T()
        DT_vio_tb3 = measurement.tb3.vio_frame.get_relative_motion_in_T()
        T_vicon_tb2 = measurement.tb2.vicon_frame.sampled_T
        T_vicon_tb3 = measurement.tb3.vicon_frame.sampled_T

        experiment_data = {}
        experiment_data["name"] = name
        experiment_data["sample_freq"] = sample_freq
        experiment_data["drones"] = {}
        experiment_data["drones"]["drone_0"] = {"DT_slam": DT_vio_tb2, "T_real": T_vicon_tb2, "Q_slam": Q_vio}
        experiment_data["drones"]["drone_1"] = {"DT_slam": DT_vio_tb3, "T_real": T_vicon_tb3, "Q_slam": Q_vio}
        experiment_data["uwb"] = uwb
        experiment_data["los_state"] = uwb_los


        measurements.append(measurement)
        # experiment_data["eps_d"] = np.abs(measurement.uwb.real_d - measurement.uwb.sampled_d)

        experiments.append(experiment_data)
    return experiments, measurements

def create_experimental_sim_data(data_folder, sig_v, sig_w, sig_uwb):
    experiments, measurements = create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
    for experiment in experiments:
        T_0_prev = np.eye(4)
        T_1_prev = np.eye(4)
        for i, uwb in enumerate(experiment["uwb"]):
            T_0 = experiment["drones"]["drone_0"]["T_real"][i]
            T_1 = experiment["drones"]["drone_1"]["T_real"][i]
            uwb = np.linalg.norm(T_0[0:3, 3] - T_1[0:3, 3])
            experiment["uwb"][i] = uwb + np.random.normal(0, sig_uwb)
            if i > 0:
                dT_0 = inv_transformation_matrix(T_0_prev) @ T_0
                t0_noise =  np.random.normal(0, sig_v/experiment["sample_freq"], size=(3))
                theta_0_noise = np.random.normal(0, sig_w/experiment["sample_freq"])
                T_0_noise = transformation_matrix_from_rot_vect([0, 0, theta_0_noise], t0_noise)
                dT_0_noise = dT_0 @ T_0_noise
                experiment["drones"]["drone_0"]["DT_slam"][i-1] = dT_0_noise
                dT_1 = inv_transformation_matrix(T_1_prev) @ T_1
                t1_noise = np.random.normal(0, sig_v / experiment["sample_freq"], size=(3))
                theta_1_noise = np.random.normal(0, sig_w / experiment["sample_freq"])
                T_1_noise = transformation_matrix_from_rot_vect([0, 0, theta_1_noise], t1_noise)
                dT_1_noise = dT_1 @ T_1_noise
                experiment["drones"]["drone_1"]["DT_slam"][i - 1] = dT_1_noise

            T_0_prev = T_0
            T_1_prev = T_1




    return experiments, measurements