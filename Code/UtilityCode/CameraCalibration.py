import os

import quaternion
from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.rosbag1 import Writer as ROS1Writer
from rosbags.image import compressed_image_to_cvimage
import sqlite3

from rosbags.serde import deserialize_cdr, cdr_to_ros1, serialize_cdr, serialize_ros1
from rosbags.typesys.types import sensor_msgs__msg__Imu as Imu
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import std_msgs__msg__Header as Header
from rosbags.typesys.types import builtin_interfaces__msg__Time as Time

import numpy as np
import cv2
import csv
#from cv_bridge import CvBridge
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def make_raw_image_msg(msg, folder, save=True, show=False):
    # image = np.array(compressed_image_to_cvimage(msg), dtype=np.dtype('uint8'))
    # message = Image(Header(stamp=Time(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec), frame_id=msg.header.frame_id),
    #                 image.shape[0], image.shape[1], "mono8",is_bigendian=False, step=image.shape[1], data=image)
    image = cv2.imdecode(msg.data, cv2.IMREAD_GRAYSCALE)
    name = folder+ str(int(msg.header.stamp.sec* 1e9+msg.header.stamp.nanosec)) + ".png"
    if save:
        cv2.imwrite(name, image)
    if show:
        cv2.imshow(folder, image)
        cv2.waitKey(10)

def read_color_image(msg, folder, save=True, show=False):
    image = cv2.imdecode(msg.data, cv2.IMREAD_COLOR)
    name = folder + str(int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)) + ".png"
    if save:
        cv2.imwrite(name, image)
    if show:
        cv2.imshow(folder, image)
        cv2.waitKey(10)

def make_imu_msg(msg, inverse_imu=False , redo_imu=False):
    if inverse_imu or redo_imu:
        # change the x-axis to match the body frame of the robot
        dq = quaternion.from_rotation_vector(np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]))
        dq.x = -dq.x
        ang = quaternion.as_rotation_vector(dq)
        line = [int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec),
                ang[0], ang[1], ang[2],
                -msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
    else:
        line = [int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec),
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
    return line




if __name__=="__main__":
    save = False
    redo_imu = True
    inverse_imu = True
    show_left = True
    show_right = True
    show_RGB = False

    rosbag_dir = "../../test_cases/RPE_2_agents_LOS/Experiments/"
    rosbag2_name = "exp1_full"
    rosbag1_name = "tb3_cam_cal_1_folder/"

    imu_topic = "/oak/imu/data"
    rgb_topic = "/tb3/oakd/rgb/image_rect/compressed"
    left_topic= "/tb3/oakd/left/image_rect/compressed"
    right_topic= "/tb3/oakd/right/image_rect/compressed"

    rgb_dir = rosbag_dir+rosbag1_name+"cam_rgb/"
    l_dir = rosbag_dir+rosbag1_name+"cam_l/"
    r_dir = rosbag_dir+rosbag1_name+"cam_r/"
    if save:
        if os.path.exists(rosbag_dir+rosbag1_name):
            os.system("rm -r "+rosbag_dir+rosbag1_name)
        os.mkdir(rosbag_dir+rosbag1_name)
        os.mkdir(rgb_dir)
        os.mkdir(r_dir)
        os.mkdir(l_dir)
    elif redo_imu:
        if os.path.exists(rosbag_dir + rosbag1_name+"imu.csv"):
            os.system("rm "+rosbag_dir+rosbag1_name+"imu.csv")

    imu_lines =[]

    with ROS2Reader(rosbag_dir + rosbag2_name) as ros2_reader:
        ros2_conns = [x for x in ros2_reader.connections]
        print(ros2_conns)
        ros2_messages = ros2_reader.messages(connections=ros2_conns)
        for m, msg in enumerate(ros2_messages):
            (connection, timestamp, rawdata) = msg
            if (connection.topic == imu_topic):
                data = deserialize_cdr(rawdata, connection.msgtype)
                imu_line = make_imu_msg(data, inverse_imu)
                imu_lines.append(imu_line)

            if (connection.topic == left_topic):
                data = deserialize_cdr(rawdata, connection.msgtype)
                make_raw_image_msg(data, l_dir ,save=save, show=show_left)

            if (connection.topic == right_topic):
                data = deserialize_cdr(rawdata, connection.msgtype)
                make_raw_image_msg(data, r_dir,save=save, show=show_right)

            if (connection.topic == rgb_topic):
                data = deserialize_cdr(rawdata, connection.msgtype)
                read_color_image(data, rgb_dir,save=save, show=show_RGB)

    if save or redo_imu:
        with open(rosbag_dir+rosbag1_name+"imu.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["timestamp","omega_x","omega_y","omega_z","alpha_x","alpha_y","alpha_z"])
            csvwriter.writerows(imu_lines)