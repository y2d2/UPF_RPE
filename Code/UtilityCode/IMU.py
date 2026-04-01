import deprecated
from rosbags.rosbag2 import Reader as ROS2Reader
import sqlite3

from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import os
import collections

from scipy.signal import butter, lfilter, lfilter_zi
import quaternion

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class IMU:
    def __init__(self, window_size=10):
        self.measurement = np.empty((6, 0))
        self.filtered_signal = np.empty((6, 0))
        self.rt_filtered_signal = np.empty((6, 0))
        self.t = []

        self.window_size = window_size
        self.imu_prev_t = None
        self.v = []
        self.imu_v = [np.array([0, 0, 0])]
        self.cam_p = []
        self.cam_p_norm = []
        self.cam_q = []
        self.imu_dt = []
        self.imu_a = []
        self.imu_a_avg = []
        self.imu_w = []
        self.imu_w_avg = []
        self.imu_Dt = []
        self.imu_q = []
        self.grav_q = []
        self.g = 9.81
        self.cam_t = np.array([-0.0291, -0.00241, 0.0019])
        self.T_ex = np.array([[1, 0, 0, -0.0291],
                              [0, 1, 0, -0.00241],
                              [0, 0, 1, 0.0019],
                              [0, 0, 0., 1.]])
        self.set_filter()
        self.set_low_filter()

        self.acc_uf = np.empty((0, 3))
        self.gyr_uf = np.empty((0, 3))
        self.acc = np.empty((0, 3))
        self.gyr = np.empty((0, 3))
        self.acc_lf = np.empty((0, 3))
        self.gyr_lf = np.empty((0, 3))

        self.acc_bias = np.empty((0, 3))
        self.gyr_bias = np.empty((0, 3))

        self.roll_uf = []
        self.pitch_uf = []
        self.roll = []
        self.pitch = []
        self.roll_lf = []
        self.pitch_lf = []

        self.v_imu = np.empty((0, 3))
    #-------------------------------------------------
    # Data aqcuisition
    #-------------------------------------------------

    def start_new_frame(self):
        self.imu_a.append(np.empty((0, 3)))
        self.imu_w.append(np.empty((0, 3)))
        self.imu_dt.append(np.empty(0))
        self.imu_a_avg.append(np.zeros(3))
        self.imu_w_avg.append(np.zeros(3))
        self.imu_Dt.append(0)
        if len(self.imu_Dt) > self.window_size:
            self.imu_a.pop(0)
            self.imu_w.pop(0)
            self.imu_dt.pop(0)
            self.imu_v.pop(0)
            self.imu_w_avg.pop(0)
            self.imu_a_avg.pop(0)
            self.imu_Dt.pop(0)
        return self.imu_v[0]

    def get_imu_message(self, data):
        # zet in same orientation as the world frame ( imu has a left handed orientation)
        if len(self.imu_dt) != 0:
            t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
            if self.imu_prev_t is not None:
                dt = t - self.imu_prev_t
                self.imu_dt[-1] = np.append(self.imu_dt[-1], np.array([dt]))
                acc_uf = np.array([data.linear_acceleration.z, data.linear_acceleration.y, data.linear_acceleration.x])
                gyr_uf = np.array([-data.angular_velocity.z, -data.angular_velocity.y, -data.angular_velocity.x])
                self.gyr_uf = np.append(self.gyr_uf, gyr_uf.reshape(1, 3), axis=0)
                self.acc_uf = np.append(self.acc_uf, acc_uf.reshape(1, 3), axis=0)

                acc, gyr = self.run_filter(acc_uf, gyr_uf)
                self.acc = np.append(self.acc, acc.reshape(1, 3), axis=0)
                self.gyr = np.append(self.gyr, gyr.reshape(1, 3), axis=0)
                self.imu_a_avg[-1] += acc  * dt
                self.imu_w_avg[-1] += gyr  * dt

                acc_lf, gyr_lf = self.run_low_filter(acc_uf, gyr_uf)
                self.acc_lf = np.append(self.acc_lf, acc_lf.reshape(1, 3), axis=0)
                self.gyr_lf = np.append(self.gyr_lf, gyr_lf.reshape(1, 3), axis=0)
                self.acc_bias = acc_lf
                self.gyr_bias = gyr_lf

                self.imu_Dt[-1] += self.imu_dt[-1][-1]

                q , roll, pitch = self.get_roll_pitch_from_g(acc)
                self.roll.append(roll)
                self.pitch.append(pitch)

                _, roll_uf, pitch_uf = self.get_roll_pitch_from_g(acc_uf)
                self.roll_uf.append(roll_uf)
                self.pitch_uf.append(pitch_uf)
                _, roll_lf, pitch_lf = self.get_roll_pitch_from_g(acc_lf)
                self.roll_lf.append(roll_lf)
                self.pitch_lf.append(pitch_lf)


            self.imu_prev_t = t

    #-------------------------------------------------
    # Filter
    #-------------------------------------------------
    def set_filter(self, order=2, cutoff_frequency=1):
        dt = 0.004
        nyquist_frequency = 0.5 * 1 / dt  # Nyquist frequency (half of the sampling rate)
        self.b, self.a= butter(order, cutoff_frequency / nyquist_frequency, btype='lowpass')

        # zi = lfilter(b, a, np.array([-9.81, 0, 0, 0, 0 ,0]) * (max(len(b), len(a)) - 1))
        zi_1 = lfilter_zi(self.b, self.a)
        zi_a_x = zi_1 * [0]
        zi_a_y = zi_1 * [0]
        zi_a_z = zi_1 * [-self.g]
        zi_g_x = zi_1 * [0]
        zi_g_y = zi_1 * [0]
        zi_g_z = zi_1 * [0]
        self.zi = [zi_a_x, zi_a_y, zi_a_z, zi_g_x, zi_g_y, zi_g_z]


    def run_filter(self, acc, gyr):
        meas = np.concatenate((acc, gyr), axis=0)
        filtered_measurement = []
        for i in range(len(meas)):
            fil_mes, self.zi[i] = lfilter(self.b, self.a, [meas[i]], zi=self.zi[i])
            filtered_measurement.append(fil_mes[0])
        return np.array(filtered_measurement[:3]), np.array(filtered_measurement[3:])

    def set_low_filter(self, order=2, cutoff_frequency=0.01):
        dt = 0.004
        nyquist_frequency = 0.5 * 1 / dt  # Nyquist frequency (half of the sampling rate)
        self.b_low, self.a_low= butter(order, cutoff_frequency / nyquist_frequency, btype='lowpass')

        # zi = lfilter(b, a, np.array([-9.81, 0, 0, 0, 0 ,0]) * (max(len(b), len(a)) - 1))
        zi_1 = lfilter_zi(self.b_low, self.a_low)
        zi_a_x = zi_1 * [0]
        zi_a_y = zi_1 * [0]
        zi_a_z = zi_1 * [-self.g]
        zi_g_x = zi_1 * [0]
        zi_g_y = zi_1 * [0]
        zi_g_z = zi_1 * [0]
        self.zi_low = [zi_a_x, zi_a_y, zi_a_z, zi_g_x, zi_g_y, zi_g_z]

    def run_low_filter(self, acc, gyr):
        meas = np.concatenate((acc, gyr), axis=0)
        filtered_measurement = []
        for i in range(len(meas)):
            fil_mes, self.zi_low[i] = lfilter(self.b_low, self.a_low, [meas[i]], zi=self.zi_low[i])
            filtered_measurement.append(fil_mes[0])
        return np.array(filtered_measurement[:3]), np.array(filtered_measurement[3:])


    #-------------------------------------------------
    # model
    #-------------------------------------------------

    def integrate(self, v0, acc_bias, gyr_bias):
        if len(self.imu_Dt)<1:
            return
        # Code can be way optimised, but for now not the point.
        self.imu_v = [v0]
        self.cam_p = []
        self.cam_p_norm = []
        self.cam_q = []
        self.imu_q = []
        self.grav_q = []
        # g_v = np.array([0,0,-g]) # aligned with world coordinate frame.
        v_imu = v0
        # q=q0
        for i in range(len(self.imu_a)):
            v = v_imu
            dv = self.imu_a_avg[i] - acc_bias*self.imu_Dt[i]
            v_t = v + dv
            q_local = quaternion.from_rotation_vector(self.imu_w_avg[i] - gyr_bias * self.imu_Dt[i])  # - gyr_bias*self.imu_Dt[i])
            p_local = v_t * self.imu_Dt[i]  # + 0.5 * dv * self.imu_Dt[i]

            v_imu = quaternion.as_rotation_matrix(q_local).T @ v_t

            T = np.eye(4)
            T[:3, :3] = quaternion.as_rotation_matrix(q_local)
            T[:3, 3] = p_local
            T = self.T_ex.T @ T @ self.T_ex
            p = T[:3, 3]
            self.cam_p.append(p)
            p_norm = p / np.linalg.norm(p)
            if p_norm[0] < 0:
                p_norm = -p_norm
            self.cam_p_norm.append(p_norm)
            self.cam_q.append(quaternion.from_rotation_matrix(T[:3, :3]))


            self.imu_v.append(v_imu)
            self.imu_q.append(q_local)
        self.v_imu = np.append(self.v_imu, v_imu.reshape(1, 3), axis=0)
            # self.grav_q.append(self.get_roll_pitch_from_g(self.imu_a[i][-1]-acc_bias))

    @deprecated.deprecated(version='1.0', reason="Will assume more or less costante roll and pitch ")
    def get_roll_pitch_from_g(self, acc):
        g = acc / np.linalg.norm(acc)
        roll = np.arctan2(g[1], -g[2])
        pitch = np.arctan2(-g[0], np.sqrt(g[1] ** 2 + g[2] ** 2))
        yaw = 0
        q_yaw = quaternion.from_rotation_vector(np.array([0, 0, yaw]))
        q_pitch = quaternion.from_rotation_vector(np.array([0, pitch, 0]))
        q_roll = quaternion.from_rotation_vector(np.array([roll, 0, 0]))
        # q = quaternion.from_rotation_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))
        grav_q = q_yaw * q_pitch * q_roll
        return grav_q, roll, pitch

    @deprecated.deprecated(version='1.0', reason="Use self.acc_bias and self.gyr_bias instead")
    def get_bias(self, g):
        g_vec = np.array([0, 0, 0])
        bias_acc = np.empty((0, 3))
        bias_gyr = np.empty((0, 3))
        for i in range(len(self.imu_a)):
            for j in range(len(self.imu_a[i])):
                q = self.get_roll_pitch_from_g(self.imu_a[i][j])
                acc = quaternion.as_rotation_matrix(q).T @ self.imu_a[i][j] - g_vec
                bias_acc = np.append(bias_acc, acc.reshape(1, 3), axis=0)
                bias_gyr = np.append(bias_gyr, self.imu_w[i][j].reshape(1, 3), axis=0)
        bias_gyr = np.mean(bias_gyr, axis=0)
        bias_acc = np.mean(bias_acc, axis=0)

        return bias_acc, bias_gyr


    #-------------------------------------------------
    # Plot
    #-------------------------------------------------

    def plot(self):
        _, ax = plt.subplots(3,3)
        labels = ["x", "y", "z"]
        for i in range(3):
            ax[i,0].plot(self.acc_uf[:,i], label="acc_"+labels[i])
            ax[i,0].plot(self.acc[:, i], label="acc_fil"+labels[i])
            ax[i,0].plot(self.acc_lf[:, i], label="acc_lf"+labels[i])
            ax[i,1].plot(self.gyr_uf[:,i], label="gyr_"+labels[i])
            ax[i,1].plot(self.gyr[:,i], label="gyr_fil"+labels[i])
            ax[i,1].plot(self.gyr_lf[:, i], label="gyr_lf"+labels[i])

        ax[0,2].plot(self.roll_uf, label="roll")
        ax[0,2].plot(self.roll, label="roll_fil")
        ax[0,2].plot(self.roll_lf, label="roll_lf")
        ax[1,2].plot(self.pitch_uf, label="pitch")
        ax[1,2].plot(self.pitch, label="pitch_fil")
        ax[1,2].plot(self.pitch_lf, label="pitch_lf")

        for a in ax:
            for b in a:
                b.legend()
                b.grid(True)

    def plot_v(self):
        plt.figure()
        labels = ["x", "y", "z"]
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(self.v_imu[:, i], label="v_"+labels[i])
            plt.legend()
            plt.grid(True)


if __name__=="__main__":
    rosbag_dir = "/home/yuri/Documents/PhD/ROS_WS/sharedDrive/Experiments/VIO_test/rosbag2_2023_09_26-08_28_21"

    tb2_imu_topic = "/oakd/imu/data"
    #tb3_imu_topic = "/tb3/oakd/imu/data"
    tb2_imu = IMU()
    tb2_imu.start_new_frame()
    # tb3_imu = IMU()
    # tb3_imu.start_new_frame()

    n = 0
    n_max = 15000*1e9
    with ROS2Reader(rosbag_dir) as ros2_reader:

        ros2_conns = [x for x in ros2_reader.connections]
        ros2_messages = ros2_reader.messages(connections=ros2_conns)
        for m, msg in enumerate(ros2_messages):
            (connection, timestamp, rawdata) = msg
            #print(timestamp)


            if n > n_max:
                break
            if (connection.topic == tb2_imu_topic):
                n += 1
                data = deserialize_cdr(rawdata, connection.msgtype)
                tb2_imu.get_imu_message(data)
                if (n % 25) == 0 and n != 0:
                    print(n, tb2_imu.imu_v)
                    tb2_imu.start_new_frame()
                    tb2_imu.integrate(np.zeros(3), tb2_imu.acc_bias, tb2_imu.gyr_bias)
            #
            # if (connection.topic == tb3_imu_topic):
            #     data = deserialize_cdr(rawdata, connection.msgtype)
            #     tb3_imu.get_imu_message(data)
    # #
    # # tb3_imu.filter()
    # # tb2_imu.filter()
    #tb3_imu.plot()
    tb2_imu.plot()
    tb2_imu.plot_v()
    plt.show()
