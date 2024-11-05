import numpy as np
import quaternion
import matplotlib
import pickle as pkl

matplotlib.use('Qt5Agg', force=True)
import matplotlib.pyplot as plt
from Code.UtilityCode import Transformation_Matrix_Fucntions as TMF


def quaternion_difference(q1, q2):
    # Calculate the quaternion difference: q_diff = q2 * q1_inverse
    q1_inverse = np.conj(q1)
    q_diff = q2 * q1_inverse
    return q_diff


def quaternion_to_axis_angle(q):
    # Convert a quaternion to axis-angle representation
    angle = 2 * np.arccos(q.real)
    axis = q.imag / np.sqrt(1 - q.real ** 2)
    return axis, angle


def angular_velocity_between_quaternions(q1, q2, time_difference):
    q_diff = quaternion_difference(q1, q2)
    axis, angle = quaternion_to_axis_angle(q_diff)
    angular_velocity = axis * (angle / time_difference)
    return angular_velocity


class Frame():
    def __init__(self):
        # raw sampels
        self.R = np.eye(3)
        self.t = np.empty((0))
        self.p = np.empty((0, 3))
        self.q = np.empty((0, 4))
        self.v = np.empty((0, 3))
        self.v_cor = np.empty((0, 3))
        self.w = np.empty((0, 3))
        self.T = np.empty((0, 4, 4))
        self.DT = np.empty((0, 4, 4))

        self.prev_odom_o = quaternion.from_float_array(np.zeros(4))
        self.prev_odom_t = np.zeros(3)
        self.prev_t = None

        # Processed samples
        self.sample_frequency = None
        self.sampled_t = np.empty((0))
        self.sampled_T = np.empty((0, 4, 4))
        #Depracticated>>
        self.sampled_p = np.empty((0, 3))
        self.sampled_q = np.empty((0, 4))
        self.sampled_v = np.empty((0, 3))
        self.sampled_w = np.empty((0, 3))
    #-----------------
    # Data acquisition
    #-----------------

    def get_full_measurement(self, t, p, q, v=None, w=None):
        T = TMF.transformation_matrix_from_q(q, p)
        if self.t.shape[0] == 0:
            DT = np.eye(4)
            self.DT = np.vstack((self.DT, DT.reshape(1, *DT.shape)))
            v = np.zeros(3)
            w = np.zeros(3)
        else:
            dt = t - self.t[-1]
            if v is None:
                DT = TMF.inv_transformation_matrix(self.T[-1]) @ T
                v = TMF.get_translation(DT) / dt
            if w is None:
                DT = TMF.inv_transformation_matrix(self.T[-1]) @ T
                w = TMF.get_rotation_vector(DT) / dt
            dq = quaternion.as_float_array(quaternion.from_rotation_vector(w * dt))
            DT = TMF.transformation_matrix_from_q(dq, v*dt)

        self.T = np.vstack((self.T, T.reshape(1, *np.eye(4).shape)))
        self.DT = np.vstack((self.DT, DT.reshape(1, *DT.shape)))
        self.p = np.vstack((self.p, p))
        self.q = np.vstack((self.q, q))
        self.v = np.vstack((self.v, v))
        self.w = np.vstack((self.w, w))
        self.t = np.append(self.t, t)

    # @deprecated(reason="Use get_full_measurement instead")
    # def get_measurement(self, t, p, q, v = None):
    #     if self.prev_t is not None:
    #         dt = (t - self.prev_t)
    #         R = self.R@quaternion.as_rotation_matrix(self.prev_odom_o)
    #         if v is None:
    #             v = R.T @ (p - self.prev_odom_t) / dt
    #         else:
    #             v = v
    #         diff = quaternion.from_float_array(q)* np.conj(self.prev_odom_o)
    #         w = quaternion.as_rotation_vector(diff) / dt
    #     else:
    #         v = np.zeros(3)
    #         w = np.zeros(3)
    #     self.prev_t = t
    #     self.prev_odom_t = p
    #     self.prev_odom_o = quaternion.from_float_array(q)
    #
    #     self.get_full_measurement(t, p, q, v, w)

    #-----------------
    # Data saving and loading
    #-----------------
    def load_raw_frame(self, frame):
        self.t = frame.t
        self.p = frame.p
        self.q = frame.q
        self.v = frame.v
        self.w = frame.w

    def load_sampled_frame(self, frame):
        self.sample_frequency = frame.sample_frequency
        self.sampled_t  = frame.sampled_t
        self.sampled_p  = frame.sampled_p
        self.sampled_q  = frame.sampled_q
        self.sampled_v  = frame.sampled_v
        self.sampled_w  = frame.sampled_w
        self.sampled_T = frame.sampled_T

    def split_data(self, id_0, id_1):
        frame = Frame()
        frame.sample_frequency = self.sample_frequency
        frame.sampled_t = self.sampled_t[id_0:id_1]
        frame.sampled_p = self.sampled_p[id_0:id_1]
        frame.sampled_q = self.sampled_q[id_0:id_1]
        frame.sampled_v = self.sampled_v[id_0:id_1]
        frame.sampled_w = self.sampled_w[id_0:id_1]
        frame.sampled_T = self.sampled_T[id_0:id_1]
        return frame

    def outlier_rejection(self, max_v = 0.3, max_a = 0.5, max_w =1, horizon = 10):
        v_uncor, w_uncor = self.get_relative_motion()
        self.v_cor = np.empty((0, 3))
        w_cor = np.empty((0, 3))

        self.v_cor = np.vstack((self.v_cor, v_uncor[0]))

        for i in range(1,len(v_uncor)):
            a = np.linalg.norm(v_uncor[i] -   self.v_cor[i-1]) * self.sample_frequency
            if a > max_a:
                v = self.v_cor[i-1]
            else:
                v = v_uncor[i]
            self.v_cor = np.vstack((self.v_cor, v))






    #-----------------
    # Sampling and processing
    #-----------------
    def sample(self, frequency, start_time=None, end_time=None):
        self.sample_frequency = frequency
        if start_time is None:
            start_time = self.t[0]
        if end_time is None:
            end_time = self.t[-1]
        dt = 1 / frequency
        sample_time = start_time + dt
        prev_t = self.t[0]
        prev_p = self.p[0]
        prev_q = self.q[0]
        prev_v = self.v[0]
        prev_w = self.w[0]
        for i in range(len(self.t)):
            if sample_time >= end_time:
                break
            while self.t[i] > sample_time and sample_time < end_time:
                self.sampled_t = np.append(self.sampled_t, sample_time)
                self.sampled_p = np.vstack((self.sampled_p, prev_p + (sample_time - prev_t) * (self.p[i] - prev_p) / (self.t[i] - prev_t)))
                q_slerp = quaternion.slerp(quaternion.from_float_array(prev_q), quaternion.from_float_array(self.q[i]), prev_t , self.t[i], [sample_time])
                self.sampled_q = np.vstack((self.sampled_q, quaternion.as_float_array(q_slerp)))
                T = TMF.transformation_matrix_from_q(quaternion.as_float_array(q_slerp), self.sampled_p[-1])
                self.sampled_T = np.vstack((self.sampled_T, T.reshape((1, *T.shape))))
                self.sampled_v = np.vstack((self.sampled_v, prev_v + (sample_time - prev_t) * (self.v[i] - prev_v) / (self.t[i] - prev_t)))
                self.sampled_w = np.vstack((self.sampled_w, prev_w + (sample_time - prev_t) * (self.w[i] - prev_w) / (self.t[i] - prev_t)))

                sample_time += dt

            # if self.t[i] < sample_time:
            prev_t = self.t[i]
            prev_p = self.p[i]
            prev_q = self.q[i]
            prev_v = self.v[i]
            prev_w = self.w[i]

    def inverse_transformation(self):
        p = np.empty((0, 3))
        q = np.empty((0, 4))
        T = np.empty((0, 4, 4))
        for i in range(len(self.sampled_p)):
            T_inv = TMF.inv_transformation_matrix(self.sampled_T[i])
            p = np.vstack((p, TMF.get_translation(T_inv)))
            q = np.vstack((q, TMF.get_quaternion(T_inv)))
            T = np.vstack((T, T_inv.reshape(1, *T_inv.shape)))
        self.sampled_p = p
        self.sampled_q = q
        self.sampled_T = T

    def get_corrected_transformation(self, T):
        p_cor = np.empty((0, 3))
        q_cor = np.empty((0, 4))
        T_cor = np.empty((0, 4, 4))
        for i in range(len(self.sampled_p)):
            T_e = self.sampled_T[i] @ T
            p, q = TMF.get_translation(T_e), TMF.get_quaternion(T_e)
            p_cor = np.vstack((p_cor, p))
            q_cor = np.vstack((q_cor, q))
            T_cor = np.vstack((T_cor, T_e.reshape(1, *T_e.shape)))
        return p_cor, q_cor, T_cor

    def set_corrected_transformation(self, T):
        self.sampled_p, self.sampled_q, self.sampled_T = self.get_corrected_transformation(T)

    def get_relative_motion_in_T(self):
        DT = np.empty((0, 4, 4))
        for i in range(len(self.sampled_p) - 1):
            tr = self.sampled_v[i] / self.sample_frequency
            rot = self.sampled_w[i] / self.sample_frequency
            dq = quaternion.as_float_array(quaternion.from_rotation_vector(rot))
            dT = TMF.transformation_matrix_from_q(dq, tr)
            DT = np.vstack((DT, dT.reshape(1, *dT.shape)))
            # T_i = self.sampled_T[i]
            # T_i1 = self.sampled_T[i + 1]
            # local_DT = TMF.inv_transformation_matrix(T_i) @ T_i1
            # DT = np.vstack((DT, local_DT.reshape(1, *local_DT.shape)))
        return DT

    def get_relative_motion(self, redo_bool=False):
        if self.sampled_v.shape[0] == 0 or redo_bool:
            sampled_v = np.zeros((1, 3))
            sampled_w = np.zeros((1, 3))
            for i in range(len(self.sampled_p)-1):
                T_i = TMF.transformation_matrix_from_q(self.sampled_q[i], self.sampled_p[i])
                T_i1 = TMF.transformation_matrix_from_q(self.sampled_q[i+1], self.sampled_p[i+1])
                DT = TMF.inv_transformation_matrix(T_i) @ T_i1
                sampled_v = np.vstack((sampled_v, TMF.get_translation(DT)*self.sample_frequency))
                sampled_w = np.vstack((sampled_w, TMF.get_rotation_vector(DT)*self.sample_frequency))
            return sampled_v, sampled_w
        else:
            return self.sampled_v, self.sampled_w


    #------------
    # plot methods
    #-----------
    def plot_sampled(self):
        _, ax = plt.subplots(4, 2)
        lables = ["x", "y", "z"]
        for i in range(3):
            ax[i,0].plot(self.t, self.p[:, i], '*', label="p_" + lables[i])
            ax[i,0].plot(self.sampled_t, self.sampled_p[:, i], '*',label="p_" + lables[i])
            ax[i,1].plot(self.t, self.q[:, i], '*', label="q_" + lables[i])
            ax[i,1].plot(self.sampled_t, self.sampled_q[:, i], '*', label="q_" + lables[i])
            # ax[i].plot(self.sampled_t, self.sampled_v[:, i], '*',label="v_" + lables[i])
            # ax[i].plot(self.sampled_t, self.sampled_w[:, i], '*',label="w_" + lables[i])
            ax[i,1].legend()
            ax[i,0].legend()
        ax[3,1].plot(self.t, self.q[:, 0], label="q_w")
        ax[3,1].plot(self.sampled_t, self.sampled_q[:, 0], '*', label="q_w")

    # -----------------
    # Deprecated
    # -----------------
    def get_ego_motion(self, t_0, t_1, sig_v=0, sig_w=0):
        idx_0 = np.argmax(self.t > t_0)
        idx_1 = np.argmax(self.t > t_1)
        if idx_1 == 0:
            idx_1 = len(self.t) - 1
        T0 = np.eye(4)
        T0[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(self.q[idx_0]))
        T0[:3, 3] = self.p[idx_0]
        TE = np.eye(4)
        for i in range(idx_0, idx_1):
            dt = self.t[i + 1] - self.t[i]
            Ti = np.eye(4)
            Ti[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(self.q[i + 1]))
            Ti[:3, 3] = self.p[i + 1]
            dT = TMF.inv_transformation_matrix(T0) @ Ti
            # Adding noise
            q = quaternion.from_rotation_matrix(dT[:3, :3])
            w = quaternion.as_rotation_vector(q) / dt
            w1 = w + np.random.normal(0, sig_w, 3)
            q1 = quaternion.from_rotation_vector(w1 * dt)
            dT[:3, :3] = quaternion.as_rotation_matrix(q1)
            dT[:3, 3] = dT[:3, 3] + np.random.normal(0, sig_v, 3) * dt
            # Applying the transformation
            TE = TE @ dT
            T0 = Ti
        #
        TE_0 = np.eye(4)
        TE_0[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(self.q[idx_0]))
        TE_0[:3, 3] = self.p[idx_0]

        TE_1 = np.eye(4)
        TE_1[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(self.q[idx_1]))
        TE_1[:3, 3] = self.p[idx_1]

        return TE, TE_1

    def correct_with_tranformation(self, Two):
        for i in range(len(self.p)):
            Tob = TMF.transformation_matrix_from_q(self.q[i], self.p[i])
            Twb = Two @ Tob
            self.p[i] = TMF.get_translation(Twb)
            self.q[i] = TMF.get_quaternion(Twb)

        def set_R(self, R):
            self.R = R


class Turtlebot4:
    def __init__(self, name):
        self.name = name
        self.vio_frame = Frame()
        # Transform to match x axis with body frame of the robot and z with negative gravitation
        self.vicon_frame = Frame()

        self.sample_frequency = None
        self.vio_v_error = np.empty((0, 3))
        self.vio_w_error = np.empty((0, 3))
    #-----------------
    # Data acquisition
    #-----------------
    def get_measuremend(self, data):
        t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
        p = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        o = np.array([data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z])
        v = np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z])
        w = np.array([data.twist.twist.angular.x, data.twist.twist.angular.y, data.twist.twist.angular.z])

        return t, p, o, v, w

    def update_orb(self, data):
        t, p, o, v, w = self.get_measuremend(data)
        # Tco = TMF.transformation_matrix_from_q(o, p)
        # Toc = TMF.inv_transformation_matrix(Tco)
        # R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        # trans = np.array([0, 0, 0])
        # #Tcb = TMF.transformation_matrix_from_R(R, trans)
        # #Tob = Toc @ Tcb
        # p,o = TMF.get_translation(Tob), TMF.get_quaternion(Tob)
        self.vio_frame.get_full_measurement(t, p, o, v, w)

    def update_vins(self, data):
        t, p, o, v, w = self.get_measuremend(data)
        self.vio_frame.get_full_measurement(t, p, o)

    def update_specVIO(self, data):
        o = np.array([data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z])
        R_s = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        R = quaternion.as_rotation_matrix(quaternion.from_float_array(o))

        v = R_s @ R.T @ np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z])
        data.twist.twist.linear.x = v[0]
        data.twist.twist.linear.y = v[1]
        data.twist.twist.linear.z = v[2]
        t, p, o, v, w = self.get_measuremend(data)
        self.vio_frame.get_full_measurement(t, p, o, v, w)

    def update_vicon(self, data, t=0.):
        t = t / 1e9
        try:
            odom_p = np.array([data.x_trans, data.y_trans, data.z_trans])/1e3
            odom_o = np.array([data.w, data.x_rot, data.y_rot, data.z_rot])
            self.vicon_frame.get_full_measurement(t, odom_p, odom_o)
        except AttributeError:
            try:
                t, p, o, v, w = self.get_measuremend(data)
                self.vicon_frame.get_full_measurement(t, p, o, v, w)
            except AttributeError:
                t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
                p = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
                o = np.array([data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                              data.pose.pose.orientation.z])
                self.vicon_frame.get_full_measurement(t, p, o)


    #--------------
    # Sampling and processing
    #-------------
    def sample(self, frequency, start_time=None, end_time=None):
        self.sample_frequency = frequency
        self.vio_frame.sample(frequency, start_time, end_time)
        self.vicon_frame.sample(frequency, start_time, end_time)


    def correct_orb_transformation(self, T_cor):
        self.vio_frame.inverse_transformation()
        self.vio_frame.set_corrected_transformation(T_cor)



    def get_vio_error(self, plot=False):
        vio_v, vio_w = self.vio_frame.get_relative_motion()
        vicon_v, vicon_w = self.vicon_frame.get_relative_motion(redo_bool=True)
        self.vio_v_error = vio_v - vicon_v
        self.vio_w_error = vio_w - vicon_w

        mean_v_error = np.mean(self.vio_v_error, axis=0)
        mean_w_error = np.mean(self.vio_w_error, axis=0)
        std_v_error = np.std(self.vio_v_error, axis=0)
        std_w_error = np.std(self.vio_w_error, axis=0)


        try:
            self.vio_v_cor_error = self.vio_frame.v_cor - vicon_v
            mean_v_cor_error = np.mean(self.vio_v_cor_error, axis=0)
            std_v_cor_error = np.std(self.vio_v_cor_error, axis=0)
            print("Mean v_cor error: ", mean_v_cor_error, " Std v_cor error: ", std_v_cor_error)
        except:
            mean_v_cor_error = np.nan
            std_v_cor_error = np.nan
            print("no self.vio_frame.v_cor")

        print("Mean v error: ", mean_v_error, " Std v error: ", std_v_error)
        print("Mean w error: ", mean_w_error, " Std w error: ", std_w_error)

        if plot:
            _, ax = plt.subplots(3, 2)
            labels = ["x", "y", "z"]
            for i in range(3):
                ax[i, 0].plot(vio_v[:, i], label="vio_v " + labels[i])
                ax[i, 0].plot(vicon_v[:, i], label="vicon_v " + labels[i])
                try:
                    ax[i, 0].plot(self.vio_frame.v_cor[:,i], label="vio_v_cor " + labels[i])
                except:
                    print("no self.vio_frame.v_cor")
                ax[i, 1].plot(vio_w[:, i], label="vio_w " + labels[i])
                ax[i, 1].plot(vicon_w[:, i], label="vicon_w " + labels[i])
                ax[i, 0].legend()
                ax[i, 1].legend()
        return mean_v_cor_error, std_v_cor_error, mean_v_error, std_v_error, mean_w_error, std_w_error
    #--------------
    # Plot methods
    #-------------
    def plot(self):
        _, ax = plt.subplots(3, 1)

        lables = ["x", "y", "z"]
        # for i in range(3):
        #     axt = ax[i].twinx()
        #     ax[i].plot(self.vins_frame.t[1:, ], self.vins_frame.p[1:, i], label="p_vins_" + lables[i])
        #     axt.plot(self.vicon_frame.t[1:, ], self.vicon_frame.p[1:, i], label="p_vicon_" + lables[i], color='r')
        #     axt.set_ylim(bottom=0)
        #     ax[i].legend()

        _, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(self.vio_frame.t[1:, ], self.vio_frame.v[1:, i], label="v_vins_" + lables[i])
            ax[i].plot(self.vicon_frame.t[1:, ], self.vicon_frame.v[1:, i], label="v_vicon_" + lables[i])
            ax[i].legend()

        _, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(self.vio_frame.t[1:, ], self.vio_frame.w[1:, i], label="w_vins_" + lables[i])
            ax[i].plot(self.vicon_frame.t[1:, ], self.vicon_frame.w[1:, i], label="w_vicon_" + lables[i])
            ax[i].legend()
        plt.show()

    def plot_trajectory(self, ax):
        ax.plot(self.vicon_frame.p[1:, 0], self.vicon_frame.p[1:, 1], color="tab:blue", label="vicon")
        ax.plot(self.vicon_frame.p[0,0], self.vicon_frame.p[0,1], 'o', color="tab:blue", label="vicon")
        ax.plot(self.vicon_frame.p[-1,0], self.vicon_frame.p[-1,1], 'x', color="tab:blue", label="vicon")
        ax.plot(self.vio_frame.p[1:, 0], self.vio_frame.p[1:, 1], color="tab:orange", label="vins")
        ax.plot(self.vio_frame.p[0, 0], self.vio_frame.p[0, 1], 'o', color="tab:orange", label="vicon")
        ax.plot(self.vio_frame.p[-1, 0], self.vio_frame.p[-1, 1], 'x', color="tab:orange", label="vicon")

    def plot_vio_error(self, ax= None):
        if ax is None:
            _, ax = plt.subplots(3, 2)
        labels = ["x", "y", "z"]
        for i in range(3):
            ax[i, 0].plot(self.vio_v_error[:, i], label="vio_v_error "+labels[i])
            ax[i, 1].plot(self.vio_w_error[:, i], label="vio_w_error "+labels[i])
            try:
                ax[i, 0].plot(self.vio_v_cor_error[:, i], label="vio_v_cor_error "+labels[i])
            except AttributeError:
                print("no self.vio_v_cor_error")
            ax[i, 0].legend()
            ax[i, 1].legend()



    #------------------
    # Data splitting
    #------------------
    def split_data(self, id_0, id_1):
        tb = Turtlebot4(self.name)
        tb.vio_frame = self.vio_frame.split_data(id_0, id_1)
        tb.vicon_frame = self.vicon_frame.split_data(id_0, id_1)
        return tb



    #------------------
    # Data saving and loading
    #------------------
    def load_raw_data(self, tb):
        self.sample_frequency = tb.sample_frequency
        self.vio_frame.load_raw_frame(tb.vio_frame)
        self.vicon_frame.load_raw_frame(tb.vicon_frame)

    def load_sampled_data(self, tb):
        self.sample_frequency = tb.sample_frequency
        self.vio_frame.load_sampled_frame(tb.vio_frame)
        self.vicon_frame.load_sampled_frame(tb.vicon_frame)

    def make_dict(self):
        vio_dict = {}
        vio_dict["t_vio"] = self.vio_frame.t
        vio_dict["v_vio"] = self.vio_frame.v
        vio_dict["w_vio"] = self.vio_frame.w
        vio_dict["t_vicon"] = self.vicon_frame.t
        vio_dict["v_vicon"] = self.vicon_frame.v
        vio_dict["w_vicon"] = self.vicon_frame.w
        return vio_dict

    def save_vio(self, file_name = "./test.pkl"):
        vio_dict = self.make_dict()
        with open(file_name, "wb") as f:
            pkl.dump(vio_dict, f)

    def load_vio(self, file_name = "./test.pkl"):
        with open(file_name, "rb") as f:
            vio_dict = pkl.load(f)
        self.vio_frame.t = vio_dict["t_vio"]
        self.vio_frame.v = vio_dict["v_vio"]
        self.vio_frame.w = vio_dict["w_vio"]
        self.vicon_frame.t = vio_dict["t_vicon"]
        self.vicon_frame.v = vio_dict["v_vicon"]
        self.vicon_frame.w = vio_dict["w_vicon"]