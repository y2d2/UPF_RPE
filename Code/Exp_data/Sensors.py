import re
from pathlib import Path

import numpy as np

import Code.UtilityCode.SE23 as SE23


ESP_PACKET_MSG_TYPE = "ros2_esp_bridge/msg/EspPacketStamped"
ESP_PACKET_MSG_DEF = "std_msgs/Header header\nstring raw_packet\n"


def _interpolate_vector(t0, t1, value0, value1, time):
    if t1 == t0:
        return np.array(value0, dtype=float)
    alpha = (time - t0) / (t1 - t0)
    return (1.0 - alpha) * np.array(value0, dtype=float) + alpha * np.array(value1, dtype=float)


def _find_segment(times, time):
    if len(times) == 0:
        return None
    if time < times[0] or time > times[-1]:
        return None
    if len(times) == 1:
        return 0
    for i in range(len(times) - 1):
        if times[i] <= time <= times[i + 1]:
            return i
    return len(times) - 1


def _load_rosbag_reader():
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:
        raise ImportError(
            "Bag-based Exp_data sensors require the 'rosbags' package."
        ) from exc
    return Reader, Stores, get_types_from_msg, get_typestore


class MeasuredTrajectory:
    def __init__(self, T_global=None, time=None, v_body=None, w_body=None, a_body=None):
        self.T_global = np.empty((0, 4, 4))
        self.X_OR = np.eye(5)
        self.X_RRi = []
        self.dX_i = []
        self.t = []
        self.v_body = np.empty((0, 3))
        self.w_body = np.empty((0, 3))
        self.a_body = np.empty((0, 3))
        if T_global is not None and time is not None:
            self.load_samples(T_global, time, v_body=v_body, w_body=w_body, a_body=a_body)

    def load_samples(self, T_global, time, v_body=None, w_body=None, a_body=None):
        self.T_global = np.array(T_global, dtype=float)
        self.t = list(np.array(time, dtype=float))
        n = len(self.t)
        self.v_body = np.zeros((n, 3)) if v_body is None else np.array(v_body, dtype=float)
        self.w_body = np.zeros((n, 3)) if w_body is None else np.array(w_body, dtype=float)
        self.a_body = np.zeros((n, 3)) if a_body is None else np.array(a_body, dtype=float)

        if n == 0:
            self.X_OR = np.eye(5)
            self.X_RRi = []
            self.dX_i = []
            return

        T0 = self.T_global[0]
        self.X_OR = np.eye(5)
        self.X_OR[:3, :3] = T0[:3, :3]
        self.X_OR[:3, 4] = T0[:3, 3]

        T0_inv = SE23.SE3_inverse(T0)
        self.X_RRi = []
        self.dX_i = []
        for i, T in enumerate(self.T_global):
            T_local = T0_inv @ T
            X = np.eye(5)
            X[:3, :3] = T_local[:3, :3]
            X[:3, 4] = T_local[:3, 3]
            X[:3, 3] = self.v_body[i]
            self.X_RRi.append(X)
            if i == 0:
                dX = np.eye(5)
                dX[:3, 3] = self.v_body[0]
            else:
                dX = np.linalg.inv(self.X_RRi[i - 1]) @ X
            self.dX_i.append(dX)

    def _interpolate_T(self, time):
        idx = _find_segment(self.t, time)
        if idx is None:
            print("Time is out of bounds.")
            return None
        if idx >= len(self.t) - 1 or self.t[idx] == time:
            return self.T_global[idx]
        T0 = self.T_global[idx]
        T1 = self.T_global[idx + 1]
        p = _interpolate_vector(self.t[idx], self.t[idx + 1], T0[:3, 3], T1[:3, 3], time)
        w0 = SE23.SE3_get_rotation_vector(T0)
        w1 = SE23.SE3_get_rotation_vector(T1)
        w = _interpolate_vector(self.t[idx], self.t[idx + 1], w0, w1, time)
        return SE23.SE3_from_rot_vec_and_trans(w, p)

    def get_global_postion_at_time(self, time):
        T = self._interpolate_T(time)
        if T is None:
            return None
        return T[:3, 3]

    def get_local_state_at_time(self, time):
        idx = _find_segment(self.t, time)
        if idx is None:
            print("Time is out of bounds.")
            return None
        if idx >= len(self.t) - 1 or self.t[idx] == time:
            X = self.X_RRi[idx]
            position = X[:3, 4]
            velocity = self.v_body[idx]
            angle_vect = SE23.get_w_from_SO3(X[:3, :3])
            return position, velocity, angle_vect, X

        X0 = self.X_RRi[idx]
        X1 = self.X_RRi[idx + 1]
        position = _interpolate_vector(self.t[idx], self.t[idx + 1], X0[:3, 4], X1[:3, 4], time)
        velocity = _interpolate_vector(self.t[idx], self.t[idx + 1], self.v_body[idx], self.v_body[idx + 1], time)
        w0 = SE23.get_w_from_SO3(X0[:3, :3])
        w1 = SE23.get_w_from_SO3(X1[:3, :3])
        angle_vect = _interpolate_vector(self.t[idx], self.t[idx + 1], w0, w1, time)
        X = SE23.SE23_from_t_v_rot(position, velocity, angle_vect)
        return position, velocity, angle_vect, X

    def get_current_state(self):
        if len(self.t) == 0:
            return None
        return self.get_local_state_at_time(self.t[-1])[:3] + (self.t[-1],)

    def stream_body_frame_measurements(self):
        for i, time in enumerate(self.t):
            yield self.a_body[i], self.v_body[i], self.w_body[i], time


class OdometrySensor:
    def __init__(self, trajectory=None, velocity_bool=False):
        self.true_trajectory = trajectory
        self.odom_trajectory = trajectory
        self.velocity_bool = velocity_bool

    def sensor_measurement(self, time):
        state = self.odom_trajectory.get_local_state_at_time(time)
        if state is None:
            return None, None, None
        _, velocity, _, _ = state
        idx = _find_segment(self.odom_trajectory.t, time)
        if idx is None:
            return None, None, None
        a = self.odom_trajectory.a_body[idx]
        w = self.odom_trajectory.w_body[idx]
        if self.velocity_bool:
            return None, velocity, w
        return a, None, w

    def get_new_measurement(self, *args, **kwargs):
        if "time" in kwargs:
            time = kwargs["time"]
        elif len(args) == 1:
            time = args[0]
        elif len(args) >= 4:
            time = args[3]
        else:
            raise TypeError("Measured odometry sensors expect a time argument.")
        return self.sensor_measurement(time)

    @classmethod
    def _messages_from_topic(cls, bag_path, topic, msgtype=None, register_esp=False):
        Reader, Stores, get_types_from_msg, get_typestore = _load_rosbag_reader()
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        if register_esp:
            typestore.register(get_types_from_msg(ESP_PACKET_MSG_DEF, ESP_PACKET_MSG_TYPE))

        bag_path = Path(bag_path).expanduser().resolve()
        with Reader(bag_path) as reader:
            for connection, _, rawdata in reader.messages():
                if connection.topic != topic:
                    continue
                decode_type = msgtype or connection.msgtype
                yield typestore.deserialize_cdr(rawdata, decode_type)


class VIO(OdometrySensor):
    def __init__(self, trajectory=None):
        super().__init__(trajectory=trajectory, velocity_bool=True)

    @classmethod
    def from_rosbag(cls, bag_path, odom_topic="/tb2/odom"):
        times = []
        T_global = []
        v_body = []
        w_body = []
        for msg in cls._messages_from_topic(bag_path, odom_topic):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])
            q = np.array([
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ])
            T = np.eye(4)
            T[:3, :3] = SE23.quaternion.as_rotation_matrix(SE23.quaternion.from_float_array(q))
            T[:3, 3] = p
            times.append(stamp)
            T_global.append(T)
            v_body.append([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ])
            w_body.append([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ])

        trajectory = MeasuredTrajectory(T_global, times, v_body=v_body, w_body=w_body)
        return cls(trajectory=trajectory)


class VIO_2D(VIO):
    def sensor_measurement(self, time):
        _, velocity, w = super().sensor_measurement(time)
        if velocity is not None:
            velocity = np.array(velocity, dtype=float)
            velocity[2] = 0.0
        if w is not None:
            w = np.array(w, dtype=float)
            w[0:2] = 0.0
        return None, velocity, w


class IMU(OdometrySensor):
    def __init__(self, trajectory=None):
        super().__init__(trajectory=trajectory, velocity_bool=False)
        self.b_acc = [np.zeros(3)]
        self.b_gyro = [np.zeros(3)]

    @classmethod
    def from_rosbag(cls, bag_path, imu_topic="/a200_0957/sensors/imu_0/data", T_OR=None, v_R0=None):
        times = []
        a_body = []
        w_body = []
        for msg in cls._messages_from_topic(bag_path, imu_topic):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            times.append(stamp)
            a_body.append([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ])
            w_body.append([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ])

        if len(times) == 0:
            trajectory = MeasuredTrajectory()
            return cls(trajectory=trajectory)

        if T_OR is None:
            T_OR = np.eye(4)
        if v_R0 is None:
            v_R0 = np.zeros(3)

        T_global = [np.array(T_OR, dtype=float)]
        velocities = [np.array(v_R0, dtype=float)]
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            T_prev = T_global[-1]
            v_prev = velocities[-1]
            R_prev = T_prev[:3, :3]
            a_prev = np.array(a_body[i - 1], dtype=float)
            w_prev = np.array(w_body[i - 1], dtype=float)
            R_new = R_prev @ SE23.get_SO3_rotation_matrix(w_prev, dt)
            v_new = v_prev + a_prev * dt
            p_new = T_prev[:3, 3] + (R_prev @ v_prev) * dt + 0.5 * (R_prev @ a_prev) * dt ** 2
            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = p_new
            T_global.append(T_new)
            velocities.append(v_new)

        trajectory = MeasuredTrajectory(T_global, times, v_body=velocities, w_body=w_body, a_body=a_body)
        return cls(trajectory=trajectory)

    def sensor_measurement(self, time):
        a, _, w = super().sensor_measurement(time)
        return a, None, w


class IMU_2D(IMU):
    def sensor_measurement(self, time):
        a, _, w = super().sensor_measurement(time)
        if a is not None:
            a = np.array(a, dtype=float)
            a[2] = 0.0
        if w is not None:
            w = np.array(w, dtype=float)
            w[0:2] = 0.0
        return a, None, w


class InterRobotDistanceSensor:
    def __init__(self, t=None, d=None, d_true=None, raw_packets=None, packet_decoder=None):
        self.t = [] if t is None else list(np.array(t, dtype=float))
        if d is None:
            self.d = [None] * len(self.t)
        else:
            self.d = list(np.array(d, dtype=float))
        self.d_true = [] if d_true is None else list(np.array(d_true, dtype=float))
        self.raw_packets = [] if raw_packets is None else list(raw_packets)
        self.packet_decoder = packet_decoder or self.default_packet_decoder

    @staticmethod
    def default_packet_decoder(raw_packet):
        values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_packet)
        if not values:
            raise ValueError(f"Could not decode a distance from raw packet: {raw_packet}")
        for value in reversed(values):
            distance = float(value)
            if distance > 0:
                return distance
        return float(values[-1])

    @classmethod
    def from_rosbag(cls, bag_path, uwb_topic="/tb2/uwb", packet_decoder=None):
        times = []
        raw_packets = []
        for msg in OdometrySensor._messages_from_topic(
            bag_path,
            uwb_topic,
            msgtype=ESP_PACKET_MSG_TYPE,
            register_esp=True,
        ):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            times.append(stamp)
            raw_packets.append(msg.raw_packet)
        return cls(t=times, raw_packets=raw_packets, packet_decoder=packet_decoder)

    def get_new_measurement(self, time):
        idx = _find_segment(self.t, time)
        if idx is None:
            print("Time is out of bounds.")
            return None
        if self.d[idx] is None and idx < len(self.raw_packets):
            self.d[idx] = self.packet_decoder(self.raw_packets[idx])
        if idx >= len(self.t) - 1 or self.t[idx] == time:
            return self.d[idx]
        if self.d[idx + 1] is None and idx + 1 < len(self.raw_packets):
            self.d[idx + 1] = self.packet_decoder(self.raw_packets[idx + 1])
        return float(_interpolate_vector(self.t[idx], self.t[idx + 1], self.d[idx], self.d[idx + 1], time))
