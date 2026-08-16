import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import upf_rte.UtilityCode.SE23 as SE23


ESP_PACKET_MSG_TYPE = "ros2_esp_bridge/msg/EspPacketStamped"
ESP_PACKET_MSG_DEF = "std_msgs/Header header\nstring raw_packet\n"
ESP_IMU_SAMPLE_FIELD_COUNT = 15
ESP_RANGE_FIELD_COUNT = 4


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


@dataclass
class EspImuSample:
    time_ms: int
    temperature: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float
    q0: float
    q1: float
    q2: float
    q3: float
    ros_time: float | None = None


@dataclass
class EspRangeMeasurement:
    rid: int
    dist_m: float
    fp_rssi: int
    rx_rssi: int


@dataclass
class EspPacket:
    packet_id: int
    packet_time: int
    imu_sample_count: int
    imu_dropped_samples: int
    imu_time_ms: int
    imu_point_count: int
    imu_samples: list[EspImuSample] = field(default_factory=list)
    valid_ranges: int = 0
    ranges: list[EspRangeMeasurement] = field(default_factory=list)


class EspPacketParser:
    @staticmethod
    def _normalize_row(row):
        normalized = list(row)
        while normalized and normalized[-1] == "":
            normalized.pop()
        return normalized

    @staticmethod
    def parse_row(row):
        normalized = EspPacketParser._normalize_row(row)
        if len(normalized) < 6:
            raise ValueError(f"Row too short to parse packet: {normalized}")

        packet_id = int(normalized[0])
        packet_time = int(normalized[1])
        imu_sample_count = int(normalized[2])
        imu_dropped_samples = int(normalized[3])
        imu_time_ms = int(normalized[4])
        imu_point_count = int(normalized[5])

        cursor = 6
        imu_samples = []
        for _ in range(imu_point_count):
            next_cursor = cursor + ESP_IMU_SAMPLE_FIELD_COUNT
            if next_cursor > len(normalized):
                raise ValueError(f"Not enough fields for {imu_point_count} IMU samples: {normalized}")
            fields = normalized[cursor:next_cursor]
            imu_samples.append(
                EspImuSample(
                    time_ms=int(fields[0]),
                    temperature=float(fields[1]),
                    ax=float(fields[2]),
                    ay=float(fields[3]),
                    az=float(fields[4]),
                    gx=float(fields[5]),
                    gy=float(fields[6]),
                    gz=float(fields[7]),
                    mx=float(fields[8]),
                    my=float(fields[9]),
                    mz=float(fields[10]),
                    q0=float(fields[11]),
                    q1=float(fields[12]),
                    q2=float(fields[13]),
                    q3=float(fields[14]),
                )
            )
            cursor = next_cursor

        if cursor >= len(normalized):
            raise ValueError(f"Missing valid_ranges field: {normalized}")

        valid_ranges = int(normalized[cursor])
        cursor += 1

        ranges = []
        for _ in range(valid_ranges):
            next_cursor = cursor + ESP_RANGE_FIELD_COUNT
            if next_cursor > len(normalized):
                raise ValueError(f"Not enough fields for {valid_ranges} ranges: {normalized}")
            fields = normalized[cursor:next_cursor]
            ranges.append(
                EspRangeMeasurement(
                    rid=int(fields[0]),
                    dist_m=float(fields[1]),
                    fp_rssi=int(fields[2]),
                    rx_rssi=int(fields[3]),
                )
            )
            cursor = next_cursor

        if cursor != len(normalized):
            raise ValueError(f"Unexpected trailing fields: {normalized[cursor:]}")

        return EspPacket(
            packet_id=packet_id,
            packet_time=packet_time,
            imu_sample_count=imu_sample_count,
            imu_dropped_samples=imu_dropped_samples,
            imu_time_ms=imu_time_ms,
            imu_point_count=imu_point_count,
            imu_samples=imu_samples,
            valid_ranges=valid_ranges,
            ranges=ranges,
        )

    @staticmethod
    def parse_packet_string(packet_string):
        return EspPacketParser.parse_row(packet_string.split(";"))


def _message_stamp_to_time(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


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

    def plot_trajectory(self, ax, label="", color="k", linestyle="-", marker=""):
        if self.T_global.shape[0] == 0:
            return
        positions = self.T_global[:, :3, 3]
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        if hasattr(ax, "plot3D"):
            ax.plot3D(x, y, z, label=label, color=color, linestyle=linestyle, marker=marker)
        else:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)


class OdometrySensor:
    # TODO : remove the true trajectory. This is uknown. In stead make an addtional class from Odometry Sensor that is called GT position>
    # This could then be used to Import VICON or similar data.
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

    @classmethod
    def _esp_packets_from_rosbag(cls, bag_path, topic):
        for msg in cls._messages_from_topic(
            bag_path,
            topic,
            msgtype=ESP_PACKET_MSG_TYPE,
            register_esp=True,
        ):
            ros_time = _message_stamp_to_time(msg)
            try:
                packet = EspPacketParser.parse_packet_string(msg.raw_packet)
            except ValueError:
                continue
            yield ros_time, packet


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


class ESP_IMU:
    def __init__(self, t=None, raw_time_ms=None, a_body=None, w_body=None, m_body=None, temperature=None, q=None):
        self.t = [] if t is None else list(np.array(t, dtype=float))
        self.raw_time_ms = [] if raw_time_ms is None else list(np.array(raw_time_ms, dtype=int))
        self.a_body = np.empty((0, 3)) if a_body is None else np.array(a_body, dtype=float)
        self.w_body = np.empty((0, 3)) if w_body is None else np.array(w_body, dtype=float)
        self.m_body = np.empty((0, 3)) if m_body is None else np.array(m_body, dtype=float)
        self.temperature = np.empty((0,)) if temperature is None else np.array(temperature, dtype=float)
        self.q = np.empty((0, 4)) if q is None else np.array(q, dtype=float)

    @classmethod
    def from_rosbag(cls, bag_path, packet_topic="/tb2/uwb"):
        samples = []
        for ros_time, packet in OdometrySensor._esp_packets_from_rosbag(bag_path, packet_topic):
            for sample in packet.imu_samples:
                sample.ros_time = ros_time - (packet.imu_time_ms - sample.time_ms) / 1000.0
                samples.append(sample)

        samples.sort(key=lambda sample: (sample.ros_time, sample.time_ms))
        if not samples:
            return cls()

        return cls(
            t=[sample.ros_time for sample in samples],
            raw_time_ms=[sample.time_ms for sample in samples],
            a_body=[[sample.ax, sample.ay, sample.az] for sample in samples],
            w_body=[[sample.gx, sample.gy, sample.gz] for sample in samples],
            m_body=[[sample.mx, sample.my, sample.mz] for sample in samples],
            temperature=[sample.temperature for sample in samples],
            q=[[sample.q0, sample.q1, sample.q2, sample.q3] for sample in samples],
        )

    def get_new_measurement(self, time):
        idx = _find_segment(self.t, time)
        if idx is None:
            print("Time is out of bounds.")
            return None, None, None
        if idx >= len(self.t) - 1 or self.t[idx] == time:
            return self.a_body[idx], None, self.w_body[idx]

        a = _interpolate_vector(self.t[idx], self.t[idx + 1], self.a_body[idx], self.a_body[idx + 1], time)
        w = _interpolate_vector(self.t[idx], self.t[idx + 1], self.w_body[idx], self.w_body[idx + 1], time)
        return a, None, w

    def to_measured_trajectory(self, T_OR=None, v_R0=None):
        if len(self.t) == 0:
            return MeasuredTrajectory()

        if T_OR is None:
            T_OR = np.eye(4)
        if v_R0 is None:
            v_R0 = np.zeros(3)

        T_global = [np.array(T_OR, dtype=float)]
        velocities = [np.array(v_R0, dtype=float)]
        for i in range(1, len(self.t)):
            dt = self.t[i] - self.t[i - 1]
            if dt < 0:
                continue
            T_prev = T_global[-1]
            v_prev = velocities[-1]
            R_prev = T_prev[:3, :3]
            a_prev = np.array(self.a_body[i - 1], dtype=float)
            w_prev = np.array(self.w_body[i - 1], dtype=float)
            R_new = R_prev @ SE23.get_SO3_rotation_matrix(w_prev, dt)
            v_new = v_prev + a_prev * dt
            p_new = T_prev[:3, 3] + (R_prev @ v_prev) * dt + 0.5 * (R_prev @ a_prev) * dt ** 2
            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = p_new
            T_global.append(T_new)
            velocities.append(v_new)

        return MeasuredTrajectory(T_global, self.t, v_body=velocities, w_body=self.w_body, a_body=self.a_body)

    def plot_trajectory(self, ax, label="", color="k", linestyle="-", marker="", T_OR=None, v_R0=None):
        self.to_measured_trajectory(T_OR=T_OR, v_R0=v_R0).plot_trajectory(
            ax=ax,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )


class InterRobotDistanceSensor:
    def __init__(self, t=None, d=None, d_true=None, raw_packets=None, packet_decoder=None, range_ids=None):
        self.t = [] if t is None else list(np.array(t, dtype=float))
        if d is None:
            self.d = [None] * len(self.t)
        else:
            self.d = list(np.array(d, dtype=float))
        self.d_true = [] if d_true is None else list(np.array(d_true, dtype=float))
        self.raw_packets = [] if raw_packets is None else list(raw_packets)
        self.range_ids = [] if range_ids is None else list(np.array(range_ids, dtype=int))
        self.packet_decoder = packet_decoder or self.default_packet_decoder

    @staticmethod
    def default_packet_decoder(raw_packet):
        packet = EspPacketParser.parse_packet_string(raw_packet)
        if not packet.ranges:
            raise ValueError(f"Could not decode a distance from raw packet: {raw_packet}")
        return packet.ranges[0].dist_m

    @classmethod
    def from_rosbag(cls, bag_path, uwb_topic="/tb2/uwb", packet_decoder=None, range_id=None):
        times = []
        distances = []
        range_ids = []
        for ros_time, packet in OdometrySensor._esp_packets_from_rosbag(bag_path, uwb_topic):
            if not packet.ranges:
                continue
            range_measurement = None
            if range_id is None:
                range_measurement = packet.ranges[0]
            else:
                for measurement in packet.ranges:
                    if measurement.rid == range_id:
                        range_measurement = measurement
                        break
            if range_measurement is None:
                continue
            times.append(ros_time)
            distances.append(range_measurement.dist_m)
            range_ids.append(range_measurement.rid)
        return cls(t=times, d=distances, packet_decoder=packet_decoder, range_ids=range_ids)

    def get_new_measurement(self, time):
        idx = _find_segment(self.t, time)
        if idx is None:
            print("Time is out of bounds.")
            return None
        if idx >= len(self.t) - 1 or self.t[idx] == time:
            return self.d[idx]
        return float(_interpolate_vector(self.t[idx], self.t[idx + 1], self.d[idx], self.d[idx + 1], time))

    def plot(self, ax=None, label="Measured UWB distance", true_label="True UWB distance"):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt
        if self.d_true:
            ax.plot(self.t[:len(self.d_true)], self.d_true, label=true_label, color="g")
        ax.plot(self.t[:len(self.d)], self.d, label=label, color="b", linestyle="", marker="x")
        ax.legend()
