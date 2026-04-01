import numpy as np
import Code.UtilityCode.SE23 as SE23
from Code.Simulation.Trajectories import Trajectory

class OdometrySensor():
    def __init__(self, T_OR, v_RR0, time, velocity_bool=False):
        self.true_trajectory = Trajectory(T_OR, v_RR0, time)
        self.odom_trajectory = Trajectory(T_OR, v_RR0, time)

        # Defines whether the odometry sensor receives velocity or acceleration inputs
        self.velocity_bool = velocity_bool

    def sensor_measurement(self,a, v, w, time):
        print("Please create the Sensor specific behavior.")
        a_odom = a
        w_odom = w
        v_odom = v
        return a_odom, v_odom, w_odom

    def get_new_measurement(self, a, v, w, time):
        # It is assumed that v and w are given in the body frame of the robot.
        # If a is provided, then this will be used.
        if a is None and v is None:
            print("Please provide either acceleration or velocity input.")
            return
        a_new, v_new, w_new, time_new = self.true_trajectory.do_step(a, v, w, time)
        a_odom,v_odom, w_odom = self.sensor_measurement(a_new,v_new, w_new, time_new)
        self.odom_trajectory.do_step(a_odom, v_odom, w_odom, time)


    def get_measurement_from_true_trajectory(self):
        self.odom_trajectory = Trajectory(self.true_trajectory.X_OR, self.true_trajectory.X_R0, self.true_trajectory.t[0])
        for a, v, w, t in self.true_trajectory.stream_body_frame_measurements():
            a_odom, v_odom, w_odom = self.sensor_measurement(a, v, w, t)
            self.odom_trajectory.do_step(a_odom, v_odom, w_odom, t)

class VIO(OdometrySensor):
    def __init__(self, T_OR, v_RR0, time,  sig_w, sig_v):
        super().__init__(T_OR, v_RR0, time, velocity_bool=True)
        #VIO noise parameters
        self.sig_w = sig_w
        self.sig_v = sig_v

    def sensor_measurement(self, a, v, w, time):
        v_noise = v + np.random.randn(3) * self.sig_v
        w_noise = w + np.random.randn(3) * self.sig_w
        return None, v_noise, w_noise

class IMU(OdometrySensor):
    def __init__(self,T_OR, v_RR0, time,  sig_w, sig_a, sig_ba, sig_bw):
        super().__init__(T_OR, v_RR0, time, velocity_bool=False)

        # IMU noise parameters
        self.sig_w = sig_w
        self.sig_a = sig_a

        # Slow chanign biases
        self.sig_ba = sig_ba
        self.sig_bw = sig_bw

        # Randomised bias for IMU
        self.b_acc = [np.random.randn(3) * 1.0]
        self.b_gyro = [np.random.randn(3) * 1.0]

    def sensor_measurement(self, a, v, w, time):
        a_imu = a + self.b_acc[-1] + np.random.randn(3) * self.sig_a
        w_imu = w + self.b_gyro[-1] + np.random.randn(3) * self.sig_w
        # random walk of biases:
        self.b_acc.append(self.b_acc[-1] + np.random.randn(3) * self.sig_ba)
        self.b_gyro.append(self.b_gyro[-1] + np.random.randn(3) * self.sig_bw)
        return a_imu, None, w_imu


class IMU_2D(IMU):
    def sensor_measurement(self, a, v, w, time):
        a_imu, _, w_imu = super().sensor_measurement(a, v, w, time)
        a_imu[2] = 0
        w_imu[0:2] = 0
        return a_imu, None, w_imu

class VIO_2D(VIO):
    def sensor_measurement(self, a, v, w, time):
        _, v_imu, w_imu = super().sensor_measurement(a, v, w, time)
        v_imu[2] = 0
        w_imu[0:2] = 0
        return None, v_imu, w_imu

##################################################################################
###  INTER ROBOT DISTANCE SENSOR
##################################################################################
class InterRobotDistanceSensor():
    def __init__(self, traj_1: Trajectory, traj_2: Trajectory, sig_d):
        self.traj_1 = traj_1
        self.traj_2 = traj_2
        self.sig_d = sig_d
        self.t = []
        self.d = []
        self.d_true = []

    def get_new_measurement(self, time):
        p1 = self.traj_1.get_global_postion_at_time(time)
        p2 = self.traj_2.get_global_postion_at_time(time)

        d = np.linalg.norm(p1 - p2)
        self.d_true.append(d)
        d_noise = d + np.random.randn() * self.sig_d
        self.t.append(time)
        self.d.append(d_noise)
        return d_noise

    def plot(self, ax):
        ax.plot(self.t, self.d_true, color="g", label="True UWB distance")
        ax.plot(self.t, self.d, "--b", label="Measured UWB distance")
