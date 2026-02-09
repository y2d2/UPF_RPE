import numpy as np
import Code.UtilityCode.SE23 as SE23


class Trajectory():
    def __init__(self, T_OR, v_RR0, time=0):
        # T_OR is the initial transformation from global to robot frame.
        # v_RR0 is the initial speed of the robot expressed in its body frame.

        # Initial transformation from global to start of trajectory
        self.X_OR = np.eye(5) # Initial transformation from global to robot frame
        self.X_OR[:3,:3] = T_OR[:3,:3]
        self.X_OR[:3,4] = T_OR[:3,3]

        # Initial transformation in body frame of robot.
        self.X_RR0 = np.eye(5)
        self.X_RR0[0:3,3] = v_RR0
        self.dX_0 = np.zeros((5,5))
        self.dX_0[0:3,3] = v_RR0

        # list of SE32 transformations and states.
        self.X_RRi = [self.X_RR0]  # List of transformations from start of trajectory to each time step
        self.dX_i = [self.dX_0] # List of changes in transformation at each time step

        self.t = [time]  # List of time stamps

    def do_step(self, a, w, time, v_bool=False):
        # a and w are expressed in the body frame of the robot.
        # a is acceleration, w is angular velocity.
        # If v_bool is true, then a is velocity instead of acceleration
        try:
            dt = time - self.t[-1]
        except IndexError:
            dt = 0
        if v_bool:
            dX = SE23.SIM23_from_v_w_dt(self.X_RRi[-1], a, w, dt)
        else:
            dX = SE23.SIM23_from_a_w_dt(self.X_RRi[-1], a, w, dt)
        self.t.append(time)
        self.dX_i.append(dX)
        X_RR_new = self.X_RRi[-1] @ dX
        self.X_RRi.append(X_RR_new)

    def load_trajectory(self, T_ORi, time):
        # T_ORi is a list of SE(3) transformation of the trajectory as measured with a VICON or similar.
        self.X_OR = np.eye(5)  # Initial transformation from global to robot frame
        self.X_OR[:3, :3] = T_ORi[0][:3, :3]
        self.X_OR[:3, 4] = T_ORi[0][:3, 3]
        T_R0O = SE23.SE3_inverse(T_ORi[0])
        v_OR0 = (T_ORi[1][:3,3]  - T_ORi[0])/(time[1]-time[0])
        v_RR0 = T_R0O[:3,:3] @ v_OR0

        self.X_RR0 = np.eye(5)
        self.X_RR0[0:3, 3] = v_RR0
        self.dX_0 = np.zeros((5, 5))
        self.dX_0[0:3, 3] = v_RR0

        # list of SE32 transformations and states.
        self.X_RRi = [self.X_RR0]  # List of transformations from start of trajectory to each time step
        self.dX_i = [self.dX_0]
        self.t  = [time[0]]

        for T_OR, t in zip(T_ORi[1:], time):
            T_R0R = T_R0O @ T_OR
            X, dX = SE23.SE23_from_SE3s(self.X_RRi[-1], T_OR, self.t[-1], t)
            self.X_RRi.append(X)
            self.dX_i.append(dX)
            self.t.append(t)

    def steam_steps(self):
        previous_t = None
        for dX, t in zip(self.dX_i, self.t):
            if previous_t is not None:
                a = dX[:3, 3] / (t - previous_t)
                v = dX[:3, 4] / (t - previous_t)
                w = SE23.SE3_get_rotation_vector(dX[:3, :3]) / (t - previous_t)
                previous_t = t
                yield a, v, w, t
            else:
                previous_t = t
                yield None, None, None, t

    def get_steps(self):
        a_list = []
        w_list = []
        v_list = []
        t_list = []
        for a, v, w, t in self.steam_steps():
            a_list.append(a)
            w_list.append(w)
            v_list.append(v)
            t.append(t)
        return a_list, w_list, v_list, t_list



    def plot_trajectory(self,ax, label="", color="k", mark="",
                        linestyle="-",
                        alpha=1, i=-1, history=None):
        T_Gi = np.array(self.T_G0i)
        x = T_Gi[:, 0, 3]
        y = T_Gi[:, 1, 3]
        z = T_Gi[:, 2, 3]
        if history is None or history > i:
            j=0
        else:
            j = i - history
            if j < 0:
                j = 0
        ax.plot3D(x[j:i], y[j:i], z[j:i], label=label,
                  color=color, marker=mark, linestyle=linestyle, alpha=alpha)
        stems = ax.stem([x[i]], [y[i]], [z[i]],
                        basefmt=f"^", linefmt=f":", bottom=0, markerfmt="")
        stems.stemlines.set_color(color)
        stems.baseline.set_color(color)
        stems.markerline.set_color(color)

class OdometrySensor():
    def __init__(self, T_OR, v_RR0, time, velocity_bool=False):
        self.true_trajectory = Trajectory(T_OR, v_RR0, time)
        self.odom_trajectory = Trajectory(T_OR, v_RR0, time)

        # Defines whether the odometry sensor receives velocity or acceleration inputs
        self.velocity_bool = velocity_bool

    def sensor_measurement(self,a, w, time):
        print("Please create the Sensor specific behavior.")
        a_odom = a
        w_odom = w
        return a_odom, w_odom

    def get_new_measurement(self, a, w, time):
        # It is assumed that v and w are given in the body frame of the robot.
        self.true_trajectory.do_step(a, w, time, v_bool=self.velocity_bool)
        a_odom, w_odom = self.sensor_measurement(a, w, time)
        self.odom_trajectory.do_step(a_odom, w_odom, time, self.velocity_bool)

    def get_measurement_from_true_trajectory(self):
        self.odom_trajectory = Trajectory(self.true_trajectory.X_OR, self.true_trajectory.X_RR0, self.true_trajectory.t[0])
        for a, v, w, t in self.true_trajectory.steam_steps():
            if a is not None:
                if self.velocity_bool:
                    a = v
                a_odom, w_odom = self.sensor_measurement(a, w, t)
                self.odom_trajectory.do_step(a_odom, w_odom, t, self.velocity_bool)

class VIO(OdometrySensor):
    def __init__(self, T_OR, v_RR0, time,  sig_w, sig_v):
        super().__init__(T_OR, v_RR0, time, velocity_bool=True)

        #VIO noise parameters
        self.sig_w = sig_w
        self.sig_v = sig_v

    def sensor_measurement(self, a, w, time):
        a_noise = a + np.random.randn(3) * self.sig_v
        w_noise = w + np.random.randn(3) * self.sig_w
        return a_noise, w_noise

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

    def sensor_measurement(self, a, w, time):
        a_imu = a + self.b_acc[-1] + np.random.randn(3) * self.sig_a
        w_imu = w + self.b_gyro[-1] + np.random.randn(3) * self.sig_w
        # random walk of biases:
        self.b_acc.append(self.b_acc[-1] + np.random.randn(3) * self.sig_ba)
        self.b_gyro.append(self.b_gyro[-1] + np.random.randn(3) * self.sig_bw)
        return a_imu, w_imu

