import numpy as np

import Code.UtilityCode.SE23 as SE23


class Trajectory():
    def __init__(self, T_OR, v_R0, time=0):
        # T_OR is the initial transformation from global to robot frame.
        # v_RR0 is the initial speed of the robot expressed in its body frame.

        # Initial transformation from global to start of trajectory
        self.X_OR = np.eye(5) # Initial transformation from global to robot frame
        self.X_OR[:3,:3] = T_OR[:3,:3]
        self.X_OR[:3,4] = T_OR[:3,3]

        # Initial transformation in body frame of robot.
        self.X_R0 = np.eye(5)
        self.X_R0[0:3, 3] = v_R0
        self.dX_0 = np.zeros((5,5))
        self.dX_0[0:3,3] = v_R0

        # list of SE32 transformations and states.
        self.X_RRi = [self.X_R0]  # List of transformations from start of trajectory to each time step
        self.dX_i = [self.dX_0] # List of changes in transformation at each time step

        self.t = [time]  # List of time stamps

    def get_current_position(self):
        X_ORi = self.X_OR @ self.X_RRi[-1]
        return X_ORi[:3,4]

    def get_global_postion_at_time(self, time):
        # Get the global position at a specific time by interpolating between the two closest time steps.
        if time < self.t[0] or time > self.t[-1]:
            print("Time is out of bounds.")
            return None
        for i in range(len(self.t)-1):
            if self.t[i] <= time <= self.t[i+1]:
                t1 = self.t[i]
                t2 = self.t[i+1]
                X1 = self.X_OR @ self.X_RRi[i]
                X2 = self.X_OR @ self.X_RRi[i+1]
                # Linear interpolation of the position
                alpha = (time - t1) / (t2 - t1)
                position = (1-alpha) * X1[:3,4] + alpha * X2[:3,4]
                return position
        return None

    def get_local_state_at_time(self, time) :
        if time < self.t[0] or time > self.t[-1]:
            print("Time is out of bounds.")

        for i in range(len(self.t)-1):
            if self.t[i] <= time <= self.t[i+1]:
                t1 = self.t[i]
                t2 = self.t[i+1]
                X1 = self.X_RRi[i]
                X2 = self.X_RRi[i+1]
                # Linear interpolation of the position
                alpha = (time - t1) / (t2 - t1)
                position = (1-alpha) * X1[:3,4] + alpha * X2[:3,4]
                velocity = (1-alpha) * X1[:3,3] + alpha * X2[:3,3]

                # Interpolation of angle:
                R1 = X1[:3,:3]
                R2 = X2[:3,:3]
                angle_vect_1 = SE23.get_w_from_SO3(R1)
                angle_vect_2 = SE23.get_w_from_SO3(R2)
                angle_vect = (1-alpha) * angle_vect_1 + alpha * angle_vect_2
                X = SE23.SE23_from_t_v_rot(position, velocity,  angle_vect)
                return position, velocity, angle_vect, X
        return None

    def get_current_velocity(self):
        X_ORi = self.X_OR @ self.X_RRi[-1]
        return X_ORi[:3,3]


    def get_current_orientation(self):
        X_ORi = self.X_OR @ self.X_RRi[-1]
        return SE23.get_w_from_SO3(X_ORi[:3,:3])

    def get_current_state(self):
        X_ORi = self.X_OR @ self.X_RRi[-1]
        position = X_ORi[:3,4]
        velocity = X_ORi[:3,3]
        orientation = SE23.get_w_from_SO3(X_ORi[:3,:3])
        return position, velocity, orientation, self.t[-1]

    def get_current_body_velocity(self):
        # TODO: Check if this is correct.
        X_ORi = self.X_OR @ self.X_RRi[-1]
        X_RiO = SE23.SE23_inverse(X_ORi)
        return X_RiO[:3,3]

    def do_step(self, a, v, w, time):
        # a and w are expressed in the body frame of the robot.
        # a is acceleration, w is angular velocity.
        # If v_bool is true, then a is velocity instead of acceleration
        try:
            dt = time - self.t[-1]
        except IndexError:
            dt = 0
        if dt ==0:
            return a, v, w, time
        if a is not None:
            dX = SE23.SIM23_from_a_w_dt(self.X_RRi[-1], a, w, dt)
            # Calculate v

        else:
            dX, a = SE23.SIM23_from_v_w_dt(self.X_RRi[-1], v, w, dt)
            # Calculate a

        self.t.append(time)
        self.dX_i.append(dX)
        X_RR_new = self.X_RRi[-1] @ dX
        if v is None:
            v = np.transpose(X_RR_new[:3,:3] @ X_RR_new[:3,3])
        self.X_RRi.append(X_RR_new)
        return a, v, w, time

    def load_trajectory(self, T_ORi, time):
        # T_ORi is a list of SE(3) transformation of the trajectory as measured with a VICON or similar.
        self.X_OR = np.eye(5)  # Initial transformation from global to robot frame
        self.X_OR[:3, :3] = T_ORi[0][:3, :3]
        self.X_OR[:3, 4] = T_ORi[0][:3, 3]
        T_R0O = SE23.SE3_inverse(T_ORi[0])
        v_OR0 = (T_ORi[1][:3,3]  - T_ORi[0][:3,3])/(time[1]-time[0])
        v_RR0 = T_R0O[:3,:3] @ v_OR0

        self.X_R0 = np.eye(5)
        self.X_R0[0:3, 3] = v_RR0
        self.dX_0 = np.zeros((5, 5))
        self.dX_0[0:3, 3] = v_RR0

        # list of SE32 transformations and states.
        self.X_RRi = [self.X_R0]  # List of transformations from start of trajectory to each time step
        self.dX_i = [self.dX_0]
        self.t  = [time[0]]

        for T_OR, t in zip(T_ORi[1:], time[1:]):
            T_R0R = T_R0O @ T_OR
            X, dX = SE23.SE23_from_SE3s(self.X_RRi[-1], T_OR, self.t[-1], t)
            self.X_RRi.append(X)
            self.dX_i.append(dX)
            self.t.append(t)

    def export_SE3_trajectory(self):
        T_ORi = []
        for X_RRi in self.X_RRi:
            X_ORi = self.X_OR @ X_RRi
            T = SE23.SE23_trim_to_SE3(X_ORi)
            T_ORi.append(SE23.SE23_trim_to_SE3(X_ORi))
        return T_ORi, self.t

    def stream_body_frame_measurements(self):
        # Streams the measurments expressed in the body frame of the robot, as they would be measured by an IMU or VIO sensor.
        previous_t = None
        for dX, X, t in zip(self.dX_i, self.X_RRi, self.t):
            R = X[:3, :3]
            v = np.transpose(R) @ X[:3, 3]
            a = np.zeros(3)
            w = np.zeros(3)
            if previous_t is not None:
                a = dX[:3, 3] / (t - previous_t)
                w = SE23.SE3_get_rotation_vector(dX[:3, :3]) / (t - previous_t)
            previous_t = t
            yield a, v, w, t

    def get_steps(self):
        a_list = []
        w_list = []
        v_list = []
        t_list = []
        for a, v, w, t in self.stream_body_frame_measurements():
            a_list.append(a)
            w_list.append(w)
            v_list.append(v)
            t_list.append(t)
        return a_list, w_list, v_list, t_list


    def plot_trajectory(self,ax, label="", color="k", mark="",
                        linestyle="-",
                        alpha=1, i=-1, history=None):

        X_ORi = np.array([self.X_OR @ X_RRi for X_RRi in self.X_RRi])
        x = X_ORi[:, 0, 4]
        y = X_ORi[:, 1, 4]
        z = X_ORi[:, 2, 4]
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

    def plot_distance_to_origin(self, ax, label="", format_string="k-"):
        X_ORi = np.array([self.X_OR @ X_RRi for X_RRi in self.X_RRi])
        x = X_ORi[:, 0, 4]
        y = X_ORi[:, 1, 4]
        z = X_ORi[:, 2, 4]
        distance = np.sqrt(x**2 + y**2 + z**2)
        ax.plot(self.t, distance, format_string, label=label)

    def save_trajectory(self, filename):
        np.savez(filename, X_OR=self.X_OR, X_RRi=self.X_RRi, dX_i=self.dX_i, t=self.t)
