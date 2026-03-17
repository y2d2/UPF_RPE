from Code.Simulation.Trajectories import Trajectory
from Code.Simulation.Sensors import IMU, VIO_2D, IMU_2D
import Code.UtilityCode.SE23 as SE23
import numpy as np
import matplotlib.pyplot as plt

def random_starting_conditions():
    t = np.random.uniform([-5, -5, 0], [5, 5, 0])
    w = np.random.uniform(-1, 1)
    w = np.array([0, 0, w])
    T_OR = SE23.SE3_from_rot_vec_and_trans(w, t)
    v_R0 = np.random.uniform(-1, 1, 3)
    v_R0[-1] = 0

    return T_OR, v_R0

def plot_situation(ax, true_trajectory, targets = None, dict_of_sensors={}):
    ax.cla()
    true_trajectory.plot_trajectory(ax, label="True trajectory")
    if targets is not None:
        ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], color="red", label="Targets")
    # ax.set_xlim(0, contr.l)
    # ax.set_ylim(0, contr.b)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Time: {true_trajectory.t[-1]:.1f} s")
    for sensor in dict_of_sensors.values():
        sensor["sensor"].odom_trajectory.plot_trajectory(ax, color=sensor["color"], linestyle=sensor["linestyle"],
                                                         label=sensor["label"])
    ax.legend()

def trajectory_generator(T_OR = np.eye(4), v_R0 = np.zeros(3), trajectory_time = 100, dt = 0.1, l = 10, b = 10, z = 2,
                         dict_of_sensors = {}, plt_bool = False):

    contr = Velocity_Control_2D(T_OR=T_OR, v_R0=v_R0, dt=dt)
    contr.set_room_parameters(l, b, z)
    contr.set_control_parameters(max_speed=1.0, max_rotspeed=np.pi / 4, max_acceleration=0.2,
                                 k_v=0.4, k_omega=1.0, d_switch=0.2, max_time=20)

    if plt_bool:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    for a, v, w, t in contr.run_control_loop(total_time=trajectory_time):
        if dict_of_sensors:
            for sensor in dict_of_sensors.values():
                sensor["sensor"].get_new_measurement(a, v, w, t)
        if plt_bool:
            plot_situation(ax, contr.traj, targets=np.array(contr.target), dict_of_sensors=dict_of_sensors)
            plt.pause(0.001)

    if plt_bool:
        plt.ioff()
        plot_situation(ax, contr.traj, targets=np.array(contr.target), dict_of_sensors=dict_of_sensors)
        plt.show()
    return contr.traj

def create_still_trajectory(T_OR,  total_time=100, dt=0.1):
    traj = Trajectory(T_OR, np.zeros(3), time=0)
    time_elapsed = 0.0
    while time_elapsed < total_time:
        time_elapsed += dt
        traj.do_step(np.zeros(3), np.zeros(3), np.zeros(3),time_elapsed)
    return traj


def create_ERB(T_OR, v_RR0, time=100, dt=0.1):
    traj = Trajectory(T_OR, v_RR0, time=0)
    for t in np.linspace(0, time, int(time/dt)):
        traj.do_step(np.zeros(3), np.zeros(3),t)
    return traj


def create_ECB(T_OR, v_RR0, w, time=100, dt=0.1):
    w_vec = w * np.array([0,0,1])
    v_RR0_vec = v_RR0 * np.array([1,0,0])
    R = v_RR0 / np.abs(w)
    a = -R * w**2
    a_vec = -np.sign(w) * a * np.array([0, 1, 0])
    traj = Trajectory(T_OR, v_RR0_vec, time=0)
    for t in np.linspace(dt, time, int(time/dt)):
        traj.do_step(a_vec, w_vec,t)
    return traj



class Velocity_Control_2D():
    def __init__(self,T_OR, v_R0, dt=0.01):
        # self.sensor = sensor
        self.traj =  Trajectory(T_OR=T_OR, v_R0= v_R0)
        self.target_height =self.traj.get_current_position()[-1]
        self.target = []
        self.target_theta = [0]
        self.v_max = 0
        self.omega_max  = 0
        self.a_max = 0
        self.k_v = 0
        self.k_omega = 0
        self.d_switch = 0
        self.dt = dt
        self.mission_time = 0
        self.target_deadline = 0
        self.max_time = 0
        self.l = 0
        self.b = 0
        self.z = 0

    def set_room_parameters(self, l, b, z):
        self.l = l
        self.b = b
        self.z = z


    def set_control_parameters(self, max_speed, max_rotspeed, max_acceleration,
                                k_v=1., k_omega=1., d_switch=0.1, max_time = 10. ):
        self.v_max = max_speed
        self.omega_max = max_rotspeed
        self.a_max = max_acceleration
        self.k_v = k_v
        self.k_omega = k_omega
        self.d_switch = d_switch
        self.max_time = max_time

    def set_random_target(self):
        target = np.zeros(3)
        target[-1] = self.target_height
        target[0] = np.random.uniform(-self.l/2, self.l/2)
        target[1] = np.random.uniform(-self.b/2, self.b/2)
        self.target.append(target)
        self.target_theta.append(np.random.uniform(0, 2*np.pi))
        self.target_deadline = np.random.uniform( self.max_time/2, self.max_time)
        self.mission_time = 0

    def run_control_loop(self, total_time=100):
        time_elapsed = 0.0
        self.set_random_target()
        while time_elapsed < total_time:
            time_elapsed += self.dt
            yield self.apply_control()

    def apply_control(self):
        if self.mission_time > self.target_deadline:
            self.set_random_target()

        t_ORi, v_ORi, rot_vec, t  =self.traj.get_current_state()
        v_RiO = self.traj.get_current_body_velocity()
        v_current = np.linalg.norm(v_ORi)
        theta = rot_vec[2]

        e = self.target[-1] - t_ORi
        distance = np.linalg.norm(e)

        # --- Heading to target ---
        theta_goal = np.arctan2(e[1], e[0])
        e_theta = SE23.limit_angle(theta_goal - theta)

        # --- Final orientation error ---
        e_theta_f = SE23.limit_angle(self.target_theta[-1] - theta)

        # --- Check for final condition ---
        # if (distance < self.d_switch and np.abs(e_theta) < np.radians(5) )or self.mission_time > self.target_deadline:

        # --- Angular velocity command ---
        if distance > self.d_switch:
            omega_cmd = self.k_omega * e_theta
            k_v = self.k_v
        else:
            omega_cmd = self.k_omega * e_theta_f
            k_v = self.k_v /10

        omega_cmd = max(-self.omega_max, min(omega_cmd, self.omega_max))

        # --- Linear velocity command (distance-based) ---
        v_des = k_v * distance

        # Reduce speed if not facing target


        # No reverse driving (remove if you want reverse)
        v_cmd_raw = max(0.0, v_des)

        # Speed limit
        v_cmd_raw = min(v_cmd_raw, self.v_max)
        v_cmd_raw = v_cmd_raw * np.cos(e_theta)
        # --- Acceleration limiting ---
        dv_max = self.a_max * self.dt
        v_cmd = max(
            v_current - dv_max,
            min(v_cmd_raw, v_current + dv_max)
        )
        # --- Damping ---
        v_y = -0.1 * v_RiO[1]
        self.mission_time += self.dt
        a, v, w, time = self.traj.do_step(None, np.array([v_cmd, v_y,0]), np.array([0,0,omega_cmd]), t+self.dt)
        return a, v, w, time