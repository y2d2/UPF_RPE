import unittest

from upf_rte.Simulation.Trajectories import Trajectory
from upf_rte.Simulation.Sensors import IMU, VIO_2D, IMU_2D
from upf_rte.Simulation.Trajectories_Generator import create_ERB, create_ECB, Velocity_Control_2D, trajectory_generator, random_starting_conditions
import numpy as np
import matplotlib.pyplot as plt
import upf_rte.UtilityCode.SE23 as SE23
import matplotlib.pyplot as plt


def analytical_schuine_worp(angle, v0, g, dt):
    t_flight = (2 * v0 * np.sin(angle)) / g
    x_range = v0 * np.cos(angle) * t_flight
    v_final = np.array([v0 * np.cos(angle), -v0 * np.sin(angle), 0])

    ts = [t for t in np.arange(0, t_flight, dt)]
    xs = [v0 * np.cos(angle) * t for t in ts]
    # vys = [v0 * np.sin(angle) - g.t for t in ts ]
    ys = [v0 * np.sin(angle) * t - 0.5 * g * t ** 2 for t in ts]
    return t_flight, x_range, v_final, ts, xs, ys


class Trajectory_TestCase(unittest.TestCase):
    def test_trajectory(self):
        X_G0 = SE23.SE23_from_w_v_t(np.zeros(3), np.zeros(3), np.array([0,0,0]))


    def test_SE3s(self):
        traj = trajectory_generator(plt_bool=False)

        Ts, ts = traj.export_SE3_trajectory()
        new_traj = Trajectory(np.eye(4), np.zeros(3), time=0)
        new_traj.load_trajectory(Ts, ts)

        ax = plt.figure().add_subplot(111, projection='3d')
        traj.plot_trajectory(ax, label="Original trajectory")
        new_traj.plot_trajectory(ax, label="Exported and reloaded trajectory", color="red",linestyle="--")
        plt.show()

    def test_measurment_streaming(self):
        traj = trajectory_generator(plt_bool=False)




    def test_schuine_worp(self):
        #analystical solution for range of projectile motion:
        angle = np.pi/4
        g = 9.81
        v0 = 10
        dt = 0.01

        t_flight, x_range, v_final, ts, xs, ys = analytical_schuine_worp(angle, v0, g,dt)
        # numerical solution using trajectory class
        T_OR = np.eye(4)
        R = SE23.get_SO3_rotation_matrix(np.array([0,0,angle]))
        T_OR[:3,:3] = R
        V_RR0 = np.array([v0, 0 ,0])
        traj = Trajectory(T_OR, V_RR0, time=0)
        x_nums = []
        y_nums = []
        for t in np.arange(0, t_flight, dt):
            a = np.array([-np.cos(angle)*g,-np.sin(angle)*g, 0])
            w = np.array([0,0,0])
            traj.do_step(a, w, t+dt)
            X_OR = traj.X_OR @ traj.X_RRi[-1]
            x_nums.append(X_OR[0,4])
            y_nums.append(X_OR[1,4])

        plt.plot(x_nums, y_nums)
        plt.plot(xs, ys)
        plt.show()



    def test_ERB(self):
        traj = create_ERB(np.eye(4), np.array([1,0,0]), time=100.0, dt=0.1)
        ax = plt.figure().add_subplot(111, projection='3d')
        traj.plot_trajectory(ax)
        plt.show()

    def test_ECB(self):
        traj = create_ECB(np.eye(4), v_RR0=1, w=1, time=10.0, dt=0.001)
        ax = plt.figure().add_subplot(111, projection='3d')
        traj.plot_trajectory(ax)
        plt.show()

    def test_2D_control_room(self):
        T_OR,v_R0 =  random_starting_conditions()
        traj = trajectory_generator(T_OR = T_OR, v_R0 = v_R0, dict_of_sensors={}, plt_bool=True)

    def test_3D_control_room(self):
        T_OR,v_R0 =  random_starting_conditions()
        traj = trajectory_generator(T_OR = T_OR, v_R0 = v_R0, dict_of_sensors={}, plt_bool=True, control_type='3D')


if __name__ == '__main__':
    # unittest.main()
    # t = Trajectory_TestCase()
    X_OR0 = SE23.SE23_from_w_v_t(np.array([0,0,np.pi/2]), np.array([1,0,0]), np.array([0,0,0]))
    print(X_OR0)
