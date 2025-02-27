import unittest
import Code.Simulation.BiRobotMovement as BRM
from Code.Simulation.RobotClass import NewRobot
import numpy as np
import matplotlib.pyplot as plt
class MyTestCase(unittest.TestCase):
    def test_2D_control(self):
        sigma_dv = 0.01
        sigma_dw = 0.001
        odom_time_step = 0.1
        max_range = 5
        simulation_time_steps = 1000
        drone = BRM.drone_flight(np.array([0, 0, 2, np.pi / 4]), sigma_dv=sigma_dv, sigma_dw=sigma_dw,
                                  max_range=0, origin_bool=True, simulation_time_step=odom_time_step)
        host = BRM.drone_flight(np.array([0, 0, 0, 0]), sigma_dv=sigma_dv, sigma_dw=sigma_dw,
                                 max_range=max_range, origin_bool=True, simulation_time_step=odom_time_step)

        ctrl2d = BRM.Control1D(agent=host)


        drone.set_start_position(np.array([-10, -1, 0]), 0)
        BRM.run_simulation(time_steps = simulation_time_steps, host_agent= host, connected_agent=drone,
                           move_function=BRM.fix_connected_2D_host, kwargs={"control2d": ctrl2d})

        ax = plt.axes(projection="3d")
        host.plot_real_position(ax)
        drone.plot_real_position(ax)
        plt.show()

    def test_3D_control(self):
        sigma_dv = 0.01
        sigma_dw = 0.001
        odom_time_step = 0.1
        max_range = 25
        simulation_time_steps = 10000
        drone = BRM.drone_flight(np.array([0, 0, 0, np.pi / 4]), sigma_dv=sigma_dv, sigma_dw=sigma_dw,
                                  max_range=0, origin_bool=True, simulation_time_step=odom_time_step)
        host = BRM.drone_flight(np.array([0, 0, 0, 0]), sigma_dv=sigma_dv, sigma_dw=sigma_dw,
                                 max_range=max_range, origin_bool=True, simulation_time_step=odom_time_step)

        control_host = BRM.Control3D(agent=host,max_v = 1., max_w = 1., max_dot_v=0.2, max_dot_w=0.5)
        control_host.radius = 25
        control_host.p_pos = 0.06
        control_host.height = 20
        control_host.target_time_max=2000

        ctrld_con = BRM.Control3D(agent=drone,max_v = 1., max_w = 1., max_dot_v=0.2, max_dot_w=0.5)
        ctrld_con.radius = 25
        ctrld_con.p_pos = 0.06
        ctrld_con.height = 20
        ctrld_con.target_time_max=2000


        drone.set_start_position(np.array([0, 0, 0]), 0)
        host.set_start_position(np.array([0, 0, 0]), 0)

        BRM.run_simulation(time_steps = simulation_time_steps, host_agent= host, connected_agent=drone,
                           move_function=BRM.both_3D_control, kwargs={"control_host": control_host, "control_connected": ctrld_con})


        ax = plt.axes(projection="3d")
        host.plot_real_position(ax)
        drone.plot_real_position(ax)

        host.set_plotting_settings(color='r')
        drone.set_plotting_settings(color='b')
        ax = host.plot_projections()
        drone.plot_projections(ax)

        plt.show()
if __name__ == '__main__':
    unittest.main()
