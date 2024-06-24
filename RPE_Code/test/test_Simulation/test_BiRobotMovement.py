import unittest
import RPE_Code.Simulation.BiRobotMovement as BRM
from RPE_Code.Simulation.RobotClass import NewRobot
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

        ctrl2d = BRM.Control2D(agent=host)


        drone.set_start_position(np.array([-10, -1, 0]), 0)
        BRM.run_simulation(time_steps = simulation_time_steps, host_agent= host, connected_agent=drone,
                           move_function=BRM.fix_connected_2D_host, kwargs={"control2d": ctrl2d})

        ax = plt.axes(projection="3d")
        host.plot_real_position(ax)
        drone.plot_real_position(ax)
        plt.show()

if __name__ == '__main__':
    unittest.main()
