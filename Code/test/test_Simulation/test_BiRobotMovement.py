import unittest
import Code.Simulation.BiRobotMovement as BRM
from Code.Simulation.RobotClass import NewRobot
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_something(self):
        drone = BRM.drone_flight(np.array([0, 0, 2, np.pi / 4]), sigma_dv=self.sigma_dv, sigma_dw=self.sigma_dw,
                                  max_range=0, origin_bool=True, simulation_time_step=self.odom_time_step)
        host = BRM.drone_flight(np.array([0, 0, 0, 0]), sigma_dv=0.01, sigma_dw=0.001,
                                 max_range=self.max_range, origin_bool=True, simulation_time_step=self.odom_time_step)

        ctrl2d = BRM.Control2D()


        drone.set_start_position(np.array([-10, -1, 0]), 0)
        BRM.run_simulation(time_steps =0.1, host_agent= host, connected_agent=drone,
                           move_function=BRM.fix_connected_2D_host, kwargs={"control2d": ctrl2d})

if __name__ == '__main__':
    unittest.main()
