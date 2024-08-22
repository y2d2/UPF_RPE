import unittest
from Code.Simulation.MultiRobotClass import MultiRobotSingleSimulation

class MultiRobotSingleSimulation_TestCase(unittest.TestCase):
    def test_open_sim(self):
        sim_nr = 0
        sim_folder = "Simulations/sim_" + str(sim_nr)
        sim = MultiRobotSingleSimulation(sim_folder)
        sim.load_simulation(sigma_v =0.1, sigma_w = 0.01, sigma_d = 0.1)
        return sim
        # sim.run_simulation(self.sigma_dv, self.sigma_dw, self.sigma_uwb)
        # self.factor = int(self.uwb_rate / self.sim.parameters["simulation_time_step"])
        # if self.plot_bool:
        #     self.sim.init_plot(interactive=True)



if __name__ == '__main__':
    # unittest.main()
    t = MultiRobotSingleSimulation_TestCase()
    sim = t.test_open_sim()