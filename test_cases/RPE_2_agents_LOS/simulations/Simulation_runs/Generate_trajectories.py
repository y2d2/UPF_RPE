
import Code.Simulation.MultiRobotClass as MRC

if __name__ == "__main__":
    folder_name = '../robot_trajectories/Standard'
    MRS = MRC.MultiRobotSimulation()
    MRS.set_simulation_parameters(max_v=1, max_w=0.05, slowrate_v=0.02, slowrate_w=0.005,
                                  simulation_time_step=0.2, simulation_time=500,
                                  number_of_drones=2, max_range=25, range_origin_bool=True,
                                  trajectory_folder_name=folder_name, reset_trajectories=True)
    MRS.create_trajectories(50)