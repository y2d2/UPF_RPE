from Code.Simulation.Trajectories import Trajectory



class generated_trajectory():
    def __init__(self, T_OR, v_RR0, time=0):
        self.traj = Trajectory(T_OR, v_RR0, time)

    def do_step(self, a, w, time, v_bool=False):
        self.traj.do_step(a,w,time,v_bool)
