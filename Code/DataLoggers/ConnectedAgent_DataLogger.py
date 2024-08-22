import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Code.DataLoggers.TargetTrackingUKF_DataLogger import UKFDatalogger
from Code.Simulation.RobotClass import NewRobot
from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent, TargetTrackingParticle
from Code.DataLoggers.TargetTrackingParticle_DataLogger import TargetTrackingParticle_DataLogger, UKFTargetTrackingParticle_DataLogger
import copy


class UPFConnectedAgentDataLogger:
    def __init__(self, host_agent: NewRobot, connected_agent: NewRobot, upf_connected_agent: UPFConnectedAgent, particle_type):
        self.particle_type = particle_type
        self.host_agent = host_agent
        self.connected_agent = connected_agent
        self.upf_connected_agent = upf_connected_agent
        self.particle_count = 0
        self.particle_logs = []
        self.init_variables()
        self.i = 0

        # Host agent variables:
        self.number_of_particles = []
        self.ha_pose_stds = np.zeros((0, 4))
        self.sigma_x_ha = []
        self.dx_ha_stds = np.zeros((0, 4))
        self.dx_drift = np.zeros((0, 4))

        # Timing variables:
        self.calulation_time = []

    def find_particle_log(self, particle) -> TargetTrackingParticle_DataLogger:
        for particle_log in self.particle_logs:
            if particle_log.particle == particle:
                return particle_log
        self.add_particle(particle)
        return self.particle_logs[-1]

    def get_best_particle_log(self) -> TargetTrackingParticle_DataLogger:
        return self.find_particle_log(self.upf_connected_agent.best_particle).rpea_datalogger

    def add_particle(self, particle):
        # particle.set_datalogger(self.host_agent, self.connected_agent, name="Particle " + str(self.particle_count))
        parent_log = None
        if particle.parent is not None:
            parent_log = self.find_particle_log(particle.parent)
        particle_log = self.particle_type(self.host_agent, self.connected_agent, particle, parent = parent_log)
        self.particle_count += 1
        self.particle_logs.append(particle_log)

    def init_variables(self):
        for particle in self.upf_connected_agent.particles:
            self.add_particle(particle)
            # self.ukfs_logger.append(Datalogger(self.host_agent, self.connected_agent, particle, name="Particle "+ str(i)))

    def log_data(self, i, calculation_time=0):
        self.calulation_time.append(calculation_time)
        if self.upf_connected_agent.time_i is None:
            self.i = i
        else:
            self.i = self.upf_connected_agent.time_i
        self.log_ha_data()
        for particle in self.upf_connected_agent.particles:
            particle_log: TargetTrackingParticle_DataLogger= self.find_particle_log(particle)
            particle_log.log_data(i)


    def log_ha_data(self):
        self.number_of_particles.append(len(self.upf_connected_agent.particles))

        ha_stds = np.sqrt(np.diag(self.upf_connected_agent.ha.p_x_ha))
        self.ha_pose_stds = np.concatenate((self.ha_pose_stds, ha_stds.reshape(1, 4)), axis=0)
        self.sigma_x_ha.append(self.upf_connected_agent.sigma_x_ha)
        dx_ha_stds = np.diag(self.upf_connected_agent.ha.p_dx_ha) # np.sqrt(np.diag(self.upf_connected_agent.ha.p_dx_ha))
        self.dx_ha_stds = np.concatenate((self.dx_ha_stds, dx_ha_stds.reshape(1, 4)), axis=0)
        dx_ha = self.upf_connected_agent.ha.dx_drift
        self.dx_drift = np.concatenate((self.dx_drift, dx_ha.reshape(1, 4)), axis=0)

    # ---- Plot functions
    def plot_poses(self, ax, color_ha, color_ca,  name_ha, name_ca):
        self.plot_estimated_trajectory(ax, color=color_ca, name=name_ca)
        self.host_agent.set_plotting_settings(color=color_ha)
        self.host_agent.plot_real_position(ax, annotation=None)


    def plot_start_poses(self, ax):
        # plt.figure()
        # ax = plt.axes(projection="3d")
        self.plot_estimated_trajectory(ax, color ="r")
        # self.plot_connected_agent_trajectory(ax)
        self.plot_best_particle(ax, color="gold", alpha=1)
        self.plot_connected_agent_trajectories(ax, color="k", i = self.i)
        self.plot_host_agent_trajectory(ax, color="darkgreen", i=self.i)

        ax.legend()

    def plot_host_agent_trajectory(self, ax, color="darkgreen", name = "Host Agent", i=-1,  history=None):
        self.host_agent.set_plotting_settings(color=color)
        self.host_agent.plot_real_position(ax, annotation=name, i=i, history=history)

    def plot_connected_agent_trajectories(self, ax, color="k", i=-1):
        # self.plot_estimated_trajectory(ax, color=color, alpha=0.1)
        self.plot_connected_agent_trajectory(ax, color=color, alpha=1, i=i)
        self.plot_best_particle(ax, color=color, alpha=0.5)

    def plot_connected_agent_trajectory(self, ax, color="black", alpha=1., i=-1, history=None):
        self.connected_agent.set_plotting_settings(color=color)
        self.connected_agent.plot_real_position(ax, annotation="Connected Agent", alpha=alpha, i=i, history=history)
        # self.connected_agent.plot_slam_position(ax, annotation="Connected Agent SLAM", alpha=alpha)

    def plot_estimated_trajectory(self, ax, color="k", alpha=0.1, name = "connected agent"):
        for particle_log in self.particle_logs:
            particle_log.rpea_datalogger.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=0.1,linestyle=":", label=None)
        for particle in self.upf_connected_agent.particles:
            particle_log = self.find_particle_log(particle)
            particle_log.rpea_datalogger.plot_ca_corrected_estimated_trajectory(ax, color=color,  alpha=1, label=None)

    def plot_self(self, los=None, host_id="No host id"):
        bp_dl: TargetTrackingParticle_DataLogger =self.find_particle_log(self.upf_connected_agent.best_particle)
        fig = plt.figure(figsize=(18, 10))  # , layout="constrained")
        fig.suptitle("Host Agent: " + host_id + "; Connected agent: " + self.upf_connected_agent.id)
        ax = []
        gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        ax_3d = fig.add_subplot(gs[:3, :3], projection="3d")
        self.plot_start_poses(ax_3d)

        # ---- Best Particle Axis
        ax_best_particle = [fig.add_subplot(gs[i, -1]) for i in range(2)]
        # ax_best_particle[0].set_title("Best Particle")
        try:
            bp_dl.rpea_datalogger.plot_ukf_drift(ax_best_particle)
            ax_best_particle[0].legend(loc="upper left")
        except AttributeError:
            pass

        # ---- Host agent Axis
        ha_ax = fig.add_subplot(gs[2, -1])
        ha_ax.plot(self.sigma_x_ha, label="std of the host agent trajectory ha [m]")
        label = ["x [m]", "y [m]", "z [m]", "h [(rad)]"]
        ha_ax.set_ylabel("std of dx of the host agent")
        for i in range(4):
            ha_ax.plot(self.ha_pose_stds[:, i], label=label[i])
        ha_ax.grid(True)
        ha_ax.legend()

        dx_ha_ax = fig.add_subplot(gs[3, -1])
        label = ["x [m]", "y [m]", "z [m]", "h [(rad)]"]
        dx_ha_ax.set_xlabel("std of dx of the host agent")
        for i in range(4):
            dx_ha_ax.plot(self.dx_ha_stds[:, i], label=label[i])
        dx_ha_ax.legend()

        dx_ha_ax_1 = fig.add_subplot(gs[3, 2])
        label = ["x [m]", "y [m]", "z [m]", "h [(rad)]"]
        dx_ha_ax_1.set_xlabel("dx drift of the host agent")
        for i in range(4):
            dx_ha_ax_1.plot(self.dx_drift[:, i], label=label[i])
        dx_ha_ax_1.legend()

        # ---- Particle Axis
        particle_ax = fig.add_subplot(gs[3, 0])
        particle_ax.plot(self.number_of_particles)
        particle_ax.set_title("Number of particles")
        particle_ax.grid(True)

        likelihood_ax = fig.add_subplot(gs[3, 1])

        bp_dl.plot_self(particle_ax=likelihood_ax, los=los)

        return fig

    def plot_best_particle(self, ax, color="gold", alpha=1., history=None):
        # print(self.upf_connected_agent.best_particle)
        best_particle_log = self.find_particle_log(self.upf_connected_agent.best_particle).rpea_datalogger
        best_particle_log.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=alpha,
                                                                   label="Best Particle",  history=history)

    def plot_best_particle_variance_graph(self):
        best_particle_log = self.find_particle_log(self.upf_connected_agent.best_particle).rpea_datalogger
        best_particle_log.plot_error_graph()
        for particles in self.upf_connected_agent.particles:
            particle_log = self.find_particle_log(particles).rpea_datalogger
            particle_log.plot_error_graph()

    def plot_connected_agent(self, ax):
        bp_dl = self.find_particle_log(self.upf_connected_agent.best_particle).rpea_datalogger
        bp_dl.plot_ukf_drift(ax[:2])

        likelihood_ax = ax[2]
        likelihood_ax.plot(bp_dl.likelihood, color="darkred", label="Likelihood")
        likelihood_ax.plot(bp_dl.weight, color="darkblue", label="Weigth")
        # likelihood_ax.plot(bp_dl.los_state, color="darkgreen", label="LOS state")
        likelihood_ax.plot(0, color="k", label="# particles")
        likelihood_ax.legend()
        likelihood_ax.grid(True)

        tw = likelihood_ax.twinx()
        tw.plot(self.number_of_particles, color="k")


    def plot_ca_best_particle(self, ax, i, color, history):
        bp_dl = self.find_particle_log(self.upf_connected_agent.best_particle).rpea_datalogger
        bp_dl.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=1, label=None, i=i, history=history)

    def plot_ca_active_particles(self, ax, i, color, history):
        active_particles = []
        for par_log in self.particle_logs:
            if par_log.i > i:
                # active_particles.append(par_log)
                par_log.plot_ca_corrected_estimated_trajectory(ax, color=color, alpha=1, label=None, i =i, history=history)
                # par_log.datalogger.plot_ca_estimated_trajectory(ax, color="b", alpha=0.3, label=None, i = int(i/10)+1)
        # self.plot_connected_agent_trajectory(ax, i = i)


        # fig = plt.figure(figsize=(18, 10), projection="3d")