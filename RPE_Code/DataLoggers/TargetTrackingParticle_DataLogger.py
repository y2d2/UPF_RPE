import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from RPE_Code.DataLoggers.TargetTrackingUKF_DataLogger import UKFDatalogger
from RPE_Code.DataLoggers.NLS_DataLogger import NLSDataLogger
from RPE_Code.Simulation.RobotClass import NewRobot
from RPE_Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent, TargetTrackingParticle
import copy

class TargetTrackingParticle_DataLogger:
    def __init__(self, hostAgent: NewRobot, connectedAgent: NewRobot, particle: TargetTrackingParticle, parent =None):
        self.particle = particle
        self.rpea_datalogger = None
        self.weight = []
        self.likelihood = []
        if parent is not None:
            self.weight = copy.deepcopy(parent.weight)
            self.likelihood = copy.deepcopy(parent.likelihood)

    def log_data(self, i):
        self.rpea_datalogger.log_data(i)
        self.weight.append(self.particle.weight)
        self.likelihood.append(self.particle.likelihood)

    def plot_self(self, particle_ax : plt.Axes | None = None, los = None):
        if particle_ax is None:
            particle_ax = plt.figure().subplots(1, 1)
        particle_ax.plot(self.likelihood, label="Likelihood")
        particle_ax.plot(self.weight, label="Weigth")
        if los is not None:
            particle_ax.plot(los, color="k", label="Real LOS State")

        particle_ax.legend()
        particle_ax.grid(True)
        particle_ax.set_title("LOS state and likelihood best particle.")



class UKFTargetTrackingParticle_DataLogger(TargetTrackingParticle_DataLogger):
    def __init__(self, hostAgent: NewRobot, connectedAgent: NewRobot, particle: TargetTrackingParticle, parent =None):
        super().__init__(hostAgent, connectedAgent, particle, parent=parent)
        self.rpea_datalogger = UKFDatalogger(hostAgent, connectedAgent, particle.rpea)
        if parent is not None:
            self.rpea_datalogger = parent.rpea_datalogger.copy(particle.rpea)

class NLSTargetTrackingParticle_DataLogger(TargetTrackingParticle_DataLogger):
    def __init__(self,  hostAgent: NewRobot, connectedAgent: NewRobot, particle: TargetTrackingParticle, parent =None):
        super().__init__(hostAgent, connectedAgent, particle, parent=parent)
        self.rpea_datalogger = NLSDataLogger(particle.rpea)
        if parent is not None:
            self.rpea_datalogger = parent.rpea_datalogger.copy(particle.rpea)
