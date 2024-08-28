import unittest
import os
import Code.Simulation.MultiRobotClass
from Code.BaseLines import NLS, NLSDataLogger
import numpy as np
import pickle as pkl

from Code.UtilityCode import Measurement

from Code.Simulation import NewRobot
import matplotlib

from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
class MyTestCase(unittest.TestCase):
    def create_experimental_data(self, data_folder, sig_v, sig_w, sig_uwb):
        experiments = []
        measurements = []
        # check wether data_folder is a file or a folder

        if os.path.isfile(data_folder):
            list_of_files = [data_folder]
        else:
            list_of_files = os.listdir(data_folder)
        for sampled_data in list_of_files:
            name = sampled_data.split(".")[-2].split("/")[-1]
            measurement = Measurement()
            measurement.load_sampled_data(sampled_data)
            sample_freq = measurement.sample_frequency

            #
            sig_d = sig_v / sample_freq
            sig_phi = sig_w / sample_freq
            Q_vio = np.diag([sig_d ** 2, sig_d ** 2, sig_d ** 2, sig_phi ** 2])

            # measurement.get_uwb_distances()
            uwb = measurement.uwb.sampled_d
            uwb_los = measurement.get_uwb_LOS(sig_uwb)
            DT_vio_tb2 = measurement.tb2.vio_frame.get_relative_motion_in_T()
            DT_vio_tb3 = measurement.tb3.vio_frame.get_relative_motion_in_T()
            T_vicon_tb2 = measurement.tb2.vicon_frame.sampled_T
            T_vicon_tb3 = measurement.tb3.vicon_frame.sampled_T

            experiment_data = {}
            experiment_data["name"] = name
            experiment_data["sample_freq"] = sample_freq
            experiment_data["drones"] = {}
            experiment_data["drones"]["drone_0"] = {"DT_slam": DT_vio_tb2, "T_real": T_vicon_tb2, "Q_slam": Q_vio}
            experiment_data["drones"]["drone_1"] = {"DT_slam": DT_vio_tb3, "T_real": T_vicon_tb3, "Q_slam": Q_vio}
            experiment_data["uwb"] = uwb
            experiment_data["los_state"] = uwb_los

            measurements.append(measurement)
            # experiment_data["eps_d"] = np.abs(measurement.uwb.real_d - measurement.uwb.sampled_d)

            experiments.append(experiment_data)
        return experiments, measurements

    def create_experiment(self, results_folder, sig_v, sig_w, sig_uwb):
        alpha = 1
        kappa = -1.
        beta = 2.
        n_azimuth = 4
        n_altitude = 3
        n_heading = 4

        tas = Code.Simulation.MultiRobotClass.TwoAgentSystem(trajectory_folder="./", result_folder=results_folder)
        tas.debug_bool = True
        tas.plot_bool = True
        tas.set_ukf_properties(kappa=kappa, alpha=alpha, beta=beta, n_azimuth=n_azimuth, n_altitude=n_altitude,
                               n_heading=n_heading)
        tas.set_uncertainties(sig_v, sig_w, sig_uwb)
        return tas

    def test_run_LOS_exp(self):
        sig_v = 0.15
        sig_w = 0.05
        sig_uwb = 0.25

        main_folder = "./exp1_unobservable_sampled.pkl" # seems to be oke trajectory for presentation purposes
        # main_folder = "./exp1_sec2_los_sampled.pkl"
        results_folder ="./Real_Exp_test/"
        data_folder = main_folder

        experiment_data, _ = self.create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
        tas = self.create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool= False
        tas.set_save_results("./presentation")
        # tas.run_experiment(methods=[ "NLS", "algebraic", "upf", "losupf", "nodriftupf"], redo_bool=False, experiment_data=experiment_data)
        tas.run_experiment(methods=["losupf"], redo_bool=False, experiment_data=experiment_data)
        plt.show()
        return tas


    def run_nls_test(self, agents, NLS, nls_logger,uwb_d, time_steps, factor=10):
        host = agents["drone_0"]
        drone = agents["drone_1"]
        dx_ca = np.zeros(4)
        q_ca = np.zeros((4, 4))
        dx_ha = np.zeros(4)
        q_ha = np.zeros((4, 4))
        for j in range(0, time_steps-1):
            print("Time step: ", j)
            c_ca = get_4d_rot_matrix(dx_ca[-1])
            dx_ca = dx_ca + c_ca @ np.concatenate((drone.dx_slam[j], np.array([drone.dh_slam[j]])))
            q_ca = q_ca + c_ca @ drone.q @ c_ca.T

            c_ha = get_4d_rot_matrix(dx_ha[-1])
            dx_ha = dx_ha + c_ha @ np.concatenate((host.dx_slam[j], np.array([host.dh_slam[j]])))
            q_ha = q_ha + c_ha @ host.q @ c_ha.T

            if j % factor == 0:
                dx = np.vstack([dx_ha.reshape(1, *dx_ha.shape), dx_ca.reshape(1, *dx_ca.shape)])
                q_odom = np.vstack([q_ha.reshape(1, *q_ha.shape), q_ca.reshape(1, *q_ca.shape)])
                d = np.array([[0, uwb_d[j]], [0, 0]])
                NLS.update(d = d, dx_odom=dx, q_odom = q_odom)
                # self.NLS.calculate_mesurement_error(self.NLS.x_origin)

                # self.alg_solver.get_update(d=d, dx_ha=dx_ha, dx_ca=dx_ca, q_ha=q_ha, q_ca=q_ca)

                nls_logger.log_data(j)

                dx_ca = np.zeros(4)
                q_ca = np.zeros((4, 4))
                dx_ha = np.zeros(4)
                q_ha = np.zeros((4, 4))

        nls_logger.plot_self()

    def test_nls_windows(self):
        sig_v = 0.15
        sig_w = 0.05
        sig_uwb = 0.25
        uwb_rate = 1.

        main_folder = "./exp3_sec1_los_sampled.pkl"  # seems to be oke trajectory for presentation purposes
        # main_folder = "./exp1_sec2_los_sampled.pkl"
        results_folder = "./Real_Exp_test/"
        data_folder = main_folder

        experiment_data, _ = self.create_experimental_data(data_folder, sig_v, sig_w, sig_uwb)
        experiment_data = experiment_data[0]
        agents={}
        factor = int(uwb_rate * experiment_data["sample_freq"])
        for drone_name in experiment_data["drones"]:
            T_vicon = experiment_data["drones"][drone_name]["T_real"]
            DT_vio = experiment_data["drones"][drone_name]["DT_slam"]
            Q_vio = experiment_data["drones"][drone_name]["Q_slam"]

            drone = NewRobot()
            drone.from_experimental_data(T_vicon, DT_vio, Q_vio, experiment_data["sample_freq"])
            agents[drone_name] =  drone

        time_steps = len(experiment_data["uwb"])
        # time_steps = 500
        factor = 10
        NLS_10 = NLS(agents, 100, sig_uwb)
        nls_logger_10 = NLSDataLogger(NLS_10)
        self.run_nls_test(agents, NLS_10, nls_logger_10, experiment_data["uwb"], time_steps, factor=factor)
        with open("presentation/NLS/w100_f_1.pkl", 'wb') as file:
            pkl.dump(nls_logger_10, file)

    def test_show_unob_uncertainty(self):
        from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent, UPFConnectedAgentDataLogger

        upf0: UPFConnectedAgent = pkl.load(open("presentation/exp1_unobservable_sampled/drone_0_losupf.pkl", "rb"))
        upf0_logger: UPFConnectedAgentDataLogger = upf0.upf_connected_agent_logger

        upf1: UPFConnectedAgent = pkl.load(open("presentation/exp1_unobservable_sampled/drone_1_losupf.pkl", "rb"))
        upf1_logger: UPFConnectedAgentDataLogger = upf1.upf_connected_agent_logger

        plt.figure()
        plt.plot(upf0.best_particle.datalogger.stds[:, 1], color='red', alpha=1, linestyle="--",
                 label="Estimation of agent 1", linewidth=3)
        plt.plot(upf1.best_particle.datalogger.stds[:, 1], color='dodgerblue', alpha=1, linestyle="--",
                 label="Estimation of agent 0", linewidth=3)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("Uncertainty on azimuth [rad]", fontsize=12)
        plt.legend(fontsize=12)
        # upf0.best_particle.datalogger.plot_ukf_states()

        plt.show()

    def test_show_trajectory_estimations(self):
        from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent, UPFConnectedAgentDataLogger
        plt.ion()
        upf0: UPFConnectedAgent = pkl.load(open("presentation/exp3_sec1_los_sampled/drone_0_losupf.pkl", "rb"))
        upf0_logger: UPFConnectedAgentDataLogger = upf0.upf_connected_agent_logger
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111, projection="3d")
        plt.show()
        for i in range(1, 40):
            ax.cla()
            ax.set_xlim(-6, 3)
            ax.set_ylim(-1, 8)
            ax.set_zlim(-4, 4)
            upf0_logger.plot_ca_active_particles(ax, i * 10, history=10)
            upf0_logger.plot_host_agent_trajectory(ax, color="red", i=i * 10)
            ax.set_title("time: :" + str(i) + "s")
            # fig.savefig('./presentation/Action/' + str(i) + '.png')
            plt.pause(0.05)

    def test_create_movie(self):
        import moviepy.video.io.ImageSequenceClip
        image_folder = './presentation/Action/'
        fps = 1

        # image_files = [os.path.join(image_folder, img)
        #                for img in os.listdir(image_folder)
        #                if img.endswith(".png")]
        image_files = []
        for i in range(1,29):
            image_files.append(os.path.join(image_folder, str(i)+".png"))
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile('my_video.mp4')

if __name__ == '__main__':
    pass