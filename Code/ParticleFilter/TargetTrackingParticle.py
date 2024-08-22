import math

import numpy as np
from scipy.linalg import sqrtm, det, inv, logm
from Code.ParticleFilter.TargetTrackingUKF import TargetTrackingUKF
from Code.BaseLines.NLS import NLS
from Code.UtilityCode.utility_fuctions import cartesianToSpherical, get_4d_rot_matrix


class TargetTrackingParticle:
    def __init__(self, weight, parent=None):
        self.weight = weight
        self.parent = parent
        self.likelihood = 1.
        self.rpea = None
        self.t_si_sj = np.zeros(4)
        self.P_t_si_sj = np.zeros((4, 4))

    def run_model(self, dt_i, q_i, t_i, P_i, dt_j, q_j, d_ij, sig_uwb, time_i):
        """
        This function should invoke the RPEA algorithm to update the particle.
        This function should update the likelihood and weight of the particle.
        This function should update the t_si_sj and P_t_si_sj.
        """
        pass

    def compare(self, other_particle):
        """
        This function should compare the particle with another particle.
        """
        #TODO continue here
        def kl_divergence(mu_P, sigma_P, mu_Q, sigma_Q):
            # Proposed by chatgpt 3.5
            k = len(mu_P)
            term1 = np.log(det(sigma_Q) / det(sigma_P))
            term2 = np.trace(inv(sigma_Q).dot(sigma_P))
            term3 = (mu_Q - mu_P).T.dot(inv(sigma_Q)).dot(mu_Q - mu_P)
            return 0.5 * (term1 - k + term2 + term3)


        distance = kl_divergence(self.t_si_sj[:3], self.P_t_si_sj[:3,:3],
                                 other_particle.t_si_sj[:3], other_particle.P_t_si_sj[:3,:3])
        return distance


    def copy(self):
        pass


class UKFLOSTargetTrackingParticle(TargetTrackingParticle):
    def __init__(self, rpea: TargetTrackingUKF, weight=1., parent=None):
        super().__init__(weight, parent=parent)
        self.rpea: TargetTrackingUKF = rpea
        self.drift_correction_bool = True

    def run_model(self, dt_i, q_i, t_i, P_i, dt_j, q_j, d_ij, sig_uwb, time_i):
        self.rpea.run_filter(dt_j, q_j, t_i, P_i, d_ij, sig_uwb, self.drift_correction_bool, True, time_i)
        self.likelihood = self.rpea.kf.likelihood
        self.weight = self.weight * self.likelihood
        self.t_si_sj = self.rpea.t_si_sj
        self.P_t_si_sj = self.rpea.P_t_si_sj

    def copy(self):
        rpea_copy = self.rpea.copy()
        return UKFLOSTargetTrackingParticle(rpea_copy, self.weight)


class NLSLOSTargetTrackingParticle(TargetTrackingParticle):
    def __init__(self, rpea: NLS, weight=1., parent=None):
        super().__init__(weight, parent=parent)
        self.rpea: NLS = rpea
        self.drift_correction_bool = True

    def run_model(self, dt_i, q_i, t_i, P_i, dt_j, q_j, d_ij, sig_uwb, time_i):
        dx = np.vstack([dt_i.reshape(1, *dt_i.shape), dt_j.reshape(1, *dt_j.shape)])
        q_odom = np.vstack([q_i.reshape(1, *q_i.shape), q_j.reshape(1, *q_j.shape)])
        d = np.array([[0, d_ij], [0, 0]])
        self.rpea.update(d, dx, q_odom)
        # self.rpea.run_filter(dt_j, q_j, t_i, P_i, d_ij, sig_uwb, self.drift_correction_bool, True, time_i)
        self.t_si_sj = self.rpea.x_rel[0, 1]
        self.P_t_si_sj = get_4d_rot_matrix(self.rpea.x[0, -1]) * self.rpea.x_cov[-8:-4, -8:-4] + self.rpea.x_cov[-4:, -4:]
        self.likelihood = self.rpea.likelihood
        self.weight = self.weight * self.likelihood

    def copy(self):
        rpea_copy = self.rpea.copy()
        return NLSLOSTargetTrackingParticle(rpea_copy, self.weight)

class ListOfTargetTrackingParticles:
    def __init__(self):
        self.particles = []
        self.n_altitude = 1
        self.n_azimuth = 1
        self.n_heading = 1

    def create_particle(self, t_i, t_j_0, sig_t_j_0, d_ij, sig_d) -> TargetTrackingParticle:
        pass

    def add_particle_with_know_start_pose(self, t_j_0, sig_j):
        s = cartesianToSpherical(t_j_0[:3]).tolist()
        # self.particles.append(particle)

    def split_sphere_in_equal_areas(self, t_i, d_ij: float, sigma_uwb: float, n_altitude: int, n_azimuth: int,
                                    n_heading: int):
        """
        Function to split the area of a sphere in almost equal areas (= weights)
        Starting from the n_altitude that has to be uneven (will be made uneven).
        For each altitude
        """
        self.n_altitude = n_altitude
        self.n_azimuth = n_azimuth
        self.n_heading = n_heading
        # self.sigma_uwb = self.sigma_uwb_factor * sigma_uwb

        if n_altitude % 2 == 0:
            n_altitude += 1
        # n_altitude = n_altitude+2
        altitude_delta = np.pi / (n_altitude)
        # altitudes = np.round([-np.pi / 2  + i * altitude_delta for i in range(1,n_altitude-1)], 4)
        altitudes = np.round([-np.pi / 2 + altitude_delta / 2 + i * altitude_delta for i in range(n_altitude)], 4)
        # calculate the areas to calculate the number of azimuth angles is needed.
        altitude_surfaces = [(np.sin(altitude + altitude_delta / 2) - np.sin(altitude - altitude_delta / 2)) / 2 for
                             altitude in altitudes]
        # print(altitude_surfaces)
        area = altitude_surfaces[int((len(altitude_surfaces) - 1) / 2)] / n_azimuth
        azimuth_bins = [math.ceil(altitude_surface / area) for altitude_surface in altitude_surfaces]

        sigma_azimuths = [(2 * np.pi / azimuthBin) / np.sqrt(-8 * np.log(0.5)) for azimuthBin in azimuth_bins]
        sigma_altitude = (np.pi / n_altitude) / np.sqrt(-8 * np.log(0.5))

        headings = [-np.pi + (2 * np.pi / n_heading) * j for j in range(n_heading)]
        sigma_heading = (2 * np.pi / n_heading) / np.sqrt(-8 * np.log(0.5))

        for i, altitude in enumerate(altitudes):
            azimuths = [-np.pi + (2 * np.pi / azimuth_bins[i]) * j for j in range(azimuth_bins[i])]
            sigma_s = [2 * sigma_uwb, sigma_azimuths[i], sigma_altitude, sigma_heading]
            for azimuth in azimuths:
                # s = np.array([d_ij, azimuth, altitude], dtype=float)
                for heading in headings:
                    s = np.array([d_ij, azimuth, altitude, heading], dtype=float)
                    particle = self.create_particle(t_i, s, sigma_s, d_ij, sigma_uwb)
                    # particle.set_initial_state(s, sigma_s, heading, sigma_heading, self.sigma_uwb)
                    self.particles.append(particle)


class ListOfUKFLOSTargetTrackingParticles(ListOfTargetTrackingParticles):
    def __init__(self):
        super().__init__()
        self.alpha = 1
        self.beta = 2
        self.kappa = -1
        self.t_sj_uwb = np.zeros(4)
        self.t_si_uwb = np.zeros(4)

    def set_uwb_extrinsicity(self, t_si_uwb, t_sj_uwb):
        self.t_si_uwb = t_si_uwb
        self.t_sj_uwb = t_sj_uwb

    def set_ukf_parameters(self, kappa=-1, alpha=1, beta=2):
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def create_particle(self, t_i, t_j_0, sig_t_j_0, d_ij, sig_d) -> UKFLOSTargetTrackingParticle:
        # weight = 1. / self.n_azimuth / self.n_altitude / self.n_heading
        rpea: TargetTrackingUKF = TargetTrackingUKF(x_ha_0=t_i)
        rpea.set_uwb_extrinsicity(self.t_si_uwb, self.t_sj_uwb)
        rpea.set_ukf_properties(self.kappa, self.alpha, self.beta)
        rpea.set_initial_state(t_j_0, sig_t_j_0)

        particle = UKFLOSTargetTrackingParticle(rpea=rpea)
        return particle
