# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from filterpy.kalman import unscented_transform
from filterpy.stats import logpdf
from filterpy.common import pretty_str
from scipy.linalg import cholesky
from Code.UtilityCode.utility_fuctions import cartesianToSpherical, limit_angle
from deprecated import deprecated

@deprecated('predict_unscented_transform is deprecated. Use unscented_transform instead.')
def predict_unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                                mean_fn=None, residual_fn=None, overhead_bool=False):
    r"""
    sigmas is in spherical.
    Bassed on the unscented transform form filterpy, but adapted for spherical cartesian conversions.
    """

    kmax, n = sigmas.shape
    # sigmas_s = np.array([np.concatenate([cartesianToSpherical(sig[:3]), [sig[3]]]) for sig in sigmas]).astype(
    #     np.float64)
    sigmas_s = sigmas
    if not overhead_bool:  # Then everything can be done in spherical
        try:
            if mean_fn is None:
                # new mean is just the sum of the sigmas * weight
                # r_sigma = np.array([np.linalg.norm(sigma[:3]) for sigma in sigmas])
                x_s = np.dot(Wm, sigmas_s)  # dot = \Sigma^n_1 (W[k]*Xi[k])
                # r = np.dot(Wm, sigmas_s[:, 0])
                # phi = np.dot(Wm, sigmas_s[:, 2])
            else:
                x_s = mean_fn(sigmas_s, Wm)
        except:
            raise
        y_s = sigmas_s - x_s[np.newaxis, :]
        P = np.dot(y_s.T, np.dot(np.diag(Wc), y_s))

        if noise_cov is not None:
            P += noise_cov

    else:
        try:
            if mean_fn is None:
                # new mean is just the sum of the sigmas * weight
                # r_sigma = np.array([np.linalg.norm(sigma[:3]) for sigma in sigmas])
                x = np.dot(Wm, sigmas)  # dot = \Sigma^n_1 (W[k]*Xi[k])
                r = np.dot(Wm, sigmas_s[:, 0])
                phi = np.dot(Wm, sigmas_s[:, 2])
            else:
                x = mean_fn(sigmas, Wm)
        except:
            print(sigmas)
            raise
        x_s = np.concatenate([cartesianToSpherical(x[:3]), [x[3]]]).astype(np.float64)
        x_s[0] = r
        x_s[2] = phi
        y_s = sigmas_s - x_s[np.newaxis, :]
        # y_s = np.concatenate([cartesianToSpherical(y[:3]), [y[3]]]).astype(np.float64)
        P = np.dot(y_s.T, np.dot(np.diag(Wc), y_s))

        if noise_cov is not None:
            P += noise_cov

    return (x_s, P)


class ModifiedMerweScaledSigmaPoints(object):
    r"""


    """

    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        # pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        """
        returns sigma points in spherical coordinates.
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n) * P)
        # try:
        #     U = self.sqrt((lambda_ + n) * P)
        # except np.linalg.LinAlgError:
        #     P = np.diag(np.diag(P))
        #     print("P is not positive definite, using diagonal instead")
        #     # P[4:, :4] = np.zeros((4,4))
        #     # P[:4, 4:] = np.zeros((4,4))
        #     U = self.sqrt((lambda_ + n) * P)

        if U[1, 1] > np.pi:
            U[1, 1] = np.pi
        if U[2, 2] > np.pi / 2:
            U[2, 2] = np.pi / 2
        if U[3, 3] > np.pi :
            U[3, 3] = np.pi
        if U[-2,-2] > np.pi :
            U[-2,-2] = np.pi
        if U[-1, -1] >np.pi :
            U[-1, -1] =np.pi


        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])

        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.

        """

        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

    def __repr__(self):

        return '\n'.join([
            'MerweScaledSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('alpha', self.alpha),
            pretty_str('beta', self.beta),
            pretty_str('kappa', self.kappa),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
        ])


class ModifiedUnscentedKalmanFilter(object):
    r"""
    Based on the UnscentedKalmanFilter form filterpy
    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """

        # pylint: disable=too-many-arguments
        # OWN VARIABLES:
        self.overhead_bool = False

        # EXISING VARIABLES:
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain
        self.y = np.zeros((dim_z))  # residual
        self.z = np.array([[None] * dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self.inv = np.linalg.inv

        # these will always be a copy of x,P after odometry_integration() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        r"""
        Performs the odometry_integration step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)  # self.sigma_f is in special spherical.

        # ADAPTATION TO FIX WP2.2B1 -------------------------------------------
        # if np.abs(self.sigmas_f[0,1]) > 2/3*np.pi :
        # if not np.all(np.sign(self.sigmas_f[:,1]) == np.sign(self.sigmas_f[0,1])):
        #     print("UKF")
        #     for sigma in self.sigmas_f:
        #         if sigma[1] < 0:
        #             sigma[1] += 2*np.pi
        # -----------------------------------------------------------------------

        # and pass sigmas through the unscented transform to compute prior
        # x is in special spherical
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)
        # if np.abs(self.x[2]) > np.pi/2:
        #     self.x[1] = limit_Angle(self.x[1] + np.pi)
        #     self.x[2] = limit_Angle(np.pi - self.x[2])
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None] * self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of uwb_measurement varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = dot(Pxz, self.SI)  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # From special spherical to spherical :
        if np.abs(self.x[2]) > np.pi / 2:
            self.x[1] = self.x[1] + np.pi
            self.x[2] = np.pi - self.x[2]
        self.x[1] = limit_angle(self.x[1])
        self.x[2] = limit_angle(self.x[2])
        self.x[3] = limit_angle(self.x[3])
        self.x[-1] = limit_angle(self.x[-1])
        # if self.x[1] > np.pi:
        if self.P[1,1] > np.pi:
            self.P[1,1] = np.pi
        if self.P[2,2] > np.pi/2:
            self.P[2,2] = np.pi/2
        if self.P[3,3] > np.pi:
            self.P[3,3] = np.pi
        if self.P[-1,-1] > np.pi:
            self.P[-1,-1] = np.pi
        if self.P[-2, -2] > np.pi:
            self.P[-2, -2] = np.pi
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `uwb_measurement`.
        """
        # x_c = np.concatenate([sphericalToCartesian(x[:3]), [x[3]]])
        # Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        # N = sigmas_f.shape[0]
        # for i in range(N):
        #     # sigma_s = np.concatenate([cartesianToSpherical(sigmas_f[i, :3]), [sigmas_f[i, 3]]])
        #     dx = self.residual_x(sigmas_f[i], x)
        #     # dx = np.concatenate([cartesianToSpherical(dx_c[:3]), [dx_c[3]]])
        #     dz = self.residual_z(sigmas_h[i], uwb_measurement)
        #     Pxz += self.Wc[i] * outer(dx, dz)
        # return Pxz

        Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * outer(dx, dz)
        return Pxz

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to odometry_integration (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.

        Gets simgas in spherical
        And translates them into cartesian. => self.sigma_f is in cartesian.
        """

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas_s = self.points_fn.sigma_points(self.x, self.P)
        # sigmas_c = np.array([np.concatenate([sphericalToCartesian(s[:3]), [s[3]]]) for s in sigmas_s])
        for i, s in enumerate(sigmas_s):
            # sigma_f is in special spherical
            self.sigmas_f[i] = fx(s, dt, **fx_args)
        # if np.sign(sigmas_c[0,0]) != np.sign(self.sigmas_f[0,0]) :
        #     self.overhead_bool = True
        # if np.sign(sigmas_c[0,1]) != np.sign(self.sigmas_f[0,1]) :
        #     self.overhead_bool = True

        return None

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        """
        Performs the UKF filter over the list of measurement in `zs`.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.

        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.

        dts : None, scalar or list-like, default=None
            optional value or list of delta time to be passed into odometry_integration.

            If dtss is None then self.dt is used for all epochs.

            If it is a list where len(dts) == len(zs), then it is treated as a
            list of dt values, one per epoch. This allows you to have varying
            epoch durations.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

        Returns
        -------

        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        Examples
        --------

        .. code-block:: Python

            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts The output is then smoothed
            # with an RTS smoother.

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = ukf.batch_filter(zs, dts=dts)
            (xs, Ps, Ks) = ukf.rts_smoother(mu, cov)

        """
        # pylint: disable=too-many-arguments

        try:
            z = zs[0]
        except TypeError:
            raise TypeError('zs must be list-like')

        if self._dim_z == 1:
            if not (isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError('zs must be a list of scalars or 1D, 1 element arrays')
        else:
            if len(z) != self._dim_z:
                raise TypeError(
                    'each element in zs must be a 1D array of length {}'.format(self._dim_z))

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [self.R] * z_n

        if dts is None:
            dts = [self._dt] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return (means, covariances)

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """
        # pylint: disable=too-many-locals, too-many-arguments

        if len(Xs) != len(Ps):
            raise ValueError('Xs and Ps must have the same length')

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform

        # smoother gain
        Ks = zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((num_sigmas, dim_x))

        for k in reversed(range(n - 1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, self.Q,
                self.x_mean, self.residual_x)

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, self.inv(Pb))

            # update the smoothed estimates
            xs[k] += dot(K, self.residual_x(xs[k + 1], xb))
            ps[k] += dot(K, ps[k + 1] - Pb).dot(K.T)
            Ks[k] = K

        return (xs, ps, Ks)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'UnscentedKalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('S', self.S),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('likelihood', self.likelihood),
            pretty_str('mahalanobis', self.mahalanobis),
            pretty_str('sigmas_f', self.sigmas_f),
            pretty_str('h', self.sigmas_h),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('residual_x', self.residual_x),
            pretty_str('residual_z', self.residual_z),
            pretty_str('msqrt', self.msqrt),
            pretty_str('hx', self.hx),
            pretty_str('fx', self.fx),
            pretty_str('x_mean', self.x_mean),
            pretty_str('z_mean', self.z_mean)
        ])
