"""
author: Florian Krach & Calypso Herrera

code to generate synthetic data from stock-model SDEs
"""

import copy
import math
from math import exp, isclose, pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from configs.dataset_configs import get_rectangle_bounds
from fbm import fgn
from scipy.integrate import quad
from scipy.special import softmax
from scipy.stats import norm
from tqdm import tqdm


class StockModel:
    """
    mother class for all stock models defining the variables and methods shared
    amongst all of them, some need to be defined individually
    """

    def __init__(
        self, drift, volatility, S0, nb_paths, nb_steps, maturity, sine_coeff, **kwargs
    ):
        self.drift = drift
        self.volatility = volatility
        self.S0 = S0
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dt = maturity / nb_steps
        self.dimensions = np.size(S0)
        if sine_coeff is None:
            self.periodic_coeff = lambda t: 1
        else:
            self.periodic_coeff = lambda t: (1 + np.sin(sine_coeff * t))
        self.loss = None
        self.path_t = None
        self.path_y = None
        self.masked = False
        self.track_obs_cov_mat = False

    def generate_paths(self, **options):
        """
        generate random paths according to the model hyperparams
        :return: stock paths as np.array, dim: [nb_paths, data_dim, nb_steps]
        """
        raise ValueError("not implemented yet")

    def next_cond_exp(self, *args, **kwargs):
        """
        compute the next point of the conditional expectation starting from
        given point for given time_delta
        :return: cond. exp. at next time_point (= current_time + time_delta)
        """
        raise ValueError("not implemented yet")

    def get_cov_mat(self):
        raise NotImplementedError()

    def compute_cond_exp(
        self,
        times,
        time_ptr,
        X,
        obs_idx,
        delta_t,
        T,
        start_X,
        n_obs_ot,
        return_path=True,
        get_loss=False,
        weight=0.5,
        store_and_use_stored=True,
        start_time=None,
        **kwargs,
    ):
        """
        compute conditional expectation similar to computing the prediction in
        the model.NJODE.forward
        ATTENTION: Works correctly only for non-masked data!
        :param times: see model.NJODE.forward
        :param time_ptr: see model.NJODE.forward
        :param X: see model.NJODE.forward, as np.array
        :param obs_idx: see model.NJODE.forward, as np.array
        :param delta_t: see model.NJODE.forward, as np.array
        :param T: see model.NJODE.forward
        :param start_X: see model.NJODE.forward, as np.array
        :param n_obs_ot: see model.NJODE.forward, as np.array
        :param return_path: see model.NJODE.forward
        :param get_loss: see model.NJODE.forward
        :param weight: see model.NJODE.forward
        :param store_and_use_stored: bool, whether the loss, and cond exp path
            should be stored and reused when calling the function again
        :param start_time: None or float, if float, this is first time point
        :param kwargs: unused, to allow for additional unused inputs
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        y = start_X
        batch_size = start_X.shape[0]

        assert not self.track_obs_cov_mat or (self.track_obs_cov_mat and self.masked)

        if self.masked:
            self.observed_t = [[] for _ in range(batch_size)]
            self.observed_X = [[] for _ in range(batch_size)]

        if self.track_obs_cov_mat:
            self.obs_cov_mat = [None for x in range(batch_size)]
            self.obs_cov_mat_inv = [None for x in range(batch_size)]

        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.0]
                path_y = [y]

        last_obs_times = [0 for _ in range(batch_size)]
        for i, obs_time in tqdm(enumerate(times), total=len(times)):
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10 * delta_t:
                break
            if obs_time <= current_time:
                continue
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time

                kwargs["last_obs_times"] = last_obs_times
                y = self.next_cond_exp(y, delta_t_, current_time, **kwargs)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            # Select which batches are relevant
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]  # X_obs first dim is \in [1, batch size]
            i_obs = obs_idx[start:end]

            for bid in i_obs:
                last_obs_times[bid] = times[i]

            # add to observed, if we're tracking it
            if self.masked:
                for j, ii in enumerate(i_obs):
                    self.observed_t[ii].append(obs_time)
                    self.observed_X[ii].append(X_obs[j, 0])

                    if self.track_obs_cov_mat:
                        self.obs_cov_mat[ii] = self.get_cov_mat(self.observed_t[ii])
                        self.obs_cov_mat_inv[ii] = np.linalg.inv(self.obs_cov_mat[ii])

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs,
                    Y_obs=Y[i_obs],
                    Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs],
                    batch_size=batch_size,
                    weight=weight,
                )

            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            kwargs["last_obs_times"] = last_obs_times
            y = self.next_cond_exp(y, delta_t_, current_time, **kwargs)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def get_optimal_loss(
        self,
        times,
        time_ptr,
        X,
        obs_idx,
        delta_t,
        T,
        start_X,
        n_obs_ot,
        weight=0.5,
        M=None,
        mult=None,
        path_idxs=None,
    ):
        if mult is not None and mult > 1:
            bs, dim = start_X.shape
            _dim = round(dim / mult)
            X = X[:, :_dim]
            start_X = start_X[:, :_dim]
            if M is not None:
                M = M[:, :_dim]

        loss, _, _ = self.compute_cond_exp(
            times,
            time_ptr,
            X,
            obs_idx,
            delta_t,
            T,
            start_X,
            n_obs_ot,
            return_path=True,
            get_loss=True,
            weight=weight,
            M=M,
            path_idxs=path_idxs,
        )

        return loss


class FracBM(StockModel):
    """
    Implementing FBM via FBM package
    """

    def __init__(
        self, nb_paths, nb_steps, S0, maturity, hurst, method="daviesharte", **kwargs
    ):
        """Instantiate the FBM"""
        super().__init__(
            drift=None,
            volatility=None,
            S0=S0,
            nb_paths=nb_paths,
            nb_steps=nb_steps,
            maturity=maturity,
            sine_coeff=None,
        )
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.S0 = S0
        self.maturity = maturity
        self.hurst = hurst
        self.method = method
        self.dimensions = np.size(S0)
        self.loss = None
        self.path_t = None
        self.path_y = None
        self.masked = True
        self.track_obs_cov_mat = True

    def r_H(self, t, s):
        return 0.5 * (
            t ** (2 * self.hurst)
            + s ** (2 * self.hurst)
            - np.abs(t - s) ** (2 * self.hurst)
        )

    def get_cov_mat(self, times):
        m = np.array(times).reshape((-1, 1)).repeat(len(times), axis=1)
        return self.r_H(m, np.transpose(m))

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        t = current_t + delta_t
        next_y = np.zeros_like(y)
        for ii in range(y.shape[0]):
            if self.obs_cov_mat_inv[ii] is not None:
                r = self.r_H(np.array(self.observed_t[ii]), t)
                next_y[ii] = np.dot(
                    r,
                    np.matmul(self.obs_cov_mat_inv[ii], np.array(self.observed_X[ii])),
                )
        return next_y

    def compute_cond_exp(self, *args, **kwargs):
        assert self.dimensions == 1, "cond. exp. computation of FBM only for 1d"
        assert self.S0 == 0, "cond. exp. computation of FBM only for S0=0"
        return super().compute_cond_exp(*args, **kwargs)

    def generate_paths(self, start_X=None):
        spot_paths = np.empty((self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :, 0] = self.S0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                fgn_sample = fgn(
                    n=self.nb_steps,
                    hurst=self.hurst,
                    length=self.maturity,
                    method=self.method,
                )
                spot_paths[i, j, 1:] = np.cumsum(fgn_sample) + spot_paths[i, j, 0]
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class ReflectedBM(StockModel):
    def __init__(
        self,
        mu,
        sigma,
        max_terms,
        lb,
        ub,
        nb_paths,
        dimension,
        nb_steps,
        maturity,
        use_approx_paths_technique,
        use_numerical_cond_exp,
        **kwargs,
    ):
        assert sigma > 0
        assert lb < ub

        self.mu = mu
        self.sigma = sigma
        self.max_terms = max_terms
        self.lb = lb
        self.ub = ub
        self.nb_paths = nb_paths
        self.dimension = dimension
        self.dimensions = 1
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dt = maturity / nb_steps
        self.loss = None
        self.path_t = None
        self.path_y = None
        self.use_approx_paths_technique = use_approx_paths_technique
        self.use_numerical_cond_exp = use_numerical_cond_exp
        self.masked = True
        self.track_obs_cov_mat = False

    def _in_shape(self, x):
        return self.lb < x < self.ub

    def _generate_approx_paths(self, x0):
        spot_paths = np.zeros((self.nb_paths, self.dimensions, self.nb_steps + 1))
        spot_paths[:, :, 0] = x0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                raw_paths = generate_BM_drift_diffusion(
                    1, 1, self.nb_steps, self.dt, [self.mu], [self.sigma]
                )
                shifted_paths = raw_paths + x0

                # Deal with boundary collisions. Assume at most one collision per time step
                for k in range(len(shifted_paths[0, 0])):
                    x = shifted_paths[0, 0, k]
                    if x < self.lb:
                        dist = self.lb - x
                        shifted_paths[0, 0, k:] += 2 * dist
                    elif x > self.ub:
                        dist = x - self.ub
                        shifted_paths[0, 0, k:] -= 2 * dist

                spot_paths[i, j, 1:] = shifted_paths[0][0]

        assert np.all(spot_paths >= self.lb) and np.all(spot_paths <= self.ub)
        return spot_paths

    def _generate_true_paths(self, x0):
        raise NotImplementedError()

    def generate_paths(self, x0=None):
        if x0 is None:
            x0 = (self.lb + self.ub) / 2
        assert self._in_shape(x0)

        paths = (
            self._generate_approx_paths(x0)
            if self.use_approx_paths_technique
            else self._generate_true_paths(x0)
        )

        assert np.all(paths >= self.lb) and np.all(paths <= self.ub)

        return paths, self.dt

    def _clip_if_out_but_close(self, x, boundary, is_lower_boundary):
        out_test = x >= boundary if is_lower_boundary else x <= boundary
        if not out_test and isclose(x, boundary, rel_tol=10e-4):
            return boundary
        else:
            return x

    def reflected_bm_pdf(self, x, t, x0, t0):
        # Follow eqn 11 of
        # https://link.springer.com/content/pdf/10.1023/B:CSEM.0000049491.13935.af.pdf
        mu = self.mu
        sigma = self.sigma
        pinf = self.max_terms
        ninf = -self.max_terms
        c = self.lb
        d = self.ub
        phi = norm.cdf

        x = self._clip_if_out_but_close(x, c, True)
        x = self._clip_if_out_but_close(x, d, False)

        # Zero mass outside of the boundaries
        if not (c <= x <= d):
            return 0.0

        coeff = 1.0 / (sigma * sqrt(2 * pi * (t - t0)))
        S1 = coeff * sum(
            exp(
                (2 * mu * n * (c - d) / (sigma**2))
                + (
                    -(
                        ((x + 2 * n * (d - c) - x0 - mu * (t - t0)) ** 2)
                        / (2 * (sigma**2) * (t - t0))
                    )
                )
            )
            for n in range(ninf, pinf)
        )

        S2 = coeff * sum(
            exp(
                -(2 * mu * (n * d - (n + 1) * c + x0)) / sigma**2
                + (
                    -((2 * n * d - 2 * (n + 1) * c + x0 + x - mu * (t - t0)) ** 2)
                    / (2 * sigma**2 * (t - t0))
                )
            )
            for n in range(ninf, pinf)
        )

        coeff = (2 * mu) / sigma**2

        S3 = 0
        for n in range(0, pinf):
            t2 = 1 - phi(
                (mu * (t - t0) + 2 * n * d - 2 * (n + 1) * c + x0 + x)
                / (sigma * sqrt(t - t0))
            )
            # avoid danger because t1 can become very large, even if s2 is exactly 0
            # note the numerator of t2 [0, 1] so we don't really have to worry about it overflowing
            if isclose(t2, 0):
                # adding 0, since t2 will make the whole term 0
                continue
            t1 = (2 * mu * (n * d - (n + 1) * c + x)) / sigma**2
            S3 += exp(t1) * t2
        S3 = -coeff * S3

        # Same technique as above
        S4 = 0
        for n in range(0, pinf):
            t2 = phi(
                (mu * (t - t0) - 2 * (n + 1) * d + 2 * n * c + x0 + x)
                / (sigma * sqrt(t - t0))
            )
            if isclose(t2, 0):
                continue
            t1 = 2 * mu * (n * c - (n + 1) * d + x) / sigma**2
            S4 += exp(t1) * t2
        S4 = coeff * S4

        fin = S1 + S2 + S3 + S4

        return fin

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        assert delta_t > 0
        cond_exp = (
            self._compute_numerical_next_cond_exp(y, delta_t, current_t)
            if self.use_numerical_cond_exp
            else self._compute_true_next_cond_exp(y, delta_t, current_t)
        )

        assert np.all(cond_exp >= self.lb) and np.all(cond_exp <= self.ub)

        return cond_exp

    def _compute_numerical_next_cond_exp(self, y, delta_t, current_t):
        t0 = current_t
        t = current_t + delta_t
        out = np.zeros_like(y)
        for i, x0 in enumerate(y):
            x0 = self._clip_if_out_but_close(x0, self.lb, True)
            x0 = self._clip_if_out_but_close(x0, self.ub, False)

            integrand = lambda x: x * self.reflected_bm_pdf(x, t, x0, t0)
            # Increase error tolerance for speed, default was ~1e-8
            # NOTE can be very slow, even with this
            integral, _, _ = quad(
                integrand, self.lb, self.ub, epsabs=1e-3, full_output=True
            )
            out[i] = integral

        return out

    def _compute_true_next_cond_exp(self, y, delta_t, current_t):
        raise NotImplementedError


class Rectangle(StockModel):
    def __init__(
        self,
        width,
        length,
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        max_terms,
        nb_paths,
        dimension,
        nb_steps,
        maturity,
        use_approx_paths_technique,
        use_numerical_cond_exp,
        **kwargs,
    ):
        # TODO doubt all of these are used, same for other datasets
        assert width > 0 and length > 0

        self.width = width
        self.length = length
        self.use_approx_paths_technique = use_approx_paths_technique
        self.use_numerical_cond_exp = use_numerical_cond_exp
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        self.max_terms = max_terms
        self.nb_paths = nb_paths
        self.dimension = dimension
        self.dimensions = 1
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dt = maturity / nb_steps
        self.loss = None
        self.path_t = None
        self.path_y = None
        self.use_approx_paths_technique = use_approx_paths_technique
        self.use_numerical_cond_exp = use_numerical_cond_exp
        self.masked = True
        self.track_obs_cov_mat = False

        lb_x, ub_x, lb_y, ub_y = get_rectangle_bounds(width, length)

        self.rbm_x = ReflectedBM(
            mu=mu_x,
            sigma=sigma_x,
            max_terms=max_terms,
            lb=lb_x,
            ub=ub_x,
            nb_paths=nb_paths,
            dimension=dimension,
            nb_steps=nb_steps,
            maturity=maturity,
            use_approx_paths_technique=use_approx_paths_technique,
            use_numerical_cond_exp=use_numerical_cond_exp,
        )

        self.rbm_y = ReflectedBM(
            mu=mu_y,
            sigma=sigma_y,
            max_terms=max_terms,
            lb=lb_y,
            ub=ub_y,
            nb_paths=nb_paths,
            dimension=dimension,
            nb_steps=nb_steps,
            maturity=maturity,
            use_approx_paths_technique=use_approx_paths_technique,
            use_numerical_cond_exp=use_numerical_cond_exp,
        )

    def _in_shape(self, x):
        return self.rbm_x._in_shape(x[0]) and self.rbm_y._in_shape(x[1])

    def plot_first_path(self, paths_x, paths_y):
        # TODO eventually remove this
        path_x = paths_x[0]
        path_y = paths_y[0]
        xs = np.arange(path_x.shape[1])
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(xs, path_x, path_y)
        plt.savefig("rect_dataset_plot.png")
        plt.close(fig)

    def generate_paths(self, x0=(None, None)):
        if None in x0:
            x0 = (0, 0)
        paths_x, dt_x = self.rbm_x.generate_paths(x0[0])
        paths_y, dt_y = self.rbm_y.generate_paths(x0[1])
        assert dt_x == dt_y

        self.plot_first_path(paths_x, paths_y)  # for fun

        return np.concatenate((paths_x, paths_y), axis=1), dt_x

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        assert delta_t > 0
        # y has shape (batch size, 2), since 2 is the process dimension
        # Need to transform it into two variables with shape(batch size, 1)
        y_x = np.array([[y_x] for y_x in y[:, 0]])
        y_y = np.array([[y_x] for y_x in y[:, 1]])
        cond_exp_x = self.rbm_x.next_cond_exp(y_x, delta_t, current_t)
        cond_exp_y = self.rbm_y.next_cond_exp(y_y, delta_t, current_t)
        return np.concatenate((cond_exp_x, cond_exp_y), axis=1)


class Ball2D_BM(StockModel):
    def __init__(
        self,
        max_radius,
        nb_paths,
        nb_steps,
        maturity,
        dimension,
        **kwargs,
    ):
        assert dimension == 2
        self.max_radius = max_radius
        self.dimension = dimension
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dt = maturity / nb_steps

        # Scaling factor
        self.alpha_theta = pi

        self.loss = None
        self.path_t = None
        self.path_y = None
        self.masked = False
        self.track_obs_cov_mat = False

    def generate_paths(self, x0=None):
        if x0 is None:
            x0 = [0, 0]
        else:
            assert x0[0] ** 2 + x0[1] ** 2 <= self.max_radius**2 or isclose(
                x0[0] ** 2 + x0[1] ** 2, self.max_radius**2
            )

        radius_raw_paths = generate_BM(
            self.nb_paths,
            1,
            self.nb_steps,
            self.dt,
        )

        radius_paths = self.max_radius * np.abs(np.tanh(radius_raw_paths))

        angle_raw_paths = generate_BM(
            self.nb_paths,
            self.dimension - 1,
            self.nb_steps,
            self.dt,
        )

        x_coords = radius_paths * np.cos(self.alpha_theta * angle_raw_paths)
        y_coords = radius_paths * np.sin(self.alpha_theta * angle_raw_paths)

        res = np.zeros(shape=(self.nb_paths, self.dimension, self.nb_steps + 1))
        res[:, :, 0] = x0
        res[:, 0, 1:] = x_coords[:, 0, :]
        res[:, 1, 1:] = y_coords[:, 0, :]

        # Make sure all points are inside the ball
        for p in range(self.nb_paths):
            for s in range(self.nb_steps + 1):
                x, y = res[p, :, s]
                assert x**2 + y**2 <= self.max_radius**2 or isclose(
                    x**2 + y**2, self.max_radius**2
                )

        return res, self.dt

    def monte_carlo_radius(self, r_s, interval):
        def get_instance(sample):
            return np.abs(np.tanh(sample + np.arctanh(r_s)))

        N = 30
        res = 0
        for _ in range(N):
            sample = np.random.normal(0, sqrt(interval))
            res += (1 / N) * get_instance(sample)

        return res

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        res = np.zeros_like(y)
        s, t = current_t, current_t + delta_t
        for i, point in enumerate(y):
            # Radius conditional expectation
            r_s = np.linalg.norm(point, 2) / self.max_radius
            if r_s == 0:
                cos_s, sin_s = 0, 0
            else:
                cos_s = point[0] / (self.max_radius * r_s)
                sin_s = point[1] / (self.max_radius * r_s)

            r_t = self.monte_carlo_radius(r_s, t - s)

            # Angle conditional expectation
            pow = (self.alpha_theta) * (t - s)
            temp = (1 / sqrt(math.e)) ** pow
            cos_t = temp * cos_s
            sin_t = temp * sin_s

            res[i] = self.max_radius * r_t * np.array([cos_t, sin_t])

        return res


class BMWeights(StockModel):
    def __init__(
        self,
        vertices,
        should_compute_approx_cond_exp_paths,
        dimension,
        nb_paths,
        nb_steps,
        maturity,
        **kwargs,
    ):
        assert all(len(v) == dimension for v in vertices)
        self.vertices = np.array(vertices)
        self.dimension = dimension
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dt = maturity / nb_steps
        self.should_compute_approx_cond_exp_paths = should_compute_approx_cond_exp_paths
        if self.should_compute_approx_cond_exp_paths:
            self.motion_paths = None

        self.loss = None
        self.path_t = None
        self.path_y = None
        self.masked = False
        self.track_obs_cov_mat = False

        if len(vertices) > 4:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(vertices, incremental=False)
            assert len(hull.vertices) == len(
                vertices
            ), "Some of the points you gave aren't on the convex hull."

    def weights_to_point(self, w):
        return np.matmul(w, self.vertices)

    @property
    def paths_dir(self):
        # TODO really hacky since assumes just the first BMWeights
        return Path("..") / "data" / "training_data" / "paths" / "BMWeights-1"

    def _get_time_idx_from_time(self, t):
        idx = round(t / self.dt, 5)  # avoid floating point errors, e.g. 3.0000001
        assert float.is_integer(idx), "Strategy to get index from times failed"
        idx = int(idx)

        return idx

    def compute_cond_exp_approx(self, s, t, path_idx):
        s_idx = self._get_time_idx_from_time(s)

        fp = self.paths_dir / "motion_paths.npy"
        self.motion_paths = np.load(fp)

        W_tilde_s = self.motion_paths[path_idx, :, s_idx]  # what we condition on
        n = len(self.vertices)

        def sample_cond_exp(incr):
            cexpect = np.zeros(n)
            for i in range(n):
                num = np.exp(incr[i] + W_tilde_s[i])
                denom = 0
                for j in range(n):
                    denom += np.exp(incr[j] + W_tilde_s[j])
                cexpect[i] = num / denom

            return cexpect

        # Now do Monte Carlo
        N = 60
        weight_cond_exp = np.zeros(n)
        for _ in range(N):
            increment_sample = np.random.normal(0, sqrt(t - s), size=n)
            weight_cond_exp += (1 / N) * sample_cond_exp(increment_sample)

        res = np.array([self.weights_to_point(weight_cond_exp)])

        return res

    def generate_paths(self, x0=None):
        k = len(self.vertices)
        if x0 is None:
            x0 = np.zeros(k)

        assert len(x0) == k

        spot_motion_paths = generate_BM(
            self.nb_paths, k, self.nb_steps + 1, self.dt, x0
        )
        spot_weight_paths = softmax(spot_motion_paths, axis=1)
        spot_paths = np.zeros((self.nb_paths, self.dimension, self.nb_steps + 1))

        # Convert to points
        for p in range(self.nb_paths):
            for s in range(self.nb_steps + 1):
                w = spot_weight_paths[p, :, s]
                pt = self.weights_to_point(w)
                spot_paths[p, :, s] = pt

        if self.should_compute_approx_cond_exp_paths:
            p = self.paths_dir
            p.mkdir(parents=True, exist_ok=True)

            self.motion_paths = spot_motion_paths
            fp = p / "motion_paths.npy"
            assert not fp.is_file(), "There's already a motion path here"
            np.save(fp, self.motion_paths)

        return spot_paths, self.dt

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        if self.should_compute_approx_cond_exp_paths:
            t = delta_t + current_t
            cond_exp = np.zeros_like(y)
            for i in range(len(y)):
                path_idx = kwargs["path_idxs"][i]
                s = kwargs["last_obs_times"][i]
                cond_exp[i] = self.compute_cond_exp_approx(s, t, path_idx)
            return cond_exp
        else:
            raise NotImplementedError()


def generate_BM_drift_diffusion(nb_paths, dim, nb_steps, dt, mu, sigma, x0=None):
    # Here x0 is a _sample_
    assert dim == len(mu) == len(sigma)
    assert all(x > 0 for x in sigma)
    assert dt > 0

    sampled_numbers = np.random.standard_normal((nb_paths, dim, nb_steps))
    motion_paths = np.zeros_like(sampled_numbers)

    # wlog just replace the first sample by x0
    if x0 is not None:
        motion_paths[:, :, 0] = x0

    for p in range(nb_paths):
        for i in range(dim):
            for t in range(1, nb_steps):
                prev = motion_paths[p, i, t - 1]
                sample = sampled_numbers[p, i, t]
                curr = mu[i] * dt + sigma[i] * sqrt(dt) * sample
                motion_paths[p, i, t] = prev + curr

    return motion_paths


def generate_BM(nb_paths, dim, nb_steps, dt, x0=None):
    z = [0 for _ in range(dim)]
    o = [1 for _ in range(dim)]
    return generate_BM_drift_diffusion(nb_paths, dim, nb_steps, dt, z, o, x0)


# ==============================================================================
# this is needed for computing the loss with the true conditional expectation
def compute_loss(
    X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10, weight=0.5, M_obs=None
):
    """
    compute the loss of the true conditional expectation, as in
    model.compute_loss
    """
    if M_obs is None:
        inner = (
            2 * weight * np.sqrt(np.sum((X_obs - Y_obs) ** 2, axis=1) + eps)
            + 2 * (1 - weight) * np.sqrt(np.sum((Y_obs_bj - Y_obs) ** 2, axis=1) + eps)
        ) ** 2
    else:
        inner = (
            2 * weight * np.sqrt(np.sum(M_obs * (X_obs - Y_obs) ** 2, axis=1) + eps)
            + 2
            * (1 - weight)
            * np.sqrt(np.sum(M_obs * (Y_obs_bj - Y_obs) ** 2, axis=1) + eps)
        ) ** 2
    outer = np.sum(inner / n_obs_ot)
    return outer / batch_size


DATASETS = {
    "FBM": FracBM,
    "RBM": ReflectedBM,
    "Rectangle": Rectangle,
    "BMWeights": BMWeights,
    "Ball2D_BM": Ball2D_BM,
}


hyperparam_test_stock_models = {
    "drift": 0.2,
    "volatility": 0.3,
    "mean": 0.5,
    "poisson_lambda": 3.0,
    "speed": 0.5,
    "correlation": 0.5,
    "nb_paths": 10,
    "nb_steps": 100,
    "S0": 1,
    "maturity": 1.0,
    "dimension": 1,
}


def draw_stock_model(stock_model_name):
    # TODO need this? and the hyperparams?
    hyperparam_test_stock_models["model_name"] = stock_model_name
    stockmodel = DATASETS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths, dt = stockmodel.generate_paths()
    filename = "{}.pdf".format(stock_model_name)

    # draw a path
    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    cond_exp = np.zeros(len(one_path))
    cond_exp[0] = hyperparam_test_stock_models["S0"]
    for i in range(1, len(one_path)):
        if i % 3 == 0:
            cond_exp[i] = one_path[i]
        else:
            cond_exp[i] = cond_exp[i - 1] * exp(
                hyperparam_test_stock_models["drift"] * dt
            )

    plt.plot(dates, one_path, label="stock path")
    plt.plot(dates, cond_exp, label="conditional expectation")
    plt.legend()
    plt.savefig(filename)
    plt.close()
