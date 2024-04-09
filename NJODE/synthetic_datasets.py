"""
author: Florian Krach & Calypso Herrera

code to generate synthetic data from stock-model SDEs
"""

import copy
import os
from math import erf, exp, isclose, pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fbm import fgn
from scipy.integrate import quad
from scipy.special import softmax


# ==============================================================================
# CLASSES
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

        for i, obs_time in enumerate(times):
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
                y = self.next_cond_exp(y, delta_t_, current_time, **kwargs)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

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
        path_idx=None,
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
            path_idx=path_idx,
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
        max_z,
        nb_paths,
        dimension,
        nb_steps,
        maturity,
        use_approx_paths_technique,
        use_numerical_cond_exp,
        **kwargs,
    ):
        assert lb < ub
        assert lb + mu <= ub
        assert ub - mu >= lb  # TODO I think these are needed

        self.mu = mu
        self.sigma = sigma
        self.max_terms = max_terms
        self.lb = lb
        self.ub = ub
        self.max_z = max_z
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
        self.norm_cdf = lambda x: 0.5 * (1 + erf((x - self.mu) / self.sigma * sqrt(2)))
        self.masked = True
        self.track_obs_cov_mat = False

    def _get_bounds(self, a, b, k):
        return a + k * (b - a), b + k * (b - a)

    def _in_shape(self, x):
        return self.lb < x < self.ub

    def _proj(self, x):
        # TODO could probably pick z or k smartly. For sure should expand inside-out for k though
        # NOTE this isn't really a projection. We're trying to figure out how often it bounces
        if self._in_shape(x):
            return x

        for z in range(-self.max_z, self.max_z + 1):
            k = 2 * z + 1
            l, u = self._get_bounds(self.lb, self.ub, k)
            if l <= x <= u:
                return self.ub - (x - (self.lb + k * (self.ub - self.lb)))

            k = 2 * z
            l, u = self._get_bounds(self.lb, self.ub, k)
            if l <= x <= u:
                return self.lb + (x - (self.lb + k * (self.ub - self.lb)))

        # If wasn't able to project with the above logic, need to expand approximation
        raise Exception(
            f"Not maz_z of {self.max_z} not enough to approximate projection of {x}"
        )

    def _generate_approx_paths(self, x0):
        # Generate approximate path by manually "projecting" onto the boundaries. This is technically
        # an approximation since it has positive probability of landing on the boundary.
        spot_paths = np.empty((self.nb_paths, self.dimensions, self.nb_steps + 1))
        spot_paths[:, :, 0] = x0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                scale = 1.0 * self.maturity / (self.nb_steps + 1)
                raw_paths = (
                    scale
                    * np.cumsum(np.random.normal(self.mu, self.sigma, self.nb_steps))
                    + spot_paths[i, j, 0]
                )
                projected_paths = np.array([self._proj(x) for x in raw_paths])
                spot_paths[i, j, 1:] = projected_paths

        return spot_paths

    def _generate_true_paths(self, x0):
        # TODO will have to use rejection sampling or some MCMC method
        # https://jaketae.github.io/study/rejection-sampling/
        # TODO can write about this: trying to find a function that covers it, for rejection sampling, but there are asymptotes so not possible
        # from scipy.stats import norm
        # opts = (0.5, 0.35, 10, 1.0, 2.0)
        # mu, sigma, max_terms, lb, ub = opts
        # asdf = c(*opts)
        # x0 = (lb + ub)/2
        # t = 0.5
        # t0 = 0
        # # asdf.reflected_bm_pdf((lb + ub)/2, t, x0, t0)
        # p = lambda x: asdf.reflected_bm_pdf(x, t, x0, t0)
        # q = lambda x: norm.pdf(x, mu*(t - t0) + x0, sigma*sqrt(t - t0))
        # x = np.linspace(lb, ub, 100)
        # fig, ax = plt.subplots()
        # ax.plot(x, [p(t) for t in x], color='b')
        # ax.plot(x, [q(t) for t in x], color='r')
        # ax.plot(x, [2*q(t) for t in x], color='g') # TODO this works pretty well, but if I increase sigma it eventually gets worse. And 1/sigma, 1/(sigma**2) also wrong. Also for very small values of mu. Maybe can argue like that std dev larger than (ub - lb) doesn't make sense?
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

    def reflected_bm_pdf(self, x, t, x0, t0):
        # Follow eqn 11 of
        # https://link.springer.com/content/pdf/10.1023/B:CSEM.0000049491.13935.af.pdf
        # TODO can write about this in paper
        # NOTE: to avoid overflows, need to bring the exps into one, within each sum.
        # Additionally, sometimes need to do check to see if multiply by 0 (short cut),
        # otherwise evaluating a huge number
        # ########
        # n = pinf
        # t1 = (2*mu*(n*d - (n + 1)*c + x)) / sigma**2
        # t2 = (1 - phi((mu*(t - t0) + 2*n*d - 2*(n + 1)*c + x0 + x) / (sigma*sqrt(t - t0))))
        # print(t1, t2)
        # print(t2 * t1)
        # S3 = -coeff * sum(
        #     exp( (2*mu*(n*d - (n + 1)*c + x)) / sigma**2) \
        #     * (1 - phi((mu*(t - t0) + 2*n*d - 2*(n + 1)*c + x0 + x) / (sigma*sqrt(t - t0))))
        #     for n in range(0, pinf)
        # )

        # opts = (0.2, 0.02, 10, 4, 10)

        # x0 = (lb + ub)/2
        # t = 1
        # t0 = 0
        # asdf.reflected_bm_pdf((lb + ub)/2, t, x0, t0)

        mu = self.mu
        sigma = self.sigma
        pinf = self.max_terms
        ninf = -self.max_terms
        c = self.lb
        d = self.ub
        phi = self.norm_cdf

        # TODO hack fix when e.g. at 0.999999994 and c is 1
        #      should put this into a separate function
        rel_tol = 10e-4
        if not (c <= x) and isclose(x, c, rel_tol=rel_tol):
            x = c
        elif not (x <= d) and isclose(x, d, rel_tol=rel_tol):
            x = d

        if not (c <= x0) and isclose(x0, c, rel_tol=rel_tol):
            x0 = c
        elif not (x0 <= d) and isclose(x0, d, rel_tol=rel_tol):
            x0 = d

        # TODO consider what the pdf should be when these are not true.
        # If x (what we're asking for) is outside, then it's not possible.
        # If the previous point is outside? then the prob is undefined.
        # Just round it to closest of c or d and return that density?
        if not (c <= x <= d):
            return 0.0

        if not (c <= x0 <= d):  # TODO def rethink this
            x0 = c if abs(x0 - c) < abs(x0 - d) else d

        assert c <= x <= d
        assert c <= x0 <= d
        assert t > t0

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
        # TODO this ^ term seems to be negative sometimes and makes the whole pdf negative

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

        fin = max(fin, 0)  # TODO remove this hack fix
        # assert fin >= 0  # TODO should be positive : "the Green’s function expansion
        # enjoys superior convergence properties and prevents the emergence of negative
        # conditional densities"

        return fin

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        assert delta_t > 0
        cond_exp = (
            self._compute_numerical_next_cond_exp(y, delta_t, current_t)
            if self.use_numerical_cond_exp
            else self._compute_true_next_cond_exp(y, delta_t, current_t)
        )

        # TODO fix this, will have to uncomment once it's fixed
        # assert np.all(cond_exp >= self.lb) and np.all(cond_exp <= self.ub)

        return cond_exp

    def _compute_numerical_next_cond_exp(self, y, delta_t, current_t):
        t0 = current_t
        t = current_t + delta_t
        out = np.zeros_like(y)
        for i, x0 in enumerate(y):
            try:
                integrand = lambda x: x * self.reflected_bm_pdf(x, t, x0, t0)
                out[i] = quad(integrand, self.lb, self.ub)[0]
            except Exception as e:
                print(e)

        return out

    def _compute_true_next_cond_exp(self, y, delta_t, current_t):
        raise NotImplementedError


class Rectangle(StockModel):
    def __init__(
        self,
        width,
        length,
        base_point,
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        max_terms,
        max_z,
        nb_paths,
        dimension,
        nb_steps,
        maturity,
        use_approx_paths_technique,
        use_numerical_cond_exp,
        **kwargs,
    ):
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
        self.max_z = max_z
        self.nb_paths = nb_paths
        self.dimension = dimension
        self.dimensions = 1  # TODO not sure what this is for/if useful, same for RBM
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
        self.base_point = base_point

        self.rbm_x = ReflectedBM(
            mu=mu_x,
            sigma=sigma_x,
            max_terms=max_terms,
            lb=self.base_point[0],
            ub=self.base_point[0] + width,
            max_z=max_z,
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
            lb=self.base_point[1],
            ub=self.base_point[1] + length,
            max_z=max_z,
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
        path_x = paths_x[0]
        path_y = paths_y[0]
        xs = np.arange(path_x.shape[1])
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(xs, path_x, path_y)
        plt.savefig("rect_dataset_plot.png")
        plt.close(fig)

    def generate_paths(self, x0=(None, None)):
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


class Ball(StockModel):
    pass


class BMWeights(StockModel):
    def __init__(
        self,
        vertices,
        should_compute_approx_cond_exp_paths,
        mu,
        sigma,
        dimension,
        nb_paths,
        nb_steps,
        maturity,
        **kwargs,
    ):
        assert dimension == len(vertices[0]) == len(mu) == len(sigma)
        self.vertices = np.array(vertices)
        self.mu = mu
        self.sigma = sigma
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

    def compute_cond_exp_approx(self, s, t, path_idx):
        s_idx = round(s / self.dt, 5)  # avoid floating point errors, e.g. 5.0000001
        assert float.is_integer(s_idx), "Strategy to get index from times failed :("
        s_idx = int(s_idx)

        fp = self.paths_dir / "motion_paths.npy"
        self.motion_paths = np.load(fp)

        W_tilde_s = self.motion_paths[path_idx, :, s_idx]  # what we condition on
        n = len(self.vertices)

        def sample_cond_exp(incr):
            # TODO this can be cleaned up a lot, e.g. by bringing the exps inside
            cexpect = np.zeros(n)
            for i in range(n):
                num = np.exp(incr[i]) * np.exp(W_tilde_s[i])
                denom = 0
                for j in range(n):
                    denom += np.exp(incr[j]) * np.exp(W_tilde_s[j])
                cexpect[i] = num / denom

            return cexpect

        # Now do Monte Carlo
        N = 10  # TODO pick a smarter value, see message on 09.04
        weight_cond_exp = np.zeros(n)
        for _ in range(N):
            increment_sample = np.random.normal(0, t - s, size=n)
            weight_cond_exp += (1 / N) * sample_cond_exp(increment_sample)

        # TODO shouldn't the point already come out this way, so I shouldn't have to wrap in another array?
        #      which function is wrong?
        res = np.array([self.weights_to_point(weight_cond_exp)])

        return res

    def generate_paths(self, x0=None):
        sampled_numbers = np.random.standard_normal(
            (self.nb_paths, len(self.vertices), self.nb_steps)
        )
        for i in range(self.dimension):
            sampled_numbers[:, i, :] = (
                self.mu[i] + self.sigma[i] * sampled_numbers[:, i, :]
            )
        spot_motion_paths = np.cumsum(sampled_numbers * sqrt(self.dt), axis=2)
        spot_weight_paths = softmax(spot_motion_paths, axis=1)

        spot_paths = np.zeros((self.nb_paths, self.dimension, self.nb_steps + 1))
        if x0 is None:
            # Take a point inside the convex hull (average point)
            x0 = sum(self.vertices) / len(self.vertices)
        spot_paths[:, :, 0] = x0

        # Convert to points
        for p in range(self.nb_paths):
            for s in range(self.nb_steps):
                w = spot_weight_paths[p, :, s]
                pt = self.weights_to_point(w)
                spot_paths[p, :, s + 1] = pt  # add 1 because we prepended x0

        # TODO assert that it's all in the shape?

        if self.should_compute_approx_cond_exp_paths:
            # TODO x0 might have to be a weight
            # TODO find a better way to save things
            p = self.paths_dir
            p.mkdir(parents=True, exist_ok=True)

            self.motion_paths = spot_motion_paths
            fp = p / "motion_paths.npy"
            np.save(fp, self.motion_paths)

        return spot_paths, self.dt

    def next_cond_exp(self, y, delta_t, current_t, **kwargs):
        if self.should_compute_approx_cond_exp_paths:
            return self.compute_cond_exp_approx(
                current_t, delta_t + current_t, kwargs["path_idx"]
            )
        else:
            raise NotImplementedError()


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


# ==============================================================================
# dict for the supported stock models to get them from their name
DATASETS = {
    "FBM": FracBM,
    "RBM": ReflectedBM,
    "Rectangle": Rectangle,
    "BMWeights": BMWeights,
}
# ==============================================================================


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
