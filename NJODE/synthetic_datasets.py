"""
author: Florian Krach & Calypso Herrera

code to generate synthetic data from stock-model SDEs
"""

# ==============================================================================
from math import sqrt, exp, isnan, gamma, pi, erf, isclose
import numpy as np
import tqdm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import copy, os
from fbm import fbm, fgn  # import fractional brownian motion package

import matplotlib.animation as animation


# ==============================================================================
# CLASSES
class StockModel:
    """
    mother class for all stock models defining the variables and methods shared
    amongst all of them, some need to be defined individually
    """

    def __init__(self, drift, volatility, S0, nb_paths, nb_steps,
                 maturity, sine_coeff, **kwargs):
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

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None,
                         **kwargs):
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
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10*delta_t:
                break
            if obs_time <= current_time:
                continue
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
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

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(X_obs=X_obs, Y_obs=Y[i_obs],
                                           Y_obs_bj=Y_bj[i_obs],
                                           n_obs_ot=n_obs_ot[i_obs],
                                           batch_size=batch_size, weight=weight)

            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_, current_time)
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

    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5, M=None, mult=None):
        if mult is not None and mult > 1:
            bs, dim = start_X.shape
            _dim = round(dim / mult)
            X = X[:, :_dim]
            start_X = start_X[:, :_dim]
            if M is not None:
                M = M[:, :_dim]

        loss, _, _ = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=True, weight=weight, M=M)
        
        return loss


class FracBM(StockModel):
    """
    Implementing FBM via FBM package
    """

    def __init__(self, nb_paths, nb_steps, S0, maturity, hurst,
                 method="daviesharte", **kwargs):
        """Instantiate the FBM"""
        super().__init__(
            drift=None, volatility=None, S0=S0, nb_paths=nb_paths,
            nb_steps=nb_steps, maturity=maturity, sine_coeff=None,)
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

    def r_H(self, t, s):
        return 0.5 * (t**(2*self.hurst) + s**(2*self.hurst) -
                      np.abs(t-s)**(2*self.hurst))

    def get_cov_mat(self, times):
        m = np.array(times).reshape((-1,1)).repeat(len(times), axis=1)
        return self.r_H(m, np.transpose(m))

    def next_cond_exp(self, y, delta_t, current_t):
        t = current_t+delta_t
        next_y = np.zeros_like(y)
        for ii in range(y.shape[0]):
            if self.obs_cov_mat_inv[ii] is not None:
                r = self.r_H(np.array(self.observed_t[ii]), t)
                next_y[ii] = np.dot(r, np.matmul(
                    self.obs_cov_mat_inv[ii], np.array(self.observed_X[ii])))
        return next_y

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None,
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
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

        assert self.dimensions == 1, "cond. exp. computation of FBM only for 1d"
        assert self.S0 == 0, "cond. exp. computation of FBM only for S0=0"

        bs = start_X.shape[0]
        self.observed_t = [[] for x in range(bs)]
        self.observed_X = [[] for x in range(bs)]
        self.obs_cov_mat = [None for x in range(bs)]
        self.obs_cov_mat_inv = [None for x in range(bs)]

        y = start_X
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
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

            # add to observed
            for j, ii in enumerate(i_obs):
                self.observed_t[ii].append(obs_time)
                self.observed_X[ii].append(X_obs[j, 0])
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
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
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

    def generate_paths(self, start_X=None):
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :, 0] = self.S0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                fgn_sample = fgn(n=self.nb_steps, hurst=self.hurst,
                             length=self.maturity, method=self.method)
                spot_paths[i, j, 1:] = np.cumsum(fgn_sample)+spot_paths[i, j, 0]
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt

# TODO turn into 
class ReflectedBM(StockModel):
    def __init__(self, mu, sigma, max_terms, lb, ub, max_z, nb_paths, dimensions, nb_steps, maturity, use_approx_paths_technique):
        assert lb < ub
        assert lb + mu <= ub
        assert ub - mu >= lb # TODO I think these are needed

        self.mu = mu
        self.sigma = sigma
        self.max_terms = max_terms
        self.lb = lb
        self.ub = ub
        self.max_z = max_z
        self.nb_paths = nb_paths
        self.dimensions = dimensions
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.use_approx_paths_technique = use_approx_paths_technique
        self.norm_cdf = lambda x: 0.5 * (1 + erf((x - self.mu) / self.sigma * sqrt(2)))

    def _get_bounds(self, a, b, k):
        return a + k*(b - a), b + k*(b-a)

    def _proj(self, x):
        # TODO prove it works. Florian's idea
        # TODO could probably pick z or k smartly. For sure should expand inside-out for k though
        if self.lb < x < self.ub:
            return x

        for z in range(-self.max_z, self.max_z + 1):
            k = 2*z + 1
            l, u = self._get_bounds(self.lb, self.ub, k)
            if l <= x <= u:
                return self.ub - (x - (self.lb + k*( self.ub- self.lb)))
            
            k = 2*z
            l, u = self._get_bounds(self.lb, self.ub, k)
            if l <= x <= u:
                return self.lb + (x - (self.lb + k*(self.ub - self.lb)))

        # If wasn't able to project with the above logic, need to expand approximation
        raise Exception(f"Not maz_z of {self.max_z} not enough to approximate projection of {x}")

    def _generate_approx_paths(self, x0):
        # Generate approximate path by manually "projecting" onto the boundaries. This is technically
        # an approximation since it has positive probability of landing on the boundary.
        spot_paths = np.empty((self.nb_paths, self.dimensions, self.nb_steps + 1))
        spot_paths[:, :, 0] = x0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                scale = 1.0 * self.maturity / (self.nb_steps + 1)
                raw_paths = scale * np.cumsum(np.random.normal(self.mu, self.sigma, self.nb_steps)) + spot_paths[i, j, 0]
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

        return NotImplementedError()

    def generate_paths(self, x0):
        assert self.lb <= x0 <= self.ub

        if self.use_approx_paths_technique:
            return self._generate_approx_paths(x0)
        else:
            return self._generate_true_paths(x0)
        




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

        assert c <= x <= d
        assert c <= x0 <= d
        assert t0 < t
        
        coeff = 1.0/(sigma*sqrt(2*pi*(t - t0)))
        S1 = coeff * sum(
            exp((2*mu*n*(c - d) / (sigma**2)) \
            + (-(((x + 2*n*(d - c) - x0 - mu*(t - t0))**2) / (2*(sigma**2)*(t - t0)))))
            for n in range(ninf, pinf)
        )

        S2 = coeff * sum(
            exp(-(2 * mu * (n * d - (n + 1) * c + x0)) / sigma**2 \
            + (-((2 * n * d - 2 * (n + 1) * c + x0 + x - mu * (t - t0))**2)/(2 * sigma**2 * (t - t0))))
            for n in range(ninf, pinf)
        )

        coeff = (2*mu) / sigma ** 2

        S3 = 0
        for n in range(0, pinf):
            t2 = (1 - phi((mu*(t - t0) + 2*n*d - 2*(n + 1)*c + x0 + x) / (sigma*sqrt(t - t0))))
            # avoid danger because t1 can become very large, even if s2 is exactly 0
            # note the numerator of t2 [0, 1] so we don't really have to worry about it overflowing
            if isclose(t2, 0): 
                # adding 0, since t2 will make the whole term 0
                continue 
            t1 = (2*mu*(n*d - (n + 1)*c + x)) / sigma**2
            S3 += exp(t1) * t2
        S3 = -coeff * S3

        # Same technique as above
        S4 = 0
        for n in range(0, pinf):
            t2 = phi((mu*(t - t0) - 2*(n + 1)*d + 2*n*c + x0 + x) / (sigma * sqrt(t - t0)))
            if isclose(t2, 0):
                continue
            t1 = 2*mu*(n*c - (n + 1)*d + x) / sigma**2
            S4 += exp(t1) * t2
        S4 = coeff * S4


        return S1 + S2 + S3 + S4
    
    def next_cond_exp(self):
        raise NotImplementedError
    
    def compute_cond_exp(self):
        raise NotImplementedError



class Ball(StockModel):
    pass

class VertexApproach(StockModel):
    pass



# ==============================================================================
# this is needed for computing the loss with the true conditional expectation
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5, M_obs=None):
    """
    compute the loss of the true conditional expectation, as in
    model.compute_loss
    """
    if M_obs is None:
        inner = (2 * weight * np.sqrt(np.sum((X_obs - Y_obs) ** 2, axis=1) + eps) +
                 2 * (1 - weight) * np.sqrt(np.sum((Y_obs_bj - Y_obs) ** 2, axis=1)
                                            + eps)) ** 2
    else:
        inner = (2 * weight * np.sqrt(
            np.sum(M_obs * (X_obs - Y_obs)**2, axis=1) + eps) +
                 2 * (1 - weight) * np.sqrt(
                    np.sum(M_obs * (Y_obs_bj - Y_obs)**2, axis=1) + eps))**2
    outer = np.sum(inner / n_obs_ot)
    return outer / batch_size


# ==============================================================================
# dict for the supported stock models to get them from their name
DATASETS = {
    "FBM": FracBM,
}
# ==============================================================================


hyperparam_test_stock_models = {
    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5, "poisson_lambda": 3.,
    'speed': 0.5, 'correlation': 0.5, 'nb_paths': 10, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1}


def draw_stock_model(stock_model_name):
    hyperparam_test_stock_models['model_name'] = stock_model_name
    stockmodel = DATASETS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths, dt = stockmodel.generate_paths()
    filename = '{}.pdf'.format(stock_model_name)

    # draw a path
    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    cond_exp = np.zeros(len(one_path))
    cond_exp[0] = hyperparam_test_stock_models['S0']
    cond_exp_const = hyperparam_test_stock_models['S0']
    for i in range(1, len(one_path)):
        if i % 3 == 0:
            cond_exp[i] = one_path[i]
        else:
            cond_exp[i] = cond_exp[i - 1] * exp(
                hyperparam_test_stock_models['drift'] * dt)

    plt.plot(dates, one_path, label='stock path')
    plt.plot(dates, cond_exp, label='conditional expectation')
    plt.legend()
    plt.savefig(filename)
    plt.close()
