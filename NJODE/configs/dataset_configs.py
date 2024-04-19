import copy

import numpy as np
import torch


def merge_dicts(a, b):
    return {**a, **b}


BASE_DATA_DICTS = {
    "FBM_1_dict": {
        "model_name": "FBM",
        "nb_paths": 100,
        "nb_steps": 100,
        "S0": 0,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "return_vol": False,
        "hurst": 0.05,
        "FBMmethod": "daviesharte",
    },
    "RBM_1_dict": {
        "model_name": "RBM",
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 0,
        "sigma": 1,
        "lb": 0,
        "ub": 1,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "Rectangle_1_dict": {
        "model_name": "Rectangle",
        "nb_paths": 10,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 0.5,
        "sigma_x": 1,
        "mu_y": 1,
        "sigma_y": 2,
        "max_z": 8,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 5,
        "length": 1,
    },
    "Triangle_BM_weights_1_dict": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[1, 0], [0, 1]],
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "Ball2D_BM_1_dict": {
        "model_name": "Ball2D_BM",
        "max_radius": 10,
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
}

TEST_DATA_DICTS = {
    "RBM_DONT_HIT": {  # for comparison to normal NJODE, is it useful though?
        "model_name": "RBM",
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 0,
        "sigma": 1,
        "lb": 0,
        "ub": 20,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "RBM_STANDARD": {
        "model_name": "RBM",
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 0,
        "sigma": 1,
        "lb": 0,
        "ub": 1,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "RBM_MORE_BOUNCES": {
        "model_name": "RBM",
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 1,
        "sigma": 1,
        "lb": 0,
        "ub": 2,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "RECTANGLE_STANDARD": {
        "model_name": "Rectangle",
        "nb_paths": 10,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 0,
        "sigma_x": 1,
        "mu_y": 0,
        "sigma_y": 1,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 1,
        "length": 1,
    },
    "RECTANGLE_WIDER_WITH_MU": {
        "model_name": "Rectangle",
        "nb_paths": 10,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 0.5,
        "sigma_x": 1,
        "mu_y": 1,
        "sigma_y": 2,
        "max_z": 8,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 5,
        "length": 1,
    },
    "BM_WEIGHTS_RECTANGLE_STANDARD": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "BM_WEIGHTS_SIMPLEX2D": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[1, 0], [0, 1]],
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "BM_WEIGHTS_SIMPLEX3D": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 3,
        "obs_perc": 0.1,
    },
    "BALL2D_STANDARD": {
        "model_name": "Ball2D_BM",
        "max_radius": 10,
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "BALL2D_LARGE": {
        "model_name": "Ball2D_BM",
        "max_radius": 10,
        "nb_paths": 6,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
}


DATA_DICTS = merge_dicts(BASE_DATA_DICTS, TEST_DATA_DICTS)


def opt_Ball2D_proj(ball2d_data_dict_name):
    def optimal_proj(x):
        R = DATA_DICTS[ball2d_data_dict_name]["max_radius"]
        norm = torch.norm(x)
        if norm**2 <= R**2:
            return x
        else:
            return torch.min(x, (1 / norm) * x)

    return optimal_proj


def opt_RBM_proj(RBM_data_dict_name):
    def optimal_proj(x):
        return torch.clamp(
            x,
            DATA_DICTS[RBM_data_dict_name]["lb"],
            DATA_DICTS[RBM_data_dict_name]["ub"],
        )

    return optimal_proj


def opt_rect_proj(rect_data_dict_name):
    data_dict = DATA_DICTS[rect_data_dict_name]
    lb_x, lb_y = 0, 0
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]
    lower = torch.tensor([lb_x, lb_y])
    upper = torch.tensor([ub_x, ub_y])

    def optimal_proj(x):
        return torch.clamp(x, lower, upper)

    return optimal_proj


def opt_simplex_proj(Y):
    # TODO don't forget to cite them in my thesis
    # Algorithm by Yunmei Chen and Xiaojing Ye (2011), and their
    # implementation at https://mathworks.com/matlabcentral/fileexchange/30332
    to_sub = torch.zeros(len(Y))
    for k, y in enumerate(Y):
        y = y.clone().detach().numpy()
        # Sort descending
        sorted = copy.deepcopy(y)
        sorted.sort()
        sorted = np.flip(sorted)
        n = len(y)
        tmpsum = 0
        for i in range(0, n - 1):
            tmpsum += sorted[i]
            tmax = (tmpsum - 1) / (i + 1)
            if tmax >= sorted[i + 1]:
                break
        else:
            tmax = (tmpsum + y[n - 1] - 1) / n

        to_sub[k] = tmax

    subtracted = Y - to_sub.unsqueeze(1)
    res = torch.clamp(subtracted, torch.zeros_like(Y))
    return res

    # sorted = copy.deepcopy(y)
    # sorted.sort()
    # sorted = np.flip(sorted)
    # n = len(y)
    # tmpsum = 0
    # for i in range(0, n - 1):
    #     tmpsum += sorted[i]
    #     tmax = (tmpsum - 1) / (i + 1)
    #     if tmax >= sorted[i + 1]:
    #         break
    # else:
    #     tmax = (tmpsum + y[n - 1] - 1) / n

    # res = np.clip(y - tmax, 0, None)
    return res


OPTIMAL_PROJECTION_FUNCS = {
    "RBM_1_dict": opt_RBM_proj("RBM_1_dict"),
    "RBM_STANDARD": opt_RBM_proj("RBM_STANDARD"),
    "RBM_MORE_BOUNCES": opt_RBM_proj("RBM_MORE_BOUNCES"),
    "Rectangle_1_dict": opt_rect_proj("Rectangle_1_dict"),
    "RECTANGLE_STANDARD": opt_rect_proj("RECTANGLE_STANDARD"),
    "RECTANGLE_WIDER_WITH_MU": opt_rect_proj("RECTANGLE_WIDER_WITH_MU"),
    "BM_WEIGHTS_RECTANGLE_STANDARD": opt_rect_proj("RECTANGLE_STANDARD"),
    "BM_WEIGHTS_SIMPLEX2D": opt_simplex_proj,
    "BM_WEIGHTS_SIMPLEX3D": opt_simplex_proj,
    "Ball2D_BM_1_dict": opt_Ball2D_proj("Ball2D_BM_1_dict"),
    "BALL2D_STANDARD": opt_Ball2D_proj("BALL2D_STANDARD"),
    "BALL2D_LARGE": opt_Ball2D_proj("BALL2D_LARGE"),
}


def get_ccw_rectangle_vertices(rect_data_dict_name):
    data_dict = DATA_DICTS[rect_data_dict_name]
    lb_x, lb_y = 0, 0
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]

    # counterclockwise, starting from bottom-left
    v1, v2, v3, v4 = (lb_x, lb_y), (ub_x, lb_y), (ub_x, ub_y), (lb_x, ub_y)

    return torch.tensor([v1, v2, v3, v4]).float()


def easy_vertices(dataset_name):
    return torch.tensor(DATA_DICTS[dataset_name]["vertices"]).float()


VERTEX_APPROACH_VERTICES = {
    "Rectangle_1_dict": get_ccw_rectangle_vertices("Rectangle_1_dict"),
    "RECTANGLE_STANDARD": get_ccw_rectangle_vertices("RECTANGLE_STANDARD"),
    "RECTANGLE_WIDER_WITH_MU": get_ccw_rectangle_vertices("RECTANGLE_WIDER_WITH_MU"),
    "Triangle_BM_weights_1_dict": easy_vertices("Triangle_BM_weights_1_dict"),
    "BM_WEIGHTS_RECTANGLE_STANDARD": easy_vertices("BM_WEIGHTS_RECTANGLE_STANDARD"),
    "BM_WEIGHTS_SIMPLEX2D": easy_vertices("BM_WEIGHTS_SIMPLEX2D"),
    "BM_WEIGHTS_SIMPLEX3D": easy_vertices("BM_WEIGHTS_SIMPLEX3D"),
}


def standard_2_norm_for_lb_ub(Y, lb, ub):
    lb_norm = torch.norm(Y - float(lb), 2, dim=1)
    ub_norm = torch.norm(Y - float(ub), 2, dim=1)
    closest = torch.min(lb_norm, ub_norm)
    mask = torch.ones_like(closest)
    for i in range(len(Y)):
        if lb <= Y[i] <= ub:
            mask[i] = 0

    return mask * closest


def RBM_pen_func(data_dict):
    def pen(Y):
        return standard_2_norm_for_lb_ub(
            Y,
            data_dict["lb"],
            data_dict["ub"],
        )

    return pen


def rect_pen_func(Y, data_dict):
    lb_x, lb_y = 0, 0
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]

    # Separable so just project each coordinate independently
    projected = Y.clone().detach()
    for i in range(len(Y)):
        if not (lb_x <= Y[i][0] <= ub_x):
            projected[i][0] = torch.min(
                torch.norm(Y[i][0] - lb_x, 1), torch.norm(Y[i][0] - ub_x, 1)
            )
        if not (lb_y <= Y[i][1] <= ub_y):
            projected[i][1] = torch.min(
                torch.norm(Y[i][1] - lb_y, 1), torch.norm(Y[i][1] - ub_y, 1)
            )

    return torch.norm(Y - projected, 2, dim=1)


def zero_pen_func(Y):
    return torch.norm(Y - Y, 2, dim=1)
