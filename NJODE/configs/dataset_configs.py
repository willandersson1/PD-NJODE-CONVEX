import copy

import numpy as np
import torch


def get_rectangle_bounds(width, length, offset=(0, 0)):
    o_x, o_y = offset
    lb_x = o_x - width / 2
    ub_x = lb_x + width
    lb_y = o_y - length / 2
    ub_y = lb_y + length

    return lb_x, ub_x, lb_y, ub_y


DATA_DICTS = {
    "RBM_DONT_HIT": {  # for comparison to normal NJODE, is it useful though?
        "model_name": "RBM",
        "nb_paths": 200,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 0,
        "sigma": 1,
        "lb": 0,
        "ub": 20,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "RBM_STANDARD": {
        "model_name": "RBM",
        "nb_paths": 200,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 0,
        "sigma": 1,
        "lb": -0.5,
        "ub": 0.5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "RBM_MORE_BOUNCES": {
        "model_name": "RBM",
        "nb_paths": 200,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 1.5,
        "sigma": 1.5,
        "lb": -0.5,
        "ub": 0.5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "RECTANGLE_STANDARD": {
        "model_name": "Rectangle",
        "nb_paths": 100,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 0,
        "sigma_x": 1,
        "mu_y": 0,
        "sigma_y": 1,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 1,
        "length": 1,
    },
    "RECTANGLE_WIDER_WITH_MU": {
        "model_name": "Rectangle",
        "nb_paths": 100,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 0.6,
        "sigma_x": 2,
        "mu_y": -0.1,
        "sigma_y": 1,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 5,
        "length": 1,
    },
    "BM_WEIGHTS_RECTANGLE_STANDARD": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [
            [get_rectangle_bounds(1, 1)[0], get_rectangle_bounds(1, 1)[2]],
            [get_rectangle_bounds(1, 1)[1], get_rectangle_bounds(1, 1)[2]],
            [get_rectangle_bounds(1, 1)[1], get_rectangle_bounds(1, 1)[3]],
            [get_rectangle_bounds(1, 1)[0], get_rectangle_bounds(1, 1)[3]],
        ],
        "nb_paths": 100,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "BM_WEIGHTS_SIMPLEX2D": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[1, 0], [0, 1]],
        "nb_paths": 500,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "BM_WEIGHTS_SIMPLEX3D": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "nb_paths": 500,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 3,
        "obs_perc": 0.1,
    },
    "BALL2D_STANDARD": {
        "model_name": "Ball2D_BM",
        "max_radius": 1,
        "nb_paths": 500,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "BALL2D_LARGE": {
        "model_name": "Ball2D_BM",
        "max_radius": 10,
        "nb_paths": 500,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
}


def opt_Ball2D_proj(ball2d_data_dict_name):
    R = DATA_DICTS[ball2d_data_dict_name]["max_radius"]

    def optimal_proj(X):
        projected = X.clone().detach()
        for i in range(len(X)):
            norm = torch.norm(X[i], 2)
            if norm <= R:
                projected[i] = X[i]
            else:
                projected[i] = (1 / norm) * X[i]

        return projected

    return optimal_proj


def opt_RBM_proj(RBM_data_dict_name):
    lb = DATA_DICTS[RBM_data_dict_name]["lb"]
    ub = DATA_DICTS[RBM_data_dict_name]["ub"]

    def optimal_proj(X):
        return torch.clamp(X, lb, ub)

    return optimal_proj


def opt_rect_proj(rect_data_dict_name):
    data_dict = DATA_DICTS[rect_data_dict_name]
    lb_x, ub_x, lb_y, ub_y = get_rectangle_bounds(
        data_dict["width"], data_dict["length"]
    )

    def optimal_proj(X):
        projected = X.clone().detach()
        for i in range(len(X)):
            in_x = lb_x <= X[i][0] <= ub_x
            in_y = lb_y <= X[i][1] <= ub_y
            if in_x and in_y:
                projected[i] = X[i]
                continue
            if not in_x:
                projected[i][0] = torch.min(
                    torch.norm(X[i][0] - lb_x, 1), torch.norm(X[i][0] - ub_x, 1)
                )
            if not in_y:
                projected[i][1] = torch.min(
                    torch.norm(X[i][1] - lb_y, 1), torch.norm(X[i][1] - ub_y, 1)
                )

        return projected

    return optimal_proj


def opt_simplex_proj(Y):
    # Algorithm by Yunmei Chen and Xiaojing Ye (2011), and their
    # implementation at https://mathworks.com/matlabcentral/fileexchange/30332
    to_sub = torch.zeros(len(Y))
    for k, y_ in enumerate(Y):
        y = y_.clone().detach().numpy()
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
            tmax = (tmpsum + sorted[n - 1] - 1) / n
        to_sub[k] = tmax

    subtracted = Y - to_sub.unsqueeze(1)
    res = torch.clamp(subtracted, torch.zeros_like(Y))
    return res


OPTIMAL_PROJECTION_FUNCS = {
    "RBM_STANDARD": opt_RBM_proj("RBM_STANDARD"),
    "RBM_MORE_BOUNCES": opt_RBM_proj("RBM_MORE_BOUNCES"),
    "RECTANGLE_STANDARD": opt_rect_proj("RECTANGLE_STANDARD"),
    "RECTANGLE_WIDER_WITH_MU": opt_rect_proj("RECTANGLE_WIDER_WITH_MU"),
    "BM_WEIGHTS_RECTANGLE_STANDARD": opt_rect_proj("RECTANGLE_STANDARD"),
    "BM_WEIGHTS_SIMPLEX2D": opt_simplex_proj,
    "BM_WEIGHTS_SIMPLEX3D": opt_simplex_proj,
    "BALL2D_STANDARD": opt_Ball2D_proj("BALL2D_STANDARD"),
    "BALL2D_LARGE": opt_Ball2D_proj("BALL2D_LARGE"),
}


def get_ccw_rectangle_vertices(rect_data_dict_name):
    data_dict = DATA_DICTS[rect_data_dict_name]

    lb_x, ub_x, lb_y, ub_y = get_rectangle_bounds(
        data_dict["width"], data_dict["length"]
    )

    # counterclockwise, starting from bottom-left
    v1, v2, v3, v4 = (lb_x, lb_y), (ub_x, lb_y), (ub_x, ub_y), (lb_x, ub_y)

    return torch.tensor([v1, v2, v3, v4]).float()


def easy_vertices(dataset_name):
    return torch.tensor(DATA_DICTS[dataset_name]["vertices"]).float()


VERTEX_APPROACH_VERTICES = {
    "RECTANGLE_STANDARD": get_ccw_rectangle_vertices("RECTANGLE_STANDARD"),
    "RECTANGLE_WIDER_WITH_MU": get_ccw_rectangle_vertices("RECTANGLE_WIDER_WITH_MU"),
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
    lb, ub = data_dict["lb"], data_dict["ub"]

    def pen(Y):
        return standard_2_norm_for_lb_ub(Y, lb, ub)

    return pen


def rect_pen_func(Y, rect_data_dict_name):
    proj_func = opt_rect_proj(rect_data_dict_name)
    projected = proj_func(Y).clone().detach()

    return torch.norm(Y - projected, 2, dim=1)


def simplex_pen_func(Y):
    proj = opt_simplex_proj(Y)
    res = torch.norm(proj - Y, 2, dim=1)

    return res


def ball2D_pen_func(data_dict):
    R = data_dict["max_radius"]

    def pen(Y):
        res = torch.zeros(len(Y))
        for i, y in enumerate(Y):
            dist = torch.norm(y, 2)
            if dist > R:
                res[i] = (dist - R) ** 2

        return res

    return pen


def zero_pen_func(Y):
    return torch.norm(Y - Y, 2, dim=1)


def RBM_in_shape(RBM_data_dict_name):
    lb = DATA_DICTS[RBM_data_dict_name]["lb"]
    ub = DATA_DICTS[RBM_data_dict_name]["ub"]

    def check(Y):
        res = []
        for pred in Y:
            if lb <= pred and pred <= ub:
                res.append(True)
            else:
                res.append(False)
        return res

    return check


def rect_in_shape(rect_data_dict_name):
    data_dict = DATA_DICTS[rect_data_dict_name]
    lb_x, ub_x, lb_y, ub_y = get_rectangle_bounds(
        data_dict["width"], data_dict["length"]
    )

    def check(Y):
        res = []
        for pred in Y:
            if (
                lb_x <= pred[0]
                and pred[0] <= ub_x
                and lb_y <= pred[1]
                and pred[1] <= ub_y
            ):
                res.append(True)
            else:
                res.append(False)
        return res

    return check


def simplex_in_shape(Y):
    res = []
    for pred in Y:
        if all(y_i >= 0 for y_i in pred):
            sums_to_1 = torch.isclose(torch.sum(pred), torch.tensor(1.0))
            res.append(bool(sums_to_1))
        else:
            res.append(False)
    return res


def ball2D_in_shape(ball2d_data_dict_name):
    R = DATA_DICTS[ball2d_data_dict_name]["max_radius"]

    def check(Y):
        res = [torch.norm(pred, 2) <= R for pred in Y]
        return res

    return check


IN_SHAPE_FUNCS = {
    "RBM_STANDARD": RBM_in_shape("RBM_STANDARD"),
    "RBM_MORE_BOUNCES": RBM_in_shape("RBM_MORE_BOUNCES"),
    "RECTANGLE_STANDARD": rect_in_shape("RECTANGLE_STANDARD"),
    "RECTANGLE_WIDER_WITH_MU": rect_in_shape("RECTANGLE_WIDER_WITH_MU"),
    "BM_WEIGHTS_RECTANGLE_STANDARD": rect_in_shape("RECTANGLE_STANDARD"),
    "BM_WEIGHTS_SIMPLEX2D": simplex_in_shape,
    "BM_WEIGHTS_SIMPLEX3D": simplex_in_shape,
    "BALL2D_STANDARD": ball2D_in_shape("BALL2D_STANDARD"),
    "BALL2D_LARGE": ball2D_in_shape("BALL2D_LARGE"),
}
