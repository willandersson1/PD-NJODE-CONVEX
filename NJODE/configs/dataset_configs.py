import torch

DATA_DICTS = {
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
        "nb_paths": 10,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 1,
        "obs_perc": 0.1,
        "mu": 1.5,
        "sigma": 1.0,
        "lb": 2,
        "ub": 4,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
    },
    "Rectangle_1_dict": {
        "model_name": "Rectangle",
        "nb_paths": 3,
        "nb_steps": 1000,
        "maturity": 5.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 2.0,
        "sigma_x": 1.0,
        "mu_y": 5.0,
        "sigma_y": 1.0,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 4,
        "length": 10,
        "base_point": (1, 1),
    },
    "Triangle_BM_weights_1_dict": {
        "model_name": "BMWeights",
        "should_compute_approx_cond_exp_paths": True,
        "vertices": [[0, 0], [1, 0], [0, 1]],
        "mu": [0, 0.5, 1],
        "sigma": [2, 1, 1],
        "nb_paths": 3,
        "nb_steps": 1000,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
    },
    "Ball2D_BM_1_dict": {
        "model_name": "Ball2D_BM",
        "max_radius": 10,
        "radius_mu": 0,
        "radius_sigma": 1,
        "angle_mu": [0],
        "angle_sigma": [1],
        "nb_paths": 3,
        "nb_steps": 1000,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.05,
    },
}


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
    lb_x, lb_y = data_dict["base_point"]
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]
    lower = torch.tensor([lb_x, lb_y])
    upper = torch.tensor([ub_x, ub_y])

    def optimal_proj(x):
        return torch.clamp(x, lower, upper)

    return optimal_proj


OPTIMAL_PROJECTION_FUNCS = {
    "RBM_1_dict": opt_RBM_proj("RBM_1_dict"),
    "Rectangle_1_dict": opt_rect_proj("Rectangle_1_dict"),
}


def get_ccw_rectangle_vertices(rect_data_dict_name):
    data_dict = DATA_DICTS[rect_data_dict_name]
    lb_x, lb_y = data_dict["base_point"][0], data_dict["base_point"][1]
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]

    # counterclockwise, starting from bottom-left
    v1, v2, v3, v4 = (lb_x, lb_y), (ub_x, lb_y), (ub_x, ub_y), (lb_x, ub_y)

    return torch.tensor([v1, v2, v3, v4]).float()


def easy_vertices(dataset_name):
    return torch.tensor(DATA_DICTS[dataset_name]["vertices"]).float()


VERTEX_APPROACH_VERTICES = {
    "Rectangle_1_dict": get_ccw_rectangle_vertices("Rectangle_1_dict"),
    "Triangle_BM_weights_1_dict": easy_vertices("Triangle_BM_weights_1_dict"),
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


def rect_pen_func(Y, data_dict):
    lb_x, lb_y = data_dict["base_point"][0], data_dict["base_point"][1]
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
