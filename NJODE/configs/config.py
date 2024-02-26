"""
author: Florian Krach
"""

import socket

import torch
from configs.config_utils import data_path, get_parameter_array, makedirs

if "ada-" not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
makedirs = makedirs

flagfile = "{}flagfile.tmp".format(data_path)

saved_models_path = "{}saved_models/".format(data_path)


# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
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
        "nb_paths": 2,
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
        "nb_paths": 2,
        "nb_steps": 100,
        "maturity": 1.0,
        "dimension": 2,
        "obs_perc": 0.1,
        "mu_x": 1.5,
        "sigma_x": 1.0,
        "mu_y": 1.0,
        "sigma_y": 1.0,
        "max_z": 5,
        "max_terms": 3,
        "use_approx_paths_technique": True,
        "use_numerical_cond_exp": True,
        "width": 4,
        "length": 6,
        "base_point": (0, 0),
    },
}

# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
ode_nn = ((50, "tanh"), (50, "tanh"))
readout_nn = ((50, "tanh"), (50, "tanh"))
enc_nn = ((50, "tanh"), (50, "tanh"))

# ------------------------------------------------------------------------------
# --- Fractional Brownian Motion
FBM_models_path = "{}saved_models_FBM/".format(data_path)
param_list_FBM1 = []
param_dict_FBM1_1 = {
    "epochs": [3],
    "batch_size": [200],
    "save_every": [1],
    "learning_rate": [0.001],
    "test_size": [0.2],
    "seed": [398],
    "hidden_size": [10, 50],
    "bias": [True],
    "dropout_rate": [0.1],
    "ode_nn": [ode_nn],
    "readout_nn": [readout_nn],
    "enc_nn": [enc_nn],
    "use_rnn": [False],
    "func_appl_X": [[]],
    "solver": ["euler"],
    "weight": [0.5],
    "weight_decay": [1.0],
    "data_dict": ["FBM_1_dict"],
    "plot": [True],
    "evaluate": [True],
    "paths_to_plot": [
        (
            0,
            1,
            2,
            3,
            4,
        )
    ],
    "saved_models_path": [FBM_models_path],
}
param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_1)

overview_dict_FBM1 = dict(
    ids_from=1,
    ids_to=len(param_list_FBM1),
    path=FBM_models_path,
    params_extract_desc=(
        "dataset",
        "network_size",
        "nb_layers",
        "activation_function_1",
        "use_rnn",
        "readout_nn",
        "dropout_rate",
        "hidden_size",
        "batch_size",
        "which_loss",
        "input_sig",
        "level",
    ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        (
            "min",
            "evaluation_mean_diff",
            "evaluation_mean_diff",
            "evaluation_mean_diff_min",
        ),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_FBM_dict = {
    "model_ids": [33, 34, 35, 41, 43, 50],
    "saved_models_path": FBM_models_path,
    "which": "best",
    "paths_to_plot": [0, 1, 2, 3, 4, 5],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}

# ------------------------------------------------------------------------------
# --- Reflected Brownian Motion
RBM_models_path = "{}saved_models_RBM/".format(data_path)
param_list_RBM = []
param_dict_RBM_1 = {
    "epochs": [3],
    "batch_size": [200],
    "save_every": [1],
    "learning_rate": [0.001],
    "test_size": [0.2],
    "seed": [398],
    "hidden_size": [10],
    "bias": [True],
    "dropout_rate": [0.1],
    "ode_nn": [ode_nn],
    "readout_nn": [readout_nn],
    "enc_nn": [enc_nn],
    "use_rnn": [False],
    "func_appl_X": [[]],
    "solver": ["euler"],
    "weight": [0.5],
    "weight_decay": [1.0],
    "data_dict": ["RBM_1_dict"],
    "plot": [True],
    "evaluate": [True],
    "paths_to_plot": [(0,)],
    "saved_models_path": [RBM_models_path],
    "other_model": ["cvx_optimal_proj"],
}
param_list_RBM += get_parameter_array(param_dict=param_dict_RBM_1)

overview_dict_RBM = dict(
    ids_from=1,
    ids_to=len(param_list_RBM),
    path=RBM_models_path,
    params_extract_desc=(
        "dataset",
        "network_size",
        "nb_layers",
        "activation_function_1",
        "use_rnn",
        "readout_nn",
        "dropout_rate",
        "hidden_size",
        "batch_size",
        "which_loss",
        "input_sig",
        "level",
    ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        (
            "min",
            "evaluation_mean_diff",
            "evaluation_mean_diff",
            "evaluation_mean_diff_min",
        ),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_RBM_dict = {
    "model_ids": [0, 1],
    "saved_models_path": RBM_models_path,
    "which": "best",
    "paths_to_plot": [0, 1],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}


# ------------------------------------------------------------------------------
# --- Rectangle
Rectangle_models_path = "{}saved_models_Rectangle/".format(data_path)
param_list_Rectangle = []
param_dict_Rectangle_1 = {
    "epochs": [3],
    "batch_size": [200],
    "save_every": [1],
    "learning_rate": [0.001],
    "test_size": [0.2],
    "seed": [398],
    "hidden_size": [10],
    "bias": [True],
    "dropout_rate": [0.1],
    "ode_nn": [ode_nn],
    "readout_nn": [readout_nn],
    "enc_nn": [enc_nn],
    "use_rnn": [False],
    "func_appl_X": [[]],
    "solver": ["euler"],
    "weight": [0.5],
    "weight_decay": [1.0],
    "data_dict": ["Rectangle_1_dict"],
    "plot": [True],
    "evaluate": [True],
    "paths_to_plot": [(0,)],
    "saved_models_path": [Rectangle_models_path],
    "other_model": ["cvx_optimal_proj"],
}
param_list_Rectangle += get_parameter_array(param_dict=param_dict_Rectangle_1)

overview_dict_Rectangle = dict(
    ids_from=1,
    ids_to=len(param_list_Rectangle),
    path=Rectangle_models_path,
    params_extract_desc=(
        "dataset",
        "network_size",
        "nb_layers",
        "activation_function_1",
        "use_rnn",
        "readout_nn",
        "dropout_rate",
        "hidden_size",
        "batch_size",
        "which_loss",
        "input_sig",
        "level",
    ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        (
            "min",
            "evaluation_mean_diff",
            "evaluation_mean_diff",
            "evaluation_mean_diff_min",
        ),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_Rectangle_dict = {
    "model_ids": [0],
    "saved_models_path": Rectangle_models_path,
    "which": "best",
    "paths_to_plot": [0],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}


# TODO shouldn't these be / aren't they defined in the respective dataset classes?
def opt_RBM_proj(x, RBM_param_dict):
    return torch.clamp(
        x,
        DATA_DICTS[RBM_param_dict["data_dict"][0]]["lb"],
        DATA_DICTS[RBM_param_dict["data_dict"][0]]["ub"],
    )


def opt_rect_proj(x, rect_param_dict):
    data_dict = DATA_DICTS[rect_param_dict["data_dict"][0]]
    lb_x, lb_y = data_dict["base_point"][0], data_dict["base_point"][1]
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]
    lower = torch.tensor([lb_x, lb_y])
    upper = torch.tensor([ub_x, ub_y])
    return torch.clamp(x, lower, upper)


OPTIMAL_PROJECTION_FUNCS = {
    "RBM_1_dict": lambda x: opt_RBM_proj(x, param_dict_RBM_1),
    "Rectangle_1_dict": lambda x: opt_rect_proj(x, param_dict_Rectangle_1),
}


def standard_2_norm_for_lb_ub(Y, lb, ub):
    if lb <= Y <= ub:
        return torch.norm(Y - Y, 2)  # TODO I guess I need? to keep grad?
    if Y < lb:
        return torch.norm(Y - float(lb), 2)
    if Y > ub:
        return torch.norm(Y - float(ub), 2)


def rect_pen_func(Y, data_dict):
    lb_x, lb_y = data_dict["base_point"][0], data_dict["base_point"][1]
    ub_x, ub_y = lb_x + data_dict["width"], lb_y + data_dict["length"]
    compare_to_list = [0, 0]

    # TODO got to be a better way to do this
    if Y[0][0] < lb_x:
        compare_to_list[0] = lb_x
    elif Y[0][0] > ub_x:
        compare_to_list[0] = lb_x
    else:
        compare_to_list[0] = float(Y[0][0])  # it's good so ignore this coordinate

    if Y[0][1] < lb_y:
        compare_to_list[0] = lb_y
    elif Y[0][1] > ub_y:
        compare_to_list[0] = lb_y
    else:
        compare_to_list[0] = float(Y[0][1])

    compare_to = torch.tensor([compare_to_list], dtype=float)
    return torch.norm(Y - compare_to, 2)


# TODO actually these should be keyed by the model params, not the dataset name
CONVEX_PEN_FUNCS = {
    "RBM_1_dict": lambda Y: standard_2_norm_for_lb_ub(
        Y,
        DATA_DICTS[param_dict_RBM_1["data_dict"][0]]["lb"],
        DATA_DICTS[param_dict_RBM_1["data_dict"][0]]["ub"],
    ),
    "Rectangle_1_dict": lambda Y: rect_pen_func(
        Y, DATA_DICTS[param_dict_Rectangle_1["data_dict"][0]]
    ),
}
