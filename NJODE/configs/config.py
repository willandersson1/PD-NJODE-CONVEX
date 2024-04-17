"""
author: Florian Krach
"""

import socket

from configs.config_utils import data_path, get_parameter_array, makedirs
from configs.dataset_configs import (
    DATA_DICTS,
    rect_pen_func,
    standard_2_norm_for_lb_ub,
    zero_pen_func,
)

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
    "paths_to_plot": [(0, 1)],
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
    "model_ids": [1],
    "saved_models_path": FBM_models_path,
    "which": "best",
    "paths_to_plot": [0],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}

# ------------------------------------------------------------------------------
# --- Reflected Brownian Motion
RBM_models_path = "{}saved_models_RBM/".format(data_path)
param_list_RBM = []
param_dict_RBM_1 = {
    "epochs": [10],
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
    "paths_to_plot": [(0, 1)],
    "saved_models_path": [RBM_models_path],
    "other_model": ["optimal_projection"],
    "lmbda": [1],
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
    "paths_to_plot": [(0, 1)],
    "saved_models_path": [Rectangle_models_path],
    "other_model": ["optimal_projection"],
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

# ------------------------------------------------------------------------------
# --- Rectangle vertex approach
Rectangle_vertex_approach_models_path = (
    "{}saved_models_Rectangle_vertex_approach/".format(data_path)
)
param_list_Rectangle_vertex_approach = []
param_dict_Rectangle_1_vertex_approach = param_dict_Rectangle_1.copy()
param_dict_Rectangle_1_vertex_approach["other_model"] = ["vertex_approach"]
param_dict_Rectangle_1_vertex_approach["saved_models_path"] = [
    Rectangle_vertex_approach_models_path
]
param_list_Rectangle_vertex_approach += get_parameter_array(
    param_dict=param_dict_Rectangle_1_vertex_approach
)
overview_dict_Rectangle_vertex_approach = dict(
    ids_from=1,
    ids_to=len(param_list_Rectangle_vertex_approach),
    path=Rectangle_vertex_approach_models_path,
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
plot_paths_Rectangle_vertex_approach_dict = {
    "model_ids": [0],
    "saved_models_path": Rectangle_vertex_approach_models_path,
    "which": "best",
    "paths_to_plot": [0],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}


# ------------------------------------------------------------------------------
# --- Triangle BM weights vertex approach
# TODO the naming here is bad, it's actually vertex approach but I don't mention it
Triangle_BM_weights_models_path = "{}saved_models_Triangle_BM_weights/".format(
    data_path
)
param_list_Triangle_BM_weights = []
param_dict_Triangle_BM_weights_1 = {
    "epochs": [150],
    "batch_size": [200],
    "save_every": [10],
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
    "data_dict": ["Triangle_BM_weights_1_dict"],
    "plot": [True],
    "evaluate": [True],
    "paths_to_plot": [(0,)],
    "saved_models_path": [Triangle_BM_weights_models_path],
    "other_model": ["vertex_approach"],
}
param_list_Triangle_BM_weights += get_parameter_array(
    param_dict=param_dict_Triangle_BM_weights_1
)

overview_dict_Triangle_BM_weights = dict(
    ids_from=1,
    ids_to=len(param_list_Triangle_BM_weights),
    path=Triangle_BM_weights_models_path,
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

plot_paths_Triangle_BM_weights_dict = {
    "model_ids": [0],
    "saved_models_path": Triangle_BM_weights_models_path,
    "which": "best",
    "paths_to_plot": [0],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}


###################
# Ball2D with NJODE
Ball2D_BM_models_path = "{}saved_models_Ball2D_BM/".format(data_path)
param_list_Ball2D_BM = []
param_dict_Ball2D_BM_1 = {
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
    "data_dict": ["Ball2D_BM_1_dict"],
    "plot": [True],
    "evaluate": [True],
    "paths_to_plot": [(0, 1)],
    "saved_models_path": [Ball2D_BM_models_path],
}
param_list_Ball2D_BM += get_parameter_array(param_dict=param_dict_Ball2D_BM_1)

overview_dict_Ball2D_BM = dict(
    ids_from=1,
    ids_to=len(param_list_Ball2D_BM),
    path=Ball2D_BM_models_path,
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

plot_paths_Ball2D_BM_dict = {
    "model_ids": [1],
    "saved_models_path": Ball2D_BM_models_path,
    "which": "best",
    "paths_to_plot": [0],
    "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
}


CONVEX_PEN_FUNCS = {
    "RBM_1_dict": lambda Y: standard_2_norm_for_lb_ub(
        Y,
        DATA_DICTS["RBM_1_dict"]["lb"],
        DATA_DICTS["RBM_1_dict"]["ub"],
    ),
    "Rectangle_1_dict": lambda Y: rect_pen_func(Y, DATA_DICTS["RBM_1_dict"]),
    "Triangle_BM_weights_1_dict": zero_pen_func,
}
