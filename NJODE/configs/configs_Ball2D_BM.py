"""
author: William Andersson
"""

from configs.config_constants import (
    DEFAULT_ENC_NN,
    DEFAULT_ODE_NN,
    DEFAULT_READOUT_NN,
    get_standard_overview_dict,
    get_standard_plot_best_paths,
)
from configs.config_utils import data_path, get_parameter_array


def get_BALL2D_STANDARD_NJODE_config():
    models_path = "{}saved_models_BALL2D_STANDARD_NJODE/".format(data_path)

    param_list = []
    param_dict = {
        "epochs": [200],
        "batch_size": [200],
        "save_every": [20],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10],
        "bias": [True],
        "dropout_rate": [0.1],
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
        "use_rnn": [True],
        "residual_enc_dec": [False],
        "func_appl_X": [[]],
        "solver": ["euler"],
        "weight": [0.5],
        "weight_decay": [1.0],
        "data_dict": ["BALL2D_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "lmbda": [0],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_dict = get_standard_plot_best_paths([1], models_path, [0])

    return param_list, overview_dict, plot_paths_dict


def get_BALL2D_STANDARD_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_BALL2D_STANDARD_OPTIMAL_PROJ/".format(data_path)

    param_list = []
    param_dict = {
        "epochs": [200],
        "batch_size": [200],
        "save_every": [20],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10],
        "bias": [True],
        "dropout_rate": [0.1],
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
        "use_rnn": [True],
        "residual_enc_dec": [False],
        "func_appl_X": [[]],
        "solver": ["euler"],
        "weight": [0.5],
        "weight_decay": [1.0],
        "data_dict": ["BALL2D_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["optimal_projection"],
        "project_only_at_inference": [False],
        "lmbda": [0],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_dict = get_standard_plot_best_paths([1], models_path, [0])

    return param_list, overview_dict, plot_paths_dict


def get_BALL2D_LARGE_NJODE_config():
    models_path = "{}saved_models_BALL2D_LARGE_NJODE/".format(data_path)

    param_list = []
    param_dict = {
        "epochs": [200],
        "batch_size": [200],
        "save_every": [20],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10],
        "bias": [True],
        "dropout_rate": [0.1],
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
        "use_rnn": [True],
        "residual_enc_dec": [False],
        "func_appl_X": [[]],
        "solver": ["euler"],
        "weight": [0.5],
        "weight_decay": [1.0],
        "data_dict": ["BALL2D_LARGE"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_dict = get_standard_plot_best_paths([1], models_path, [0])

    return param_list, overview_dict, plot_paths_dict


def get_BALL2D_LARGE_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_BALL2D_LARGE_OPTIMAL_PROJ/".format(data_path)

    param_list = []
    param_dict = {
        "epochs": [200],
        "batch_size": [200],
        "save_every": [20],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10],
        "bias": [True],
        "dropout_rate": [0.1],
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
        "use_rnn": [True],
        "residual_enc_dec": [False],
        "func_appl_X": [[]],
        "solver": ["euler"],
        "weight": [0.5],
        "weight_decay": [1.0],
        "data_dict": ["BALL2D_LARGE"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["optimal_projection"],
        "project_only_at_inference": [False],
        "lmbda": [0],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_dict = get_standard_plot_best_paths([1], models_path, [0])

    return param_list, overview_dict, plot_paths_dict


def get_BALL2D_TINY_NJODE_config():
    models_path = "{}saved_models_BALL2D_TINY_NJODE/".format(data_path)

    param_list = []
    param_dict = {
        "epochs": [200],
        "batch_size": [200],
        "save_every": [20],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10],
        "bias": [True],
        "dropout_rate": [0.1],
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
        "use_rnn": [True],
        "residual_enc_dec": [False],
        "func_appl_X": [[]],
        "solver": ["euler"],
        "weight": [0.5],
        "weight_decay": [1.0],
        "data_dict": ["BALL2D_TINY"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "lmbda": [1],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_dict = get_standard_plot_best_paths([1], models_path, [0])

    return param_list, overview_dict, plot_paths_dict


def get_BALL2D_TINY_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_BALL2D_TINY_OPTIMAL_PROJ/".format(data_path)

    param_list = []
    param_dict = {
        "epochs": [200],
        "batch_size": [200],
        "save_every": [20],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10],
        "bias": [True],
        "dropout_rate": [0.1],
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
        "use_rnn": [True],
        "residual_enc_dec": [False],
        "func_appl_X": [[]],
        "solver": ["euler"],
        "weight": [0.5],
        "weight_decay": [1.0],
        "data_dict": ["BALL2D_TINY"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["optimal_projection"],
        "project_only_at_inference": [False],
        "lmbda": [1],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_dict = get_standard_plot_best_paths([1], models_path, [0])

    return param_list, overview_dict, plot_paths_dict
