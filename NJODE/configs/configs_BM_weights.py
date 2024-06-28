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


def get_BM_WEIGHTS_RECTANGLE_STANDARD_NJODE_config():
    models_path = "{}saved_models_BM_WEIGHTS_RECTANGLE_STANDARD_NJODE/".format(
        data_path
    )
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
        "data_dict": ["BM_WEIGHTS_RECTANGLE_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_RECTANGLE_STANDARD_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_BM_WEIGHTS_RECTANGLE_STANDARD_OPTIMAL_PROJ/".format(
        data_path
    )
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
        "data_dict": ["BM_WEIGHTS_RECTANGLE_STANDARD"],
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

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_RECTANGLE_STANDARD_VERTEX_APPROACH_config():
    models_path = (
        "{}saved_models_BM_WEIGHTS_RECTANGLE_STANDARD_VERTEX_APPROACH/".format(
            data_path
        )
    )
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
        "data_dict": ["BM_WEIGHTS_RECTANGLE_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["vertex_approach"],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_SIMPLEX2D_NJODE_config():
    models_path = "{}saved_models_BM_WEIGHTS_SIMPLEX2D_NJODE/".format(data_path)
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
        "data_dict": ["BM_WEIGHTS_SIMPLEX2D"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_SIMPLEX2D_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_BM_WEIGHTS_SIMPLEX2D_OPTIMAL_PROJ/".format(data_path)
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
        "data_dict": ["BM_WEIGHTS_SIMPLEX2D"],
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

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH_config():
    models_path = "{}saved_models_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH/".format(
        data_path
    )
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
        "data_dict": ["BM_WEIGHTS_SIMPLEX2D"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["vertex_approach"],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_SIMPLEX3D_NJODE_config():
    models_path = "{}saved_models_BM_WEIGHTS_SIMPLEX3D_NJODE/".format(data_path)
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
        "data_dict": ["BM_WEIGHTS_SIMPLEX3D"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "lmbda": [0],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_SIMPLEX3D_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_BM_WEIGHTS_SIMPLEX3D_OPTIMAL_PROJ/".format(data_path)
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
        "data_dict": ["BM_WEIGHTS_SIMPLEX3D"],
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

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )


def get_BM_WEIGHTS_SIMPLEX3D_VERTEX_APPROACH_config():
    models_path = "{}saved_models_BM_WEIGHTS_SIMPLEX3D_VERTEX_APPROACH/".format(
        data_path
    )
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
        "data_dict": ["BM_WEIGHTS_SIMPLEX3D"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["vertex_approach"],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], models_path, [0]
    )

    return (
        param_list,
        overview_dict,
        plot_paths_Triangle_BM_weights_dict,
    )
