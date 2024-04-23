from configs.config_constants import (
    DEFAULT_ENC_NN,
    DEFAULT_ODE_NN,
    DEFAULT_READOUT_NN,
    get_standard_overview_dict,
    get_standard_plot_best_paths,
)
from configs.config_utils import data_path, get_parameter_array


def get_Rectangle_config():
    Rectangle_models_path = "{}saved_models_Rectangle/".format(data_path)
    param_list_Rectangle = []
    param_dict_Rectangle_1 = {
        "epochs": [50],
        "batch_size": [200],
        "save_every": [5],
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
        "data_dict": ["Rectangle_1_dict"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [Rectangle_models_path],
        "other_model": ["optimal_projection"],
        "lmbda": [1],
    }
    param_list_Rectangle += get_parameter_array(param_dict=param_dict_Rectangle_1)

    overview_dict_Rectangle = get_standard_overview_dict(
        param_list_Rectangle, Rectangle_models_path
    )

    plot_paths_Rectangle_dict = get_standard_plot_best_paths(
        [0], Rectangle_models_path, [0]
    )

    return param_list_Rectangle, overview_dict_Rectangle, plot_paths_Rectangle_dict


def get_Rectangle_vertex_approach_config():
    Rectangle_vertex_approach_models_path = (
        "{}saved_models_Rectangle_vertex_approach/".format(data_path)
    )
    param_list_Rectangle_vertex_approach = []
    param_dict_Rectangle_1_vertex_approach = {
        "epochs": [1],
        "batch_size": [200],
        "save_every": [1],
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
        "data_dict": ["Rectangle_1_dict"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [Rectangle_vertex_approach_models_path],
        "other_model": ["vertex_approach"],
    }
    param_list_Rectangle_vertex_approach += get_parameter_array(
        param_dict=param_dict_Rectangle_1_vertex_approach
    )
    overview_dict_Rectangle_vertex_approach = get_standard_overview_dict(
        param_list_Rectangle_vertex_approach, Rectangle_vertex_approach_models_path
    )

    plot_paths_Rectangle_vertex_approach_dict = get_standard_plot_best_paths(
        [0], Rectangle_vertex_approach_models_path, [0]
    )

    return (
        param_list_Rectangle_vertex_approach,
        overview_dict_Rectangle_vertex_approach,
        plot_paths_Rectangle_vertex_approach_dict,
    )


def get_RECTANGLE_STANDARD_NJODE_config():
    models_path = "{}saved_models_RECTANGLE_STANDARD_NJODE/".format(data_path)
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
        "data_dict": ["RECTANGLE_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Rectangle_dict = get_standard_plot_best_paths([0], models_path, [0])

    return param_list, overview_dict, plot_paths_Rectangle_dict


def get_RECTANGLE_STANDARD_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_RECTANGLE_STANDARD_OPTIMAL_PROJ/".format(data_path)
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
        "data_dict": ["RECTANGLE_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["optimal_projection"],
        "lmbda": [0],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Rectangle_dict = get_standard_plot_best_paths([0], models_path, [0])

    return param_list, overview_dict, plot_paths_Rectangle_dict


def get_RECTANGLE_STANDARD_VERTEX_APPROACH_config():
    models_path = "{}saved_models_RECTANGLE_STANDARD_VERTEX_APPROACH/".format(data_path)
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
        "data_dict": ["RECTANGLE_STANDARD"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["vertex_approach"],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Rectangle_dict = get_standard_plot_best_paths([0], models_path, [0])

    return param_list, overview_dict, plot_paths_Rectangle_dict


def get_RECTANGLE_WIDER_WITH_MU_NJODE_config():
    models_path = "{}saved_models_RECTANGLE_WIDER_WITH_MU_NJODE/".format(data_path)
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
        "data_dict": ["RECTANGLE_WIDER_WITH_MU"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Rectangle_dict = get_standard_plot_best_paths([0], models_path, [0])

    return param_list, overview_dict, plot_paths_Rectangle_dict


def get_RECTANGLE_WIDER_WITH_MU_OPTIMAL_PROJ_config():
    models_path = "{}saved_models_RECTANGLE_WIDER_WITH_MU_OPTIMAL_PROJ/".format(
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
        "data_dict": ["RECTANGLE_WIDER_WITH_MU"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["optimal_projection"],
        "lmbda": [0],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Rectangle_dict = get_standard_plot_best_paths([0], models_path, [0])

    return param_list, overview_dict, plot_paths_Rectangle_dict


def get_RECTANGLE_WIDER_WITH_MU_VERTEX_APPROACH_config():
    models_path = "{}saved_models_RECTANGLE_WIDER_WITH_MU_VERTEX_APPROACH/".format(
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
        "data_dict": ["RECTANGLE_WIDER_WITH_MU"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [models_path],
        "other_model": ["vertex_approach"],
    }
    param_list += get_parameter_array(param_dict=param_dict)

    overview_dict = get_standard_overview_dict(param_list, models_path)

    plot_paths_Rectangle_dict = get_standard_plot_best_paths([0], models_path, [0])

    return param_list, overview_dict, plot_paths_Rectangle_dict
