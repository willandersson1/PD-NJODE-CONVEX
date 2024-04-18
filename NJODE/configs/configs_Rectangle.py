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
        "epochs": [3],
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
        "epochs": [3],
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
        "use_rnn": [False],
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
        param_dict_Rectangle_1_vertex_approach,
        overview_dict_Rectangle_vertex_approach,
        plot_paths_Rectangle_vertex_approach_dict,
    )
