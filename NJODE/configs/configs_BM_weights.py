from configs.config_constants import (
    DEFAULT_ENC_NN,
    DEFAULT_ODE_NN,
    DEFAULT_READOUT_NN,
    get_standard_overview_dict,
    get_standard_plot_best_paths,
)
from configs.config_utils import data_path, get_parameter_array


def get_Triangle_BM_weights_config():
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
        "ode_nn": [DEFAULT_ODE_NN],
        "readout_nn": [DEFAULT_READOUT_NN],
        "enc_nn": [DEFAULT_ENC_NN],
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

    overview_dict_Triangle_BM_weights = get_standard_overview_dict(
        param_list_Triangle_BM_weights, Triangle_BM_weights_models_path
    )

    plot_paths_Triangle_BM_weights_dict = get_standard_plot_best_paths(
        [0], Triangle_BM_weights_models_path, [0]
    )

    return (
        param_list_Triangle_BM_weights,
        overview_dict_Triangle_BM_weights,
        plot_paths_Triangle_BM_weights_dict,
    )
