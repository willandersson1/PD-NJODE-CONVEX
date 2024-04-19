from configs.config_constants import (
    DEFAULT_ENC_NN,
    DEFAULT_ODE_NN,
    DEFAULT_READOUT_NN,
    get_standard_overview_dict,
    get_standard_plot_best_paths,
)
from configs.config_utils import data_path, get_parameter_array


def get_Ball2D_BM_config():
    Ball2D_BM_models_path = "{}saved_models_Ball2D_BM/".format(data_path)

    param_list_Ball2D_BM = []
    param_dict_Ball2D_BM_1 = {
        "epochs": [1],
        "batch_size": [200],
        "save_every": [1],
        "learning_rate": [0.001],
        "test_size": [0.2],
        "seed": [398],
        "hidden_size": [10, 50],
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
        "data_dict": ["Ball2D_BM_1_dict"],
        "plot": [True],
        "evaluate": [True],
        "paths_to_plot": [(0, 1)],
        "saved_models_path": [Ball2D_BM_models_path],
    }
    param_list_Ball2D_BM += get_parameter_array(param_dict=param_dict_Ball2D_BM_1)

    overview_dict_Ball2D_BM = get_standard_overview_dict(
        param_list_Ball2D_BM, Ball2D_BM_models_path
    )

    plot_paths_Ball2D_BM_dict = get_standard_plot_best_paths(
        [1], Ball2D_BM_models_path, [0]
    )

    return param_list_Ball2D_BM, overview_dict_Ball2D_BM, plot_paths_Ball2D_BM_dict
