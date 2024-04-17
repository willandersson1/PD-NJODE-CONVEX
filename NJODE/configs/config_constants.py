from configs.config_utils import data_path

flagfile = "{}flagfile.tmp".format(data_path)

saved_models_path = "{}saved_models/".format(data_path)

DEFAULT_ODE_NN = ((50, "tanh"), (50, "tanh"))
DEFAULT_READOUT_NN = ((50, "tanh"), (50, "tanh"))
DEFAULT_ENC_NN = ((50, "tanh"), (50, "tanh"))

STANDARD_PARAMS_EXTRACT_DESC = (
    (
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
)

STANDARD_VAL_TEST_PARAMS_EXTRACT = (
    ("max", "epoch", "epoch", "epochs_trained"),
    (
        "min",
        "evaluation_mean_diff",
        "evaluation_mean_diff",
        "evaluation_mean_diff_min",
    ),
    ("min", "eval_loss", "eval_loss", "eval_loss_min"),
)


def get_standard_plot_best_paths(model_ids, models_path, paths):
    return {
        "model_ids": model_ids,
        "saved_models_path": models_path,
        "which": "best",
        "paths_to_plot": paths,
        "save_extras": {"bbox_inches": "tight", "pad_inches": 0.01},
    }


def get_standard_overview_dict(param_list, models_path):
    return {
        "ids_from": 1,
        "ids_to": len(param_list),
        "path": models_path,
        "params_extract_desc": STANDARD_PARAMS_EXTRACT_DESC,
        "val_test_params_extract": STANDARD_VAL_TEST_PARAMS_EXTRACT,
        "sortby": ["evaluation_mean_diff_min"],
    }
