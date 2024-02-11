"""
author: Florian Krach
"""

import socket

from configs.config_utils import get_parameter_array, makedirs, data_path

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
makedirs = makedirs

flagfile = "{}flagfile.tmp".format(data_path)

saved_models_path = '{}saved_models/'.format(data_path)


# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
FBM_1_dict = {
    'model_name': "FBM",
    'nb_paths': 100, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'return_vol': False, 'hurst': 0.05,
    'FBMmethod': "daviesharte"
}

RBM_1_dict = {
    'model_name': "RBM",
    'nb_paths': 2, 'nb_steps': 100,
    'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'mu': 1.5, 'sigma': 1.0,
    'lb': 2, 'ub': 4,
    'max_z': 5,
    'max_terms': 3,
    'use_approx_paths_technique': True,
    'use_numerical_cond_exp': True,
}

# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
ode_nn = ((50, 'tanh'), (50, 'tanh'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))
enc_nn = ((50, 'tanh'), (50, 'tanh'))

# ------------------------------------------------------------------------------
# --- Fractional Brownian Motion
FBM_models_path = "{}saved_models_FBM/".format(data_path)
param_list_FBM1 = []
param_dict_FBM1_1 = {
    'epochs': [3],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10, 50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [ode_nn],
    'readout_nn': [readout_nn],
    'enc_nn': [enc_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ['FBM_1_dict'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [FBM_models_path],
}
param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_1)

overview_dict_FBM1 = dict(
    ids_from=1, ids_to=len(param_list_FBM1),
    path=FBM_models_path,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_FBM_dict = {
    'model_ids': [33, 34, 35, 41, 43, 50],
    'saved_models_path': FBM_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

# ------------------------------------------------------------------------------
# --- Reflected Brownian Motion
RBM_models_path = "{}saved_models_RBM/".format(data_path)
param_list_RBM = []
param_dict_RBM_1 = {
    'epochs': [3],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [ode_nn],
    'readout_nn': [readout_nn],
    'enc_nn': [enc_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ['RBM_1_dict'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,)],
    'saved_models_path': [RBM_models_path],
    'other_model': ['cvx_optimal_proj']
}
param_list_RBM += get_parameter_array(param_dict=param_dict_RBM_1)

overview_dict_RBM = dict(
    ids_from=1, ids_to=len(param_list_RBM),
    path=RBM_models_path,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_RBM_dict = {
    'model_ids': [0,1],
    'saved_models_path': RBM_models_path,
    'which': 'best', 'paths_to_plot': [0,1],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}
