"""
author: Florian Krach
"""

from configs.config_NJODE1 import *

import numpy as np
import socket

from configs.config_utils import get_parameter_array, makedirs, \
    SendBotMessage, data_path, training_data_path

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
CHAT_ID = "-587067467"
ERROR_CHAT_ID = "-725470544"

SendBotMessage = SendBotMessage
makedirs = makedirs

flagfile = "{}flagfile.tmp".format(data_path)

saved_models_path = '{}saved_models/'.format(data_path)


# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
FBM_1_dict = {
    'model_name': "FBM",
    'nb_paths': 100000, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'return_vol': False, 'hurst': 0.05,
    'FBMmethod': "daviesharte"
}

# ------------------------------------------------------------------------------
BM_2D_dict = {
    'model_name': "BM2DCorr",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_sq': 0.9, 'masked': 0.,
    'dimension': 2,
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
    'epochs': [200],
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
# --- 2d Brownian Motion with correlation and correct cond. exp.
BM2D_models_path = "{}saved_models_BM2D/".format(data_path)
param_list_BM2D_1 = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_BM2D_1_2 = {
            'epochs': [200],
            'batch_size': [200],
            'save_every': [1],
            'learning_rate': [0.001],
            'test_size': [0.2],
            'seed': [398],
            'hidden_size': [size,],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_nn],
            'readout_nn': [_nn, None],
            'enc_nn': [_nn],
            'use_rnn': [True],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True],
            'level': [2, ],
            'dataset': ["BM2DCorr", ],
            'dataset_id': [None],
            'which_loss': ['easy',],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [True],
            'masked': [True],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [True],
            'saved_models_path': [BM2D_models_path],
        }
        param_list_BM2D_1 += get_parameter_array(param_dict=param_dict_BM2D_1_2)

overview_dict_BM2D_1 = dict(
    ids_from=1, ids_to=len(param_list_BM2D_1),
    path=BM2D_models_path,
    params_extract_desc=('dataset', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_y_for_ode'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_BM2D_dict = {
    'model_ids': [1,2,3,4], 'saved_models_path': BM2D_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

