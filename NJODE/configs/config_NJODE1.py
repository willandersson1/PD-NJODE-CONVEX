"""
author: Florian Krach

This file contains all configs to run the experiments from the first paper
"""
import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
# -------------------- NJODE 1 - Dataset Dicts ---------------------------------
# ==============================================================================

# ------------------------------------------------------------------------------
# default hp dict for generating Black-Scholes, Ornstein-Uhlenbeck and Heston
hyperparam_default = {
    'drift': 2., 'volatility': 0.3, 'mean': 4, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod':"daviesharte"
}

# ==============================================================================
# -------------------- NJODE 1 - Training Dicts --------------------------------
# ==============================================================================

ode_nn = ((50, 'tanh'), (50, 'tanh'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))
enc_nn = ((50, 'tanh'), (50, 'tanh'))

# ------------------------------------------------------------------------------
# --- Black-Scholes (geom. Brownian Motion), Heston and Ornstein-Uhlenbeck
param_dict1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [5],
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
    'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
    'dataset_id': [None],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)]
}
params_list1 = get_parameter_array(param_dict=param_dict1)


# ------------------------------------------------------------------------------
# convergence analysis
path_heston = '{}conv-study-Heston-saved_models/'.format(data_path)
training_size = [int(100 * 2 ** x) for x in np.linspace(1, 7, 7)]
network_size = [int(5 * 2 ** x) for x in np.linspace(1, 6, 6)]
ode_nn = [((size, 'tanh'), (size, 'tanh')) for size in network_size]
params_list_convstud_Heston = []
for _ode_nn in ode_nn:
    param_dict_convstud_Heston = {
        'epochs': [100],
        'batch_size': [20],
        'save_every': [10],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'training_size': training_size,
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["Heston"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0,)],
        'saved_models_path': [path_heston],
        'evaluate': [True]
    }
    params_list_convstud_Heston += get_parameter_array(
        param_dict=param_dict_convstud_Heston)
params_list_convstud_Heston *= 5

plot_conv_stud_heston_dict1 = dict(
    path=path_heston, x_axis="training_size", x_log=True, y_log=True,
    save_path=path_heston)
plot_conv_stud_heston_dict2 = dict(
    path=path_heston, x_axis="network_size", x_log=True, y_log=True,
    save_path=path_heston)


path_BS = '{}conv-study-BS-saved_models/'.format(data_path)
params_list_convstud_BS = []
for _ode_nn in ode_nn:
    param_dict_convstud_BS = {
        'epochs': [2], 
        'other_model': ['cvx'], # TODO remove after debugging
        'batch_size': [20],
        'save_every': [10],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'training_size': training_size,
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["BlackScholes"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0,)],
        'saved_models_path': [path_BS],
        'evaluate': [True]
    }
    params_list_convstud_BS += get_parameter_array(
        param_dict=param_dict_convstud_BS)
params_list_convstud_BS *= 5

plot_conv_stud_BS_dict1 = dict(
    path=path_BS, x_axis="training_size", x_log=True, y_log=True,
    save_path=path_BS)
plot_conv_stud_BS_dict2 = dict(
    path=path_BS, x_axis="network_size", x_log=True, y_log=True,
    save_path=path_BS)
