"""
author: Florian Krach & Calypso Herrera

code for all additional things, like plotting, getting overviews etc.
"""

import json
import os

import numpy as np
import pandas as pd
from configs import config_constants
from train_switcher import train_switcher


def get_training_overview(
    path=config_constants.saved_models_path,
    ids_from=None,
    ids_to=None,
    params_extract_desc=("network_size", "training_size", "dataset", "hidden_size"),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff", "evaluation_mean_diff", "eval_metric_min"),
        ("last", "evaluation_mean_diff", "evaluation_mean_diff", "eval_metric_last"),
        (
            "average",
            "evaluation_mean_diff",
            "evaluation_mean_diff",
            "eval_metric_average",
        ),
    ),
    early_stop_after_epoch=0,
    sortby=None,
    save_file=None,
):
    """
    function to get the important metrics and hyper-params for each model in the
    models_overview.csv file
    :param path: str, where the saved models are
    :param ids_from: None or int, which model ids to consider start point
    :param ids_to: None or int, which model ids to consider end point
    :param params_extract_desc: list of str, names of params to extract from the
            model description dict, special:
                - network_size: gets size of first layer of enc network
                - nb_layers: gets number of layers of enc network
                - activation_function_x: gets the activation function of layer x
                    of enc network
    :param val_test_params_extract: None or list of list with 4 string elements:
                0. "min" or "max" or "last" or "average"
                1. col_name where to look for min/max (validation), or where to
                    get last value or average
                2. if 0. is min/max: col_name where to find value in epoch where
                                     1. is min/max (test)
                   if 0. is last/average: not used
                3. name for this output column in overview file
    :param early_stop_after_epoch: int, epoch after which early stopping is
            allowed (i.e. all epochs until there are not considered)
    :param save_file:
    :return:
    """
    filename = "{}model_overview.csv".format(path)
    df = pd.read_csv(filename, index_col=0)
    if ids_from:
        df = df.loc[df["id"] >= ids_from]
    if ids_to:
        df = df.loc[df["id"] <= ids_to]

    # extract wanted information
    for param in params_extract_desc:
        df[param] = None

    if val_test_params_extract:
        for l in val_test_params_extract:
            df[l[3]] = None

    for i in df.index:
        desc = df.loc[i, "description"]
        param_dict = json.loads(desc)

        for param in params_extract_desc:
            try:
                if param == "network_size":
                    v = param_dict["enc_nn"][0][0]
                elif param == "nb_layers":
                    v = len(param_dict["enc_nn"])
                elif "activation_function" in param:
                    numb = int(param.split("_")[-1])
                    v = param_dict["enc_nn"][numb - 1][1]
                elif "-" in param:
                    p1, p2 = param.split("-")
                    v = param_dict[p1][p2]
                else:
                    v = param_dict[param]
                if isinstance(v, (list, tuple, dict)):
                    v = str(v)
            except Exception:
                v = None
            df.loc[i, param] = v

        id = df.loc[i, "id"]
        file_n = "{}id-{}/metric_id-{}.csv".format(path, id, id)
        df_metric = pd.read_csv(file_n, index_col=0)
        if early_stop_after_epoch:
            df_metric = df_metric.loc[df_metric["epoch"] > early_stop_after_epoch]

        if val_test_params_extract:
            for l in val_test_params_extract:
                if l[0] == "max":
                    f = np.nanmax
                elif l[0] == "min":
                    f = np.nanmin

                if l[0] in ["min", "max"]:
                    try:
                        ind = (
                            df_metric.loc[df_metric[l[1]] == f(df_metric[l[1]])]
                        ).index[0]
                        df.loc[i, l[3]] = df_metric.loc[ind, l[2]]
                    except Exception:
                        pass
                elif l[0] == "last":
                    df.loc[i, l[3]] = df_metric[l[1]].values[-1]
                elif l[0] == "average":
                    df.loc[i, l[3]] = np.nanmean(df_metric[l[1]])

    if sortby:
        df.sort_values(axis=0, ascending=True, by=sortby, inplace=True)

    # save
    if save_file is not False:
        if save_file is None:
            save_file = "{}training_overview-ids-{}-{}.csv".format(
                path, ids_from, ids_to
            )
        df.to_csv(save_file)

    return df


def plot_paths_from_checkpoint(
    saved_models_path=config_constants.saved_models_path,
    model_ids=(1,),
    which="best",
    paths_to_plot=(0,),
    LOB_plot_errors=False,
    plot_boxplot_only=False,
    **options,
):
    """
    function to plot paths (using plot_one_path_with_pred) from a saved model
    checkpoint
    :param model_ids: list of int, the ids of the models to load and plot
    :param which: one of {'best', 'last', 'both'}, which checkpoint to load
    :param paths_to_plot: list of int, see train.train.py, set to None, if only
        LOB_plot_errors should be executed
    :param LOB_plot_errors: bool, whether to plot the error distribution for
        LOB model
    :param plot_boxplot_only: bool, whether to only plot the boxplot of LOB
    :param options: feed directly to train
    :return:
    """
    model_overview_file_name = "{}model_overview.csv".format(saved_models_path)
    if not os.path.exists(model_overview_file_name):
        print("No saved model_overview.csv file")
        return 1
    else:
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)

    for model_id in model_ids:
        if model_id not in df_overview["id"].values:
            print("model_id={} does not exist yet -> skip".format(model_id))
        else:
            desc = (
                df_overview["description"].loc[df_overview["id"] == model_id]
            ).values[0]
            params_dict = json.loads(desc)
            params_dict["model_id"] = model_id
            params_dict["resume_training"] = True
            params_dict["plot_only"] = True
            params_dict["paths_to_plot"] = paths_to_plot
            params_dict["parallel"] = True
            params_dict["saved_models_path"] = saved_models_path
            if LOB_plot_errors:
                params_dict["plot_errors"] = True
                if paths_to_plot is None:
                    params_dict["plot_only"] = False
                if plot_boxplot_only:
                    params_dict["plot_boxplot_only"] = True
                else:
                    params_dict["plot_boxplot_only"] = False
            for key in options:
                params_dict[key] = options[key]

            if which in ["best", "both"]:
                params_dict["load_best"] = True
                train_switcher(**params_dict)
            if which in ["last", "both"]:
                params_dict["load_best"] = False
                train_switcher(**params_dict)
