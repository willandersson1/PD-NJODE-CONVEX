"""
author: Florian Krach & Calypso Herrera

data utilities for creating and loading synthetic test datasets
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from absl import app, flags
from configs import config_utils
from configs.dataset_configs import DATA_DICTS
from synthetic_datasets import DATASETS as _STOCK_MODELS
from torch.utils.data import Dataset

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_params", None, "name of the dict with data hyper-params")
flags.DEFINE_string("dataset_name", None, "name of the dataset to generate")
flags.DEFINE_integer("seed", 0, "seed for making dataset generation reproducible")

data_path = config_utils.data_path
training_data_path = config_utils.training_data_path


# =====================================================================================================================
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_dataset_overview(training_data_path=training_data_path):
    data_overview = "{}dataset_overview.csv".format(training_data_path)
    makedirs(training_data_path)
    if not os.path.exists(data_overview):
        df_overview = pd.DataFrame(
            data=None, columns=["name", "id", "description"], dtype=object
        )
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview


def create_dataset(stock_model_name, hyperparam_dict=None, seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_name: str, name of the stockmodel, see _STOCK_MODELS
    :param hyperparam_dict: dict, contains all needed parameters for the model
            it can also contain additional options for dataset generation:
                - masked    None, float or array of floats. if None: no mask is
                            used; if float: lambda of the poisson distribution;
                            if array of floats: gives the bernoulli probability
                            for each coordinate to be observed
                - timelag_in_dt_steps   None or int. if None: no timelag used;
                            if int: number of (dt) steps by which the 1st
                            coordinate is shifted to generate the 2nd coord.,
                            this is used to generate mask accordingly (such that
                            second coord. is observed whenever the information
                            is already known from first coord.)
                - timelag_shift1    bool, if True: observe the second coord.
                            additionally only at one step after the observation
                            times of the first coord. with the given prob., if
                            False: observe the second coord. additionally at all
                            times within timelag_in_dt_steps after observation
                            times of first coordinate, at each time with given
                            probability (in masked); default: True
                - X_dependent_observation_prob   not given or str, if given:
                            string that can be evaluated to a function that is
                            applied to the generated paths to get the
                            observation probability for each coordinate
                - obs_scheme   dict, if given: specifies the observation scheme
                - obs_noise    dict, if given: add noise to the observations
                            the dict needs the following keys: 'distribution'
                            (defining the distribution of the noise), and keys
                            for the parameters of the distribution (depending on
                            the used distribution); supported distributions
                            {'normal'}. Be aware that the noise needs to be
                            centered for the model to be able to learn the
                            correct dynamics.

    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict["model_name"] = stock_model_name
    original_desc = json.dumps(hyperparam_dict, sort_keys=True)
    obs_perc = hyperparam_dict["obs_perc"]

    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    # stock paths shape: [nb_paths, dim, time_steps]
    stock_paths, dt = stockmodel.generate_paths()
    size = stock_paths.shape

    observed_dates = np.random.random(size=(size[0], size[2]))
    if "X_dependent_observation_prob" in hyperparam_dict:
        print("use X_dependent_observation_prob")
        prob_f = eval(hyperparam_dict["X_dependent_observation_prob"])
        obs_perc = prob_f(stock_paths)
    observed_dates = (observed_dates < obs_perc) * 1
    observed_dates[:, 0] = 1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)

    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = "{}-{}".format(stock_model_name, time_id)
    path = "{}{}/".format(training_data_path, file_name)
    hyperparam_dict["dt"] = dt
    if os.path.exists(path):
        print("Path already exists - abort")
        raise ValueError
    df_app = pd.DataFrame(
        data=[[stock_model_name, time_id, original_desc]],
        columns=["name", "id", "description"],
    )
    df_overview = pd.concat([df_overview, df_app], ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open("{}data.npy".format(path), "wb") as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open("{}metadata.txt".format(path), "w") as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [nb_paths, dimension, time_steps]
    return path, time_id


def _get_time_id(
    stock_model_name="BlackScholes", time_id=None, path=training_data_path
):
    """
    if time_id=None, get the time id of the newest dataset with the given name
    :param stock_model_name: str
    :param time_id: None or int
    :return: int, time_id
    """
    if time_id is None:
        df_overview, _ = get_dataset_overview(path)
        df_overview = df_overview.loc[df_overview["name"] == stock_model_name]
        if len(df_overview) > 0:
            time_id = np.max(df_overview["id"].values)
        else:
            time_id = None
    return time_id


def _get_dataset_name_id_from_dict(data_dict):
    # NOTE this is used! don't know why it's greyed out

    if isinstance(data_dict, str):
        data_dict = DATA_DICTS[data_dict]

    desc = json.dumps(data_dict, sort_keys=True)
    df_overview, _ = get_dataset_overview()
    which = df_overview.loc[df_overview["description"] == desc].index
    if len(which) == 0:
        raise ValueError(
            "the given dataset does not exist yet, please generate it "
            "first using data_utils.py. \ndata_dict: {}".format(data_dict)
        )
    elif len(which) > 1:
        print(
            "WARNING: multiple datasets match the description, returning the "
            "last one. To uniquely identify the wanted dataset, please "
            "provide the dataset_id instead of the data_dict."
        )
    return list(df_overview.loc[which[-1], ["name", "id"]].values)


def load_metadata(stock_model_name="BlackScholes", time_id=None):
    """
    load the metadata of a dataset specified by its name and id
    :return: dict (with hyperparams of the dataset)
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = "{}{}-{}/".format(training_data_path, stock_model_name, int(time_id))
    with open("{}metadata.txt".format(path), "r") as f:
        hyperparam_dict = json.load(f)
    return hyperparam_dict


def load_dataset(stock_model_name="BlackScholes", time_id=None):
    """
    load a saved dataset by its name and id
    :param stock_model_name: str, name
    :param time_id: int, id
    :return: np.arrays of stock_paths, observed_dates, number_observations
                dict of hyperparams of the dataset
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = "{}{}-{}/".format(training_data_path, stock_model_name, int(time_id))

    if stock_model_name == "LOB":
        with open("{}data.npy".format(path), "rb") as f:
            samples = np.load(f)
            times = np.load(f)
            eval_samples = np.load(f)
            eval_times = np.load(f)
            eval_labels = np.load(f)
        with open("{}metadata.txt".format(path), "r") as f:
            hyperparam_dict = json.load(f)
        return samples, times, eval_samples, eval_times, eval_labels, hyperparam_dict

    with open("{}metadata.txt".format(path), "r") as f:
        hyperparam_dict = json.load(f)
    with open("{}data.npy".format(path), "rb") as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
        if "obs_noise" in hyperparam_dict:
            obs_noise = np.load(f)
        else:
            obs_noise = None

    return stock_paths, observed_dates, nb_obs, hyperparam_dict, obs_noise


class IrregularDataset(Dataset):
    """
    class for iterating over a dataset
    """

    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, hyperparam_dict, obs_noise = load_dataset(
            stock_model_name=model_name, time_id=time_id
        )
        if idx is None:
            idx = np.arange(hyperparam_dict["nb_paths"])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]
        self.path_idxs = idx

    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        obs_noise = None
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {
            "idx": idx,
            "stock_path": self.stock_paths[idx],
            "observed_dates": self.observed_dates[idx],
            "nb_obs": self.nb_obs[idx],
            "dt": self.metadata["dt"],
            "obs_noise": obs_noise,
            "path_idx": self.path_idxs[idx],
        }


def CustomCollateFnGen(func_names=None):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param func_names: list of str, with all function names, see _get_func
    :return: collate function, int (multiplication factor of dimension before
                and after applying the functions)
    """
    # get functions that should be applied to X, additionally to identity
    functions = []
    mult = len(functions) + 1

    def custom_collate_fn(batch):
        dt = batch[0]["dt"]
        stock_paths = np.concatenate([b["stock_path"] for b in batch], axis=0)
        observed_dates = np.concatenate([b["observed_dates"] for b in batch], axis=0)
        obs_noise = None
        mask = None
        nb_obs = torch.tensor(np.concatenate([b["nb_obs"] for b in batch], axis=0))

        # here axis=1, since we have elements of dim
        #    [batch_size, data_dimension] => add as new data_dimensions
        start_X = torch.tensor(stock_paths[:, :, 0], dtype=torch.float32)
        X = []
        M = None
        start_M = None
        times = []
        time_ptr = [0]
        obs_idx = []
        current_time = 0.0
        counter = 0
        for t in range(1, observed_dates.shape[-1]):
            current_time += dt
            if observed_dates[:, t].sum() > 0:
                times.append(current_time)
                for i in range(observed_dates.shape[0]):
                    if observed_dates[i, t] == 1:
                        counter += 1
                        # here axis=0, since only 1 dim (the data_dimension),
                        #    i.e. the batch-dim is cummulated outside together
                        #    with the time dimension
                        X.append(stock_paths[i, :, t])
                        obs_idx.append(i)
                time_ptr.append(counter)

        assert len(obs_idx) == observed_dates[:, 1:].sum()
        path_idxs = [b["path_idx"][0] for b in batch]
        res = {
            "times": np.array(times),
            "time_ptr": np.array(time_ptr),
            "obs_idx": torch.tensor(obs_idx, dtype=torch.long),
            "start_X": start_X,
            "n_obs_ot": nb_obs,
            "X": torch.tensor(np.array(X), dtype=torch.float32),
            "true_paths": stock_paths,
            "observed_dates": observed_dates,
            "true_mask": mask,
            "obs_noise": obs_noise,
            "M": M,
            "start_M": start_M,
            "path_idxs": path_idxs,
        }
        return res

    return custom_collate_fn, mult


def main(arg):
    """
    function to generate datasets
    """
    del arg
    if FLAGS.dataset_name:
        dataset_name = FLAGS.dataset_name
        print("dataset_name: {}".format(dataset_name))
    else:
        raise ValueError("Please provide --dataset_name")
    if FLAGS.dataset_params:
        dataset_params = DATA_DICTS[FLAGS.dataset_params]
        print("dataset_params: {}".format(dataset_params))
    else:
        raise ValueError("Please provide --dataset_params")

    create_dataset(
        stock_model_name=dataset_name, hyperparam_dict=dataset_params, seed=FLAGS.seed
    )


if __name__ == "__main__":
    app.run(main)
