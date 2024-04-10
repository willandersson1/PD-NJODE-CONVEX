"""
author: Florian Krach
"""

import synthetic_datasets
import train

from NJODE.configs.dataset_configs import DATA_DICTS


def train_switcher(**params):
    """
    function to call the correct train function depending on the dataset. s.t.
    parallel training easily works altough different fuctions need to be called
    :param params: all params needed by the train function, as passed by
            parallel_training
    :return: function call to the correct train function
    """
    if "dataset" not in params:
        if "data_dict" not in params:
            raise KeyError('the "dataset" needs to be specified')
        else:
            data_dict = params["data_dict"]
            if isinstance(data_dict, str):
                data_dict = DATA_DICTS[data_dict]
            params["dataset"] = data_dict["model_name"]
    if (
        params["dataset"] in list(synthetic_datasets.DATASETS)
        or "combined" in params["dataset"]
        or "FBM[" in params["dataset"]
    ):
        return train.train(**params)
    else:
        raise ValueError('the specified "dataset" is not supported')
