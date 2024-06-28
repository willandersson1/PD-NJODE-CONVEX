# Path-Dependent Neural Jump ODEs in convex spaces

This repository is the official implementation of the master's thesis _PD-NJ-ODE for Predictions in Convex Spaces_, which in turn is an extension of the second part of the series of works on Neural Jump ODEs.

The one this codebase builds directly on is [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284), which was published as an extension of the original Neural Jump ODE paper [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR).

Much of what was in the original codebase has been removed or simplified to streamline the code, especially as many of the techniques are not relevant for the thesis. However, the core and general structure remains. 

As such, this repo does not support running experiments from the other papers in the NJ series.

In general, the code should be easy to follow either inherently or because of the provided comments and docstrings. In some cases it might be worth referring to the original repo. 

## Requirements

This code was executed using Python 3.7.

To install requirements, download this Repo and cd into it.

Then create a new environment and install all dependencies and this repo.
With [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
 ```sh
conda create --name njode python=3.7
conda activate njode
pip install -r requirements.txt
 ```

--------------------------------------------------------------------------------
## Usage, License & Citation

This code can be used in accordance with the [LICENSE](LICENSE).

If you find this code useful or include parts of it in your own work, 
please cite our papers:  

- [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284)
    ```
    @article{PDNJODE
      url = {https://arxiv.org/abs/2206.14284},
      author = {Krach, Florian and Nübel, Marc and Teichmann, Josef},
      title = {Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs},
      publisher = {arXiv},
      year = {2022},
    }
    ```

- [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR)

    ```
    @inproceedings{
    herrera2021neural,
    title={Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering},
    author={Calypso Herrera and Florian Krach and Josef Teichmann},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=JFKR3WqwyXR}
    }
    ```



## Acknowledgements and References
This code is based on the code-repo of the second paper [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284). The relevant acknowledgements and references are copied from that repo to below. 

The code in the second paper is based on the code-repo of the first paper [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR):
https://github.com/HerreraKrachTeichmann/NJODE  
Parts of this code are based on and/or copied from the code of:
https://github.com/edebrouwer/gru_ode_bayes, of the paper
[GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series](https://arxiv.org/abs/1905.12374)
and the code of: https://github.com/YuliaRubanova/latent_ode, of the paper
[Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907)
and the code of: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/, of the paper
[DeepLOB: Deep convolutional neural networks for limit order books](https://arxiv.org/abs/1808.03668).


--------------------------------------------------------------------------------
# Instructions for running experiments mentioned in the thesis

Configs for the datasets can be found in `NJODE/configs/dataset_configs.py`, while the code to generate the datasets is in `NJODE/synthetic_datasets.py`. The settings are the ones mentioned in the thesis. 

To generate a dataset, execute the following command with the appropriate values for `dataset_name` and `dataset_params`, which can be found in the relevant config files for the existing experiments. A new folder will be created with the training data.
```sh
python data_utils.py --dataset_name=DATASET_NAME --dataset_params=DATASET_PARAMS
```

For example, the 1-simplex dataset can be generated by running 
```sh
python data_utils.py --dataset_name=BMWeights --dataset_params=BM_WEIGHTS_SIMPLEX2D
```

The configs for the models in the experiments are in `NJODE/configs/`, in particular `configs_Ball2D_BM.py`, `configs_BM_weigts.py`, `configs_RBM.py`, and `configs_Rectangle.py`. The models themselves are implemented in `NJODE/models.py`. Running a model (training, validating, and testing) according to the already-declared params list can be done by executing the following command. Make sure to replace `PARAMS_LIST` and `OVERVIEW_DICT` with the ones you want. The `first_id`, `NB_JOBS`, and `NB_CPUS` parameters can obviously also be changed.
```sh
python run.py --params=PARAMS_LIST --get_overview=OVERVIEW_DICT --first_id=1 --NB_JOBS=1 --NB_CPUS=1
```

For example, for the vertex approach on the 1-simplex 
```sh
python run.py --params=param_list_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH --get_overview=overview_dict_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH --first_id=1 --NB-JOBS=1 --NB_CPUS=1
```

This will create a new folder with an appropriate name that will store the results, along with plots, metrics, checkpoints, etc.

The other flags you can use are as follows, although they are also documented in the code. Additionally, parallel training is possible.

List of all flags:
- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **first_id**: First id of the given list / to start training of
- **get_overview**: name of the dict (defined in config.py) defining input for extras.get_training_overview
- **USE_GPU**: whether to use GPU for training
- **ANOMALY_DETECTION**: whether to run in torch debug mode
- **SEND**: whether to send results via telegram
- **NB_CPUS**: nb of CPUs used by each training
- **model_ids**: List of model ids to run
- **DEBUG**: whether to run parallel in debug mode
- **saved_models_path**: path where the models are saved
- **overwrite_params**: name of dict (defined in config.py) to use for overwriting params
- **plot_paths**: name of the dict (in config.py) defining input for extras.plot_paths_from_checkpoint
- **climate_crossval**: name of the dict (in config.py) defining input for extras.get_cross_validation
- **plot_conv_study**: name of the dict (in config.py) defining input for extras.plot_convergence_study


