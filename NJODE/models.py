"""
authors: William Andersson & Florian Krach & Marc Nuebel & Calypso Herrera

implementation of the model for NJ-ODE
"""

import copy as copy
import os

import iisignature as sig
import numpy as np

# =====================================================================================================================
import torch
from configs.dataset_configs import zero_pen_func


# =====================================================================================================================
def init_weights(m, bias=0.0):  # initialize weights for model for linear NN
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(bias)


def save_checkpoint(model, optimizer, path, epoch, retrain_epoch=0):
    """
    save a trained torch model and the used optimizer at the given path, s.t.
    training can be resumed at the exact same point
    :param model: a torch model, e.g. instance of NJODE
    :param optimizer: a torch optimizer
    :param path: str, the path where to save the model
    :param epoch: int, the current epoch
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, "checkpt.tar")
    torch.save(
        {
            "epoch": epoch,
            "weight": model.weight,
            "retrain_epoch": retrain_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def get_ckpt_model(ckpt_path, model, optimizer, device):
    """
    load a saved torch model and its optimizer, inplace
    :param ckpt_path: str, path where the model is saved
    :param model: torch model instance, of which the weights etc. should be
            reloaded
    :param optimizer: torch optimizer, which should be loaded
    :param device: the device to which the model should be loaded
    """
    ckpt_path = os.path.join(ckpt_path, "checkpt.tar")
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    state_dict = checkpt["model_state_dict"]
    optimizer.load_state_dict(checkpt["optimizer_state_dict"])
    model.load_state_dict(state_dict)
    model.epoch = checkpt["epoch"]
    model.weight = checkpt["weight"]
    if "retrain_epoch" in checkpt:
        model.retrain_epoch = checkpt["retrain_epoch"]
    model.to(device)


def compute_loss(
    X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10, weight=0.5, M_obs=None
):
    """
    loss function from the paper
    :param X_obs: torch.tensor, the true X values at the observations
    :param Y_obs: torch.tensor, the predicted values at the observation
    :param Y_obs_bj: torch.tensor, the predicted values before the jump at the
            observation
    :param n_obs_ot: torch.tensor, the number of observations over the entire
            time-line for each element of the batch
    :param batch_size: int or float
    :param eps: float, a small constant which is added before taking torch.sqrt
            s.t. the sqrt never becomes zero (which would yield NaNs for the
            gradient)
    :param weight: float in [0,1], weighting of the two parts of the loss
            function,
                0.5: standard loss as described in paper
                (0.5, 1): more weight to be correct after the jump, can be
                theoretically justified similar to standard loss
                1: only the value after the jump is trained
    :param M_obs: None or torch.tensor, if not None: same size as X_obs with
            0  and 1 entries, telling which coordinates were observed
    :return: torch.tensor (with the loss, reduced to 1 dim)
    """
    if M_obs is None:
        inner = (
            2 * weight * torch.sqrt(torch.sum((X_obs - Y_obs) ** 2, dim=1) + eps)
            + 2
            * (1 - weight)
            * torch.sqrt(torch.sum((Y_obs_bj - Y_obs) ** 2, dim=1) + eps)
        ) ** 2
    else:
        inner = (
            2
            * weight
            * torch.sqrt(torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) + eps)
            + 2
            * (1 - weight)
            * torch.sqrt(torch.sum(M_obs * (Y_obs_bj - Y_obs) ** 2, dim=1) + eps)
        ) ** 2
    outer = torch.sum(inner / n_obs_ot)
    return outer / batch_size


nonlinears = {  # dictionary of used non-linear activation functions. Reminder inputs
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "prelu": torch.nn.PReLU,
}


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    if nn_desc is not None and len(nn_desc) == 0:
        return torch.nn.Identity()
    if nn_desc is None:
        layers = [
            torch.nn.Linear(in_features=input_size, out_features=output_size, bias=bias)
        ]  # take linear NN if
        # not specified otherwise
    else:
        layers = [
            torch.nn.Linear(
                in_features=input_size, out_features=nn_desc[0][0], bias=bias
            )
        ]  # first linear
        # layer to specified dimension
        if len(nn_desc) > 1:
            for i in range(
                len(nn_desc) - 1
            ):  # add multiple layers if multiple were given as input
                layers.append(
                    nonlinears[nn_desc[i][1]]()
                )  # add layer with specified activation function
                layers.append(torch.nn.Dropout(p=dropout_rate))  # add dropout layer
                layers.append(
                    torch.nn.Linear(
                        nn_desc[i][0],
                        nn_desc[i + 1][
                            0
                        ],  # add linear layer between specified dimensions
                        bias=bias,
                    )
                )
        layers.append(
            nonlinears[nn_desc[-1][1]]()
        )  # last specified activation function
        layers.append(torch.nn.Dropout(p=dropout_rate))  # add another dropout layer
        layers.append(
            torch.nn.Linear(
                in_features=nn_desc[-1][0], out_features=output_size, bias=bias
            )
        )  # linear
        # output layer
    return torch.nn.Sequential(*layers)  # return the constructed NN


# =====================================================================================================================
class ODEFunc(torch.nn.Module):
    """
    implementing continuous update between observatios, f_{\theta} in paper
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        ode_nn,
        dropout_rate=0.0,
        bias=True,
        input_current_t=False,
        input_sig=False,
        sig_depth=3,
        coord_wise_tau=False,
        input_scaling_func="tanh",
        use_current_y_for_ode=False,
    ):
        super().__init__()  # initialize class with given parameters
        self.input_current_t = input_current_t
        self.sig_depth = sig_depth
        self.use_current_y_for_ode = use_current_y_for_ode
        if input_scaling_func in ["id", "identity"]:
            self.sc_fun = torch.nn.Identity()
            print("neuralODE use input scaling with identity (no scaling)")
        else:
            self.sc_fun = torch.tanh
            print("neuralODE use input scaling with tanh")

        # create feed-forward NN, f(H,X,tau,t-tau)
        if coord_wise_tau:
            add = 2 * input_size
        else:
            add = 2
        if input_current_t:
            if coord_wise_tau:
                add += input_size
            else:
                add += 1
        if use_current_y_for_ode:
            add += input_size
        self.f = get_ffnn(  # get a feedforward NN with the given specifications
            input_size=input_size + hidden_size + add,
            output_size=hidden_size,
            nn_desc=ode_nn,
            dropout_rate=dropout_rate,
            bias=bias,
        )

    def forward(self, x, h, tau, tdiff, signature=None, current_y=None):
        # dimension should be (batch, input_size) for x, (batch, hidden) for h,
        #    (batch, 1) for times

        input_f = torch.cat([self.sc_fun(x), self.sc_fun(h), tau, tdiff], dim=1)

        if self.input_current_t:
            input_f = torch.cat([input_f, tau + tdiff], dim=1)
        if self.use_current_y_for_ode:
            input_f = torch.cat([input_f, self.sc_fun(current_y)], dim=1)

        df = self.f(input_f)
        return df


class FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks with tanh applied to inputs and the
    option to use a residual NN version
    """

    def __init__(
        self,
        input_size,
        output_size,
        nn_desc,
        dropout_rate=0.0,
        bias=True,
        residual=False,
        masked=False,
        recurrent=False,
        input_sig=False,
        sig_depth=3,
        clamp=None,
        input_t=False,
        t_size=None,
        **kwargs,
    ):
        super().__init__()

        self.use_lstm = False
        in_size = input_size
        if recurrent:
            in_size += output_size
        self.recurrent = recurrent
        self.output_size = output_size
        self.input_sig = input_sig
        self.sig_depth = sig_depth
        self.input_t = input_t
        if self.input_t:
            in_size += t_size
        self.clamp = clamp
        self.lstm = None
        self.ffnn = get_ffnn(
            input_size=in_size,
            output_size=output_size,
            nn_desc=nn_desc,
            dropout_rate=dropout_rate,
            bias=bias,
        )

        if residual:
            print(
                "use residual network: input_size={}, output_size={}".format(
                    input_size, output_size
                )
            )
            if input_size <= output_size:
                self.case = 1
            if input_size > output_size:
                self.case = 2
        else:
            self.case = 0

    def forward(self, nn_input, mask=None, sig=None, h=None, t=None):
        identity = None
        if self.case == 1:
            identity = torch.zeros((nn_input.shape[0], self.output_size)).to(
                self.device
            )
            identity[:, 0 : nn_input.shape[1]] = nn_input
        elif self.case == 2:
            identity = nn_input[:, 0 : self.output_size]

        if self.recurrent:
            assert h is not None
            x = nn_input
        else:
            x = torch.tanh(nn_input)  # maybe not helpful
        if self.recurrent:
            x = torch.cat((x, h), dim=1)
        if self.input_t:
            x = torch.cat((x, t), dim=1)
        out = self.ffnn(x.float())

        if self.case == 0:
            pass
        else:
            out = identity + out

        if self.clamp is not None:
            out = torch.clamp(out, min=-self.clamp, max=self.clamp)

        return out

    @property
    def device(self):
        device = next(self.parameters()).device
        return device


class NJODE(torch.nn.Module):
    """
    NJ-ODE model
    """

    def __init__(  # initialize the class by naming relevant features
        self,
        input_size,
        hidden_size,
        output_size,
        ode_nn,
        readout_nn,
        enc_nn,
        use_rnn,
        in_shape_func,
        penalising_func,
        lmbda=0,
        bias=True,
        dropout_rate=0,
        solver="euler",
        weight=0.5,
        weight_decay=1.0,
        **options,
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param options: kwargs, used:
                - "classifier_nn"
                - "options" with arg a dict passed from train.train
                    used kwords: 'which_loss', 'residual_enc_dec',
                    'residual_enc', 'residual_dec',
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode', 'enc_input_t' are used)
        """
        super().__init__()

        self.epoch = 1
        self.retrain_epoch = 0
        self.weight = weight
        self.weight_decay = weight_decay
        self.use_rnn = use_rnn  # use RNN for jumps

        # get options from the options of train input
        options1 = options["options"]
        self.in_shape_func = in_shape_func
        self.lmbda = lmbda
        self.penalising_func = penalising_func

        self.residual_enc = True
        self.residual_dec = True
        # for backward compatibility, set residual_enc to False as default
        #   if RNN is used. (before, it was not possible to use residual
        #   connections with RNNs)
        if self.use_rnn:
            self.residual_enc = False
        if "residual_enc_dec" in options1:
            residual_enc_dec = options1["residual_enc_dec"]
            self.residual_enc = residual_enc_dec
            self.residual_dec = residual_enc_dec
        if "residual_enc" in options1:
            self.residual_enc = options1["residual_enc"]
        if "residual_dec" in options1:
            self.residual_dec = options1["residual_dec"]

        self.input_current_t = False
        if "input_current_t" in options1:
            self.input_current_t = options1["input_current_t"]
        self.input_sig = False
        self.level = 2
        if "level" in options1:
            self.level = options1["level"]
        self.sig_depth = sig.siglength(input_size + 1, self.level)
        self.masked = False
        self.use_y_for_ode = True
        if "use_y_for_ode" in options1:
            self.use_y_for_ode = options1["use_y_for_ode"]
        self.use_current_y_for_ode = False
        if "use_current_y_for_ode" in options1:
            self.use_current_y_for_ode = options1["use_current_y_for_ode"]
        self.coord_wise_tau = False
        self.enc_input_t = False
        if "enc_input_t" in options1:
            self.enc_input_t = options1["enc_input_t"]
        self.clamp = None
        if "clamp" in options1:
            self.clamp = options1["clamp"]
        self.ode_input_scaling_func = "tanh"
        if "ode_input_scaling_func" in options1:
            self.ode_input_scaling_func = options1["ode_input_scaling_func"]
        self.class_loss_weight = 1.0
        self.loss_weight = 1.0
        t_size = 2
        if self.coord_wise_tau:
            t_size = 2 * input_size

        use_observation_as_input = None
        if "use_observation_as_input" in options1:
            use_observation_as_input = options1["use_observation_as_input"]
        if use_observation_as_input is None:
            self.use_observation_as_input = lambda x: True
        elif isinstance(use_observation_as_input, bool):
            self.use_observation_as_input = lambda x: use_observation_as_input
        elif isinstance(use_observation_as_input, float):
            self.use_observation_as_input = (
                lambda x: np.random.random() < use_observation_as_input
            )
        elif isinstance(use_observation_as_input, str):
            self.use_observation_as_input = eval(use_observation_as_input)
        val_use_observation_as_input = None
        if "val_use_observation_as_input" in options1:
            val_use_observation_as_input = options1["val_use_observation_as_input"]
        if val_use_observation_as_input is None:
            self.val_use_observation_as_input = self.use_observation_as_input
        elif isinstance(val_use_observation_as_input, bool):
            self.val_use_observation_as_input = lambda x: val_use_observation_as_input
        elif isinstance(val_use_observation_as_input, float):
            self.val_use_observation_as_input = (
                lambda x: np.random.random() < val_use_observation_as_input
            )
        elif isinstance(val_use_observation_as_input, str):
            self.val_use_observation_as_input = eval(val_use_observation_as_input)

        self.ode_f = ODEFunc(
            input_size=input_size,
            hidden_size=hidden_size,
            ode_nn=ode_nn,
            dropout_rate=dropout_rate,
            bias=bias,
            input_current_t=self.input_current_t,
            input_sig=self.input_sig,
            sig_depth=self.sig_depth,
            coord_wise_tau=self.coord_wise_tau,
            input_scaling_func=self.ode_input_scaling_func,
            use_current_y_for_ode=self.use_current_y_for_ode,
        )
        self.encoder_map = FFNN(
            input_size=input_size,
            output_size=hidden_size,
            nn_desc=enc_nn,
            dropout_rate=dropout_rate,
            bias=bias,
            recurrent=self.use_rnn,
            masked=self.masked,
            residual=self.residual_enc,
            input_sig=self.input_sig,
            sig_depth=self.sig_depth,
            input_t=self.enc_input_t,
            t_size=t_size,
        )
        self.readout_map = FFNN(
            input_size=hidden_size,
            output_size=output_size,
            nn_desc=readout_nn,
            dropout_rate=dropout_rate,
            bias=bias,
            residual=self.residual_dec,
            clamp=self.clamp,
        )

        self.solver = solver
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def weight_decay_step(self):
        inc = self.weight - 0.5
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(
        self, h, delta_t, current_time, last_X, tau, signature=None, current_y=None
    ):
        h = h + delta_t * self.ode_f(
            x=last_X,
            h=h,
            tau=tau,
            tdiff=current_time - tau,
            signature=signature,
            current_y=current_y,
        )

        current_time += delta_t
        return h, current_time

    def recreate_data(self, times, time_ptr, X, obs_idx, start_X):
        """
        recreates matrix of all observations
        first dim: which data path
        second dim: which time
        """
        # shape: [batch_size, time_steps+1, dimension]
        data = np.empty(shape=(start_X.shape[0], 1 + len(times), start_X.shape[1]))
        data[:] = np.nan
        data[:, 0, :] = start_X.detach().cpu().numpy()

        X = X.detach().cpu().numpy()
        for j, time in enumerate(times):
            start = time_ptr[j]
            end = time_ptr[j + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            data[i_obs, j + 1, :] = X_obs
        times_new = np.concatenate(([0], times), axis=0)

        return times_new, data

    def get_signature(self, times, time_ptr, X, obs_idx, start_X):
        """
        Input: See forward
        Returns: signature of paths as nested list
        """
        # reconstructing the data, shape: [batch_size, time_steps+1, dim]
        times_new, data = self.recreate_data(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, start_X=start_X
        )

        # list of list of lists, shape: [batch_size, obs_dates[j], sig_length]
        signature = []
        for j in range(data.shape[0]):  # iterate over batch
            data_j = data[j, :, :]
            observed_j = []
            for i in range(data_j.shape[0]):
                # if the current batch-sample has an observation at the current
                #   time, add it to the list of observations
                if not np.all(np.isnan(data_j[i])):
                    observed_j += [i]
            data_j = data_j[observed_j, :]

            # replace no observations with last observation
            for i in range(1, data_j.shape[0]):
                ks = np.isnan(data_j[i, :])
                data_j[i, ks] = data_j[i - 1, ks]

            times_j = times_new[observed_j].reshape(-1, 1)
            # add times to data for signature call
            path_j = np.concatenate((times_j, data_j), axis=1)
            # the following computes the signatures of all partial paths, from
            #   start to each point of the path
            signature.append(sig.sig(path_j, self.level, 2))

        return signature

    def forward(
        self,
        times,
        time_ptr,
        X,
        obs_idx,
        delta_t,
        T,
        start_X,
        n_obs_ot,
        return_path=False,
        get_loss=True,
        until_T=False,
        M=None,
        start_M=None,
        dim_to=None,
        return_at_last_obs=False,
        epoch=None,
    ):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :param return_at_last_obs: bool, whether to return the hidden state at
                the last observation time or at the final time
        :param epoch: int, the current epoch

        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """
        epoch = 0 if epoch is None else epoch

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = data_dim
        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, data_dim).to(self.device)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1).to(self.device)
        current_time = 0.0
        loss = torch.tensor(0.0).to(self.device)
        c_sig = None
        num_in, num_out = 0, 0

        start_M = None

        h = self.encoder_map(
            start_X,
            mask=start_M,
            sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)).to(self.device),
            t=torch.cat((tau, current_time - tau), dim=1).to(self.device),
        )

        if return_path:
            path_t = [0]
            path_h = [h]
            path_y = [self.full_readout(h)]
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (
                obs_time - 1e-10 * delta_t
            ):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                h, current_time = self.ode_step(
                    h,
                    delta_t_,
                    current_time,
                    last_X=last_X,
                    tau=tau,
                    signature=c_sig,
                    current_y=self.full_readout(h),
                )

                # NOTE: here (and elsewhere when checking isin) using the "vanilla"
                #  self.readout_map instead of the new one. This is correct!
                isin = self.in_shape_func(self.readout_map(h))
                num_in += isin.count(True)
                num_out += isin.count(False)
                loss += self.additional_term(h, batch_size, delta_t)

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.full_readout(h))

            # Reached an observation - only update those elements of the batch,
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            M_obs = None

            # decide whether to use observation as input
            if self.training:  # check whether model is in training or eval mode
                use_as_input = self.use_observation_as_input(epoch)
            else:
                use_as_input = self.val_use_observation_as_input(epoch)

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = self.full_readout(h)
            if use_as_input:
                X_obs_impute = X_obs
                temp = h.clone()
                c_sig_iobs = None

                temp[i_obs.long()] = self.encoder_map(
                    X_obs_impute,
                    mask=M_obs,
                    sig=c_sig_iobs,
                    h=h[i_obs],
                    t=torch.cat((tau[i_obs], current_time - tau[i_obs]), dim=1),
                )
                h = temp
                Y = self.full_readout(h)

                # update h and sig at last observation
                h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
                sig_at_last_obs = c_sig

                isin = self.in_shape_func(self.readout_map(h))
                num_in += isin.count(True)
                num_out += isin.count(False)
            else:
                Y = Y_bj

            if get_loss:
                loss += compute_loss(
                    X_obs=X_obs[:, :dim_to],
                    Y_obs=Y[i_obs.long(), :dim_to],
                    Y_obs_bj=Y_bj[i_obs.long(), :dim_to],
                    n_obs_ot=n_obs_ot[i_obs.long()],
                    batch_size=batch_size,
                    weight=self.weight,
                    M_obs=M_obs,
                )

            # make update of last_X and tau, that is not inplace
            #    (otherwise problems in autograd)
            if use_as_input:
                temp_X = last_X.clone()
                temp_tau = tau.clone()
                temp_X[i_obs.long()] = X_obs_impute
                if self.coord_wise_tau:
                    _M = torch.zeros_like(temp_tau)
                    _M[i_obs] = M_obs
                    temp_tau[_M == 1] = obs_time.astype(np.float64)
                else:
                    temp_tau[i_obs.long()] = obs_time.astype(np.float64)
                last_X = temp_X
                tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                path_y.append(Y)

        # after every observation has been processed, propagating until T
        if until_T:
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                h, current_time = self.ode_step(
                    h,
                    delta_t_,
                    current_time,
                    last_X=last_X,
                    tau=tau,
                    signature=c_sig,
                    current_y=self.full_readout(h),
                )

                isin = self.in_shape_func(self.readout_map(h))
                num_in += isin.count(True)
                num_out += isin.count(False)
                loss += self.additional_term(h, batch_size, delta_t)

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.full_readout(h))

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            # NOTE the output is different if returning the path
            return (
                h,
                loss,
                np.array(path_t),
                torch.stack(path_h),
                torch.stack(path_y)[:, :, :dim_to],
            )
        else:
            return h, loss, num_in, num_out

    def evaluate(
        self,
        times,
        time_ptr,
        X,
        obs_idx,
        delta_t,
        T,
        start_X,
        n_obs_ot,
        stockmodel,
        cond_exp_fun_kwargs=None,
        diff_fun=lambda x, y: np.nanmean((x - y) ** 2),
        return_paths=False,
        M=None,
        true_paths=None,
        start_M=None,
        true_mask=None,
        mult=None,
        use_stored_cond_exp=False,
        path_idxs=None,
    ):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param stockmodel: stock_model.StockModel instance, used to compute true
                cond. exp.
        :param cond_exp_fun_kwargs: dict, the kwargs for the cond. exp. function
                currently not used
        :param diff_fun: function, to compute difference between optimal and
                predicted cond. exp
        :param return_paths: bool, whether to return also the paths
        :param M: see forward
        :param start_M: see forward
        :param true_paths: np.array, shape [batch_size, dimension, time_steps+1]
        :param true_mask: as true_paths, with mask entries
        :param mult: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim/mult. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param use_stored_cond_exp: bool, whether to recompute the cond. exp.

        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        dim = start_X.shape[1]
        dim_to = dim
        if mult is not None and mult > 1:
            dim_to = round(dim / mult)

        _, _, path_t, path_h, path_y = self.forward(
            times,
            time_ptr,
            X,
            obs_idx,
            delta_t,
            T,
            start_X,
            None,
            return_path=True,
            get_loss=False,
            until_T=True,
            M=M,
            start_M=start_M,
            dim_to=dim_to,
        )

        if true_paths is None:
            if M is not None:
                M = M.detach().numpy()[:, :dim_to]
            _, true_path_t, true_path_y = stockmodel.compute_cond_exp(
                times,
                time_ptr,
                X.detach().numpy()[:, :dim_to],
                obs_idx.detach().numpy(),
                delta_t,
                T,
                start_X.detach().numpy()[:, :dim_to],
                n_obs_ot.detach().numpy(),
                return_path=True,
                get_loss=False,
                M=M,
                store_and_use_stored=use_stored_cond_exp,
                path_idxs=path_idxs,
            )
        else:
            true_t = np.linspace(0, T, true_paths.shape[2])
            which_t_ind = []
            for t in path_t:
                which_t_ind.append(np.argmin(np.abs(true_t - t)))
            true_path_y = true_paths[:, :dim_to, which_t_ind]
            true_path_y = np.transpose(true_path_y, axes=(2, 0, 1))
            true_path_t = true_t[which_t_ind]

        if path_y.detach().numpy().shape == true_path_y.shape:
            eval_loss = diff_fun(path_y.detach().numpy(), true_path_y)
        else:
            print(path_y.detach().numpy().shape)
            print(true_path_y.shape)
            raise ValueError("Shapes do not match!")
        if return_paths:
            return eval_loss, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_loss

    def get_pred(
        self, times, time_ptr, X, obs_idx, delta_t, T, start_X, M=None, start_M=None
    ):
        """
        get predicted path
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param M: see forward
        :param start_M: see forward
        :return: dict, with prediction y and times t
        """
        self.eval()
        _, _, path_t, path_h, path_y = self.forward(
            times=times,
            time_ptr=time_ptr,
            X=X,
            obs_idx=obs_idx,
            delta_t=delta_t,
            T=T,
            start_X=start_X,
            n_obs_ot=None,
            return_path=True,
            get_loss=False,
            until_T=True,
            M=M,
            start_M=start_M,
        )
        return {"pred": path_y, "pred_t": path_t}

    def additional_term(self, h, batch_size, delta_t):
        # Return it if needed and supported, otherwise just return 0
        if hasattr(self, "lmbda") and self.lmbda is not None and self.lmbda > 0:
            point_loss = self.lmbda * self.penalising_func(self.readout_map(h))
            return (1 / batch_size) * (delta_t * torch.sum(point_loss))
        return 0

    def full_readout(self, h):
        # Quick hack to make it easier for the convex projection model
        return self.readout_map(h)


class NJODE_convex_projection(NJODE):
    """
    NJ-ODE model with optimal projection onto convex set. Inherits from normal NJODE
    """

    def __init__(  # initialize the class by naming relevant features
        self,
        input_size,
        hidden_size,
        output_size,
        ode_nn,
        readout_nn,
        enc_nn,
        use_rnn,
        in_shape_func,
        penalising_func,
        project,
        project_only_at_inference=False,
        lmbda=0,
        bias=True,
        dropout_rate=0,
        solver="euler",
        weight=0.5,
        weight_decay=1.0,
        **options,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            ode_nn=ode_nn,
            readout_nn=readout_nn,
            enc_nn=enc_nn,
            use_rnn=use_rnn,
            in_shape_func=in_shape_func,
            penalising_func=penalising_func,
            lmbda=lmbda,
            bias=bias,
            dropout_rate=dropout_rate,
            solver=solver,
            weight=weight,
            weight_decay=weight_decay,
            **options,
        )
        self.project = project
        self.project_only_at_inference = project_only_at_inference

    def full_readout(self, h):
        if self.project_only_at_inference and self.training:
            return self.readout_map(h)
        return self.project(self.readout_map(h))


class NJODE_vertex_approach(NJODE):
    """
    NJ-ODE model with vertex approach. Inherits from normal NJODE
    """

    def __init__(  # initialize the class by naming relevant features
        self,
        input_size,
        hidden_size,
        output_size,
        ode_nn,
        readout_nn,
        enc_nn,
        use_rnn,
        vertices,
        bias=True,
        dropout_rate=0,
        solver="euler",
        weight=0.5,
        weight_decay=1.0,
        **options,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=len(vertices),
            ode_nn=ode_nn,
            readout_nn=readout_nn,
            enc_nn=enc_nn,
            use_rnn=use_rnn,
            in_shape_func=lambda l: [
                True for _ in l
            ],  # small hack (vertex approach by definition always inside Q)
            penalising_func=zero_pen_func,
            lmbda=0,
            bias=bias,
            dropout_rate=dropout_rate,
            solver=solver,
            weight=weight,
            weight_decay=weight_decay,
            **options,
        )
        self.vertices = vertices  # tensor shape (1, num, d_X)

        assert vertices.shape[1] == input_size

        # Hack to do the final projection/application of convex combination
        old_readout_forward = self.readout_map.forward

        def new_readout_forward(x):
            raw_out = old_readout_forward(x)
            weights = torch.nn.Softmax(dim=1)(raw_out)
            as_point = torch.matmul(weights, self.vertices)

            return as_point

        self.readout_map.forward = new_readout_forward
