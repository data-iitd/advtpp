import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gs import gumbel_sinkhorn
from Constants import NoiseModelChoice, NoiseActivationFunction
from transformer.Models import get_non_pad_mask, Transformer
import transformer.Constants as TConst
import Utils


class MLP(nn.Module):
    def __init__(self, in_features, hidden_layers, device,
                 activation=NoiseActivationFunction.RELU):

        super().__init__()
        self.device = device
        self.layers = [in_features] + hidden_layers
        self.net = []

        act_func = None
        if activation == NoiseActivationFunction.TANH:
            act_func = nn.Tanh
        elif activation == NoiseActivationFunction.RELU:
            act_func = nn.ReLU
        elif activation == NoiseActivationFunction.LEAKY_RELU:
            act_func = nn.LeakyReLU

        for i in range(1, len(self.layers)):
            prev_layer, layer = self.layers[i - 1:i + 1]
            self.net.extend([
                nn.Linear(prev_layer, layer),
                act_func(),
            ])
        self.net.pop()  
        self.net = nn.Sequential(*self.net).to(device)
         
    def forward(self, x):
        x = x.to(self.device)
        x = self.net(x)

        return x


class NoiseGenerator(MLP):
    """
    Extension of MLP that outputs real values in the range [-min_ie_time, min_ie_time].
    Trained in a manner so that the outputs follow the constraint -:
    t_1 + \epsilon_1 < t_2 + \epsilon2 + ...
    where t_i are components of the (permuted) input and \epsilon_i is the network output.
    XXX: UNUSED
    """
    def __init__(self, in_features, hidden_layers, device, min_ie_time, noise_pow_param,
                 noise_act):
        super().__init__(in_features, hidden_layers, device, activation=noise_act)
        self.min_ie_time = min_ie_time
        self.noise_pow_param = noise_pow_param
        self.noise_act = noise_act

    def forward(self, x):
        x = super().forward(x)

        # Division by e^(param) helps control how quickly the output of the
        # neural net falls into our desired range. Set to zero if not required.
        x = x / torch.exp(torch.tensor([self.noise_pow_param])).squeeze()
        # Note: We were capping the value of x earlier to 1e-8 but this
        # affects attack training (causes fluctuation of train accuracy
        # in a plateau). So a better approach is to stop training at the
        # epoch where the values start to approach very small precision/nan.

        return x


class NoiseRNN(nn.Module):
    """
    Sequential model that outputs noise constrained so that order of the input
    is maintained if noise is added to it.
    """
    def __init__(self, opt, num_types):
        super().__init__()
        self.device = opt.device
        self.noise_model_choice = opt.noise_model

        if opt.noise_model == NoiseModelChoice.NOISE_RNN:
            self.noise_rnn = nn.RNN(input_size=1,
                                    hidden_size=opt.d_noise_rnn,
                                    batch_first=True,
                                    num_layers=opt.noise_rnn_layers,
                                    bidirectional=False, nonlinearity='relu')\
                .to(self.device)
            self.time_linear = nn.Linear(in_features=opt.d_noise_rnn, out_features=1, bias=True)\
                .to(self.device)

        elif opt.noise_model == NoiseModelChoice.NOISE_TRANSFORMER:
            self.noise_transformer = Transformer(
                num_types=num_types,
                d_model=opt.d_model,
                d_rnn=opt.d_rnn,
                d_inner=opt.d_inner_hid,
                n_layers=opt.n_layers,
                n_head=opt.n_head,
                d_k=opt.d_k,
                d_v=opt.d_v,
                dropout=opt.dropout,
            ).to(self.device)
            self.time_linear = nn.Linear(in_features=opt.d_model, out_features=1, bias=True)\
                .to(self.device)

        if opt.noise_act == NoiseActivationFunction.TANH:
            self.nonlinearity = nn.Tanh()
        elif opt.noise_act == NoiseActivationFunction.RELU:
            self.nonlinearity = nn.ReLU()
        elif opt.noise_act == NoiseActivationFunction.LEAKY_RELU:
            self.nonlinearity = nn.LeakyReLU()

        self.noise_weight = nn.Linear(in_features=opt.pad_max_len, out_features=opt.pad_max_len)\
            .to(self.device)

    def forward(self, time_input, type_input, eps_min=1e-5):
        time_input = time_input.to(self.device)
        time_input_rev = time_input.flip(1)

        if self.noise_model_choice == NoiseModelChoice.NOISE_RNN:
            output, hidden_states = self.noise_rnn(time_input_rev_unsqueezed)
        elif self.noise_model_choice == NoiseModelChoice.NOISE_TRANSFORMER:
            output, _ = self.noise_transformer(type_input, time_input)

        eps_net = self.time_linear(output).squeeze(-1)
        # 1e-5 to prevent unintentional zero values.
        # Tanh will allow for negative values as well.
        eps_net = self.nonlinearity(eps_net) + eps_min
        eps_net = eps_net * get_non_pad_mask(time_input).squeeze(-1)
        eps_net_rev = eps_net.flip(1)

        # Shift by 1 and then replace 0th column with 1st.
        eps_net_rev_shifted = torch.roll(eps_net_rev, 1)
        eps_net_rev_shifted[:, 0] = eps_net_rev_shifted[:, 1]
        time_input_rev_shifted = torch.roll(time_input_rev, 1)
        time_input_rev_shifted[:, 0] = time_input_rev_shifted[:, 1]

        noise_diff = eps_net_rev_shifted - eps_net_rev
        time_delta = time_input_rev - time_input_rev_shifted
        hinge_term = nn.functional.relu(time_delta - self.noise_weight(noise_diff))
        positive_hinge = nn.functional.relu(eps_min - (time_input + eps_net))

        return eps_net, noise_diff, hinge_term, positive_hinge


class NoiseTransformerV2(nn.Module):
    """
    Sequential model that outputs noise constrained so that order of the input
    is maintained if noise is added to it.

    In addition, if sparse mode is activated, it is similar to SparseLayer in
    that it zeroes out noise values other than the top-K noise values, permutes
    them.
    """
    def __init__(self, opt, num_types):
        super().__init__()
        self.device = opt.device

        self.noise_transformer = Transformer(
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
        ).to(self.device)
        self.time_linear = nn.Linear(in_features=opt.d_model, out_features=1, bias=True)\
            .to(self.device)
        self.bn = nn.BatchNorm1d(opt.pad_max_len).to(self.device)
        self.norm = opt.batch_norm
        self.sparse_mode = opt.sparse_mode

        self.noise_weight = nn.Linear(in_features=opt.pad_max_len, out_features=opt.pad_max_len)\
            .to(self.device)
        self.kappa = opt.kappa
        self.min_factor = opt.min_factor
        self.max_factor = opt.max_factor

    def forward(self, time_input, type_input, noise_perms=None):
        time_input = time_input.to(self.device)
        time_input_rev = time_input.flip(1)
        non_pad_mask = get_non_pad_mask(time_input).squeeze(-1)
        rev_non_pad_mask = non_pad_mask.flip(1)

        output, _ = self.noise_transformer(type_input, time_input)
        eps_net = self.time_linear(output).squeeze(-1)

        if self.norm:
            eps_net = self.bn(eps_net)

        eps_net = eps_net * non_pad_mask

        if self.sparse_mode:
            # Permute
            eps_net = torch.bmm(eps_net.unsqueeze(1).float(), noise_perms).squeeze(1)
            # Retain only top-K values. Zero out the others.
            kappa = int((self.kappa/100) * eps_net.shape[1])
            zero_mask = torch.zeros_like(eps_net)
            op_desc_idx = torch.sort(eps_net, descending=True)[1]

            zero_mask[torch.arange(zero_mask.size(0)).unsqueeze(1), op_desc_idx[:, :kappa]] = 1
            eps_net = eps_net * zero_mask

        # Shift by 1 and then replace 0th column with 1st.
        eps_net_rev = eps_net.flip(1)
        eps_net_rev_shifted = torch.roll(eps_net_rev, 1)
        eps_net_rev_shifted[:, 0] = eps_net_rev_shifted[:, 1]
        time_input_rev_shifted = torch.roll(time_input_rev, 1)
        time_input_rev_shifted[:, 0] = time_input_rev_shifted[:, 1]

        noise_diff = eps_net_rev_shifted - eps_net_rev
        time_delta = time_input_rev - time_input_rev_shifted
        noise_weighted = self.noise_weight(noise_diff)

        noise_max = self.max_factor * torch.max(time_input, dim=-1)[0].unsqueeze(-1)
        # To find the minimum, we need to disregard padding values.
        # This we will do by adding the max value in the padding positions.
        noise_min = self.min_factor * Utils.min_non_padding(time_input, non_pad_mask, noise_max)

        hinge_term = nn.functional.relu(time_delta - rev_non_pad_mask * noise_weighted)
        min_hinge = nn.functional.relu((noise_min - (time_input + eps_net)) * non_pad_mask)
        max_hinge = nn.functional.relu(((time_input + eps_net) - noise_max) * non_pad_mask)

        return eps_net, noise_diff, hinge_term, min_hinge, max_hinge


class SparseLayer(nn.Module):
    """
    Re-implementation of SparseLayer from the paper -:
    "Robust Multivariate Time-Series Forecasting: Adversarial Attacks And Defense Mechanisms"
    for the case of continuous time event sequences.

    We learn the parameters of a normal distribution from which we will sample noise vectors.
    The noise vectors will be sparsified similar to the deterministic attack, depending on
    the value of kappa.
    """
    def __init__(self, input_dim, hidden_dim, kappa, device, sparse_range=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kappa = kappa
        self.device = device
        self.sparse_range = sparse_range

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(device)
        self.fc21 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        self.fc22 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        self.bn = nn.BatchNorm1d(self.hidden_dim).to(device)

    def forward(self, batch, n_sample=100, norm=False):
        input_fc = self.fc1(batch)
        if norm:
            input_fc = F.relu(self.bn(input_fc))
        else:
            input_fc = F.relu(input_fc)

        if self.sparse_range is not None:
            mu = self.fc21(input_fc)
            std = math.sqrt(self.sparse_range)

        else:
            mu, log_var = self.fc21(input_fc), self.fc22(input_fc)
            std = torch.exp(0.5 * log_var)

        eps = (
            torch.empty(n_sample, mu.shape[0], self.input_dim)
            .normal_(0, 1)
            .to(self.device)
        )
        op = mu + eps * std
        op = op.mean(0)
        op = op * get_non_pad_mask(batch).squeeze(-1)

        # Sorting and masking
        kappa = int((self.kappa/100) * op.shape[1])
        mask = torch.zeros_like(op)
        op_desc_idx = torch.sort(op, descending=True)[1]

        mask[torch.arange(mask.size(0)).unsqueeze(1), op_desc_idx[:, :kappa]] = 1
        op = op * mask

        return op


class AdversarialGenerator(nn.Module):
    """
    Generate Gumbel-Sinkhorn permutation matrix $P$ and
    noise vector $\epsilon$.
    """
    def __init__(self, noise_generator, gphi_mlp, gs_iters, gs_tau, min_ie_time,
                 noise_model_choice, device, sparse_mode=False,
                 same_perm_matrix=False):
        super().__init__()

        self.device = device
        self.gphi_mlp = gphi_mlp
        self.noise_generator = noise_generator
        self.gs_iters = gs_iters
        self.gs_tau = gs_tau
        self.noise_model_choice = noise_model_choice
        self.min_ie_time = min_ie_time
        self.sparse_mode = sparse_mode
        self.same_perm_matrix = same_perm_matrix

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)

            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def apply_gphi_network(self, batch):
        """
        Returns both the concatenated input that is fed to the MLP before a
        Gumbel-Sinkhorn transformation is applied, as well as the output of the MLP.
        Takes a batched input of size batch_size x max_padded_length x thp_hidden_dimension.
        The input contains a batch of event marks.
        """
        a_ntimes = torch.tile(batch.unsqueeze(1), (1, batch.shape[1],1,1))
        a_ntimes_t = torch.transpose(a_ntimes, 1, 2)
        concatenated_input = torch.cat((a_ntimes_t, a_ntimes), dim=3)

        return concatenated_input, self.gphi_mlp(concatenated_input).squeeze(-1)

    def generate_perm_matrices(self, clean_enc_out, gumbel_masks=None):
        """
        Generates permutation matrices using the output of the gphi network. The
        gumbel-sinkhorn operator is applied for this purpose.
        """
        _, gphi_op = self.apply_gphi_network(clean_enc_out)
        perm_mats = gumbel_sinkhorn(gphi_op, self.gs_tau, self.gs_iters,
            gumbel_masks=gumbel_masks)

        return perm_mats

    def forward(self, batch, clean_enc_out, no_time_noise=False):
        """
        Takes as input -:
        * a batch of clean input data, consisting of event types/marks and event times.
        * a batch of desired model outputs on clean input data.

        Returns -:
        * A batch of event types permuted via GS transformation
        * A batch of event times permuted by GS transformation and then addled by noise
        * The permutation matrices for both types and times, and the noise vectors assoc.
        with the times.
        """
        event_time, time_gap, event_type = map(lambda x: x.to(self.device), batch)

        nonzero_entries = get_non_pad_mask(event_time).squeeze(-1).sum(dim=-1).long()
        pad_max_len = event_time.shape[1]
        # A list of pad_max_len x pad_max_len tensors, where the top left size x size submatrix
        # contains all-1s, while the remaining elements are all-0s.
        # These will be used to mask during the Gumbel-Sinkhorn normalization procedure.
        # This solves the problem of permuting only the non-padded part of the sequence while leaving
        # out the padded entries at the end.
        if not hasattr(self, "mask_map"):
            self.mask_map = torch.stack([
                torch.cat((
                    torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([size, pad_max_len - size])).repeat(size, 1),
                    torch.repeat_interleave(torch.tensor([1,0]), torch.tensor([0, pad_max_len])).repeat(pad_max_len - size,1)))
                    for size in range(1, pad_max_len + 1)
            ]).to(self.device)
        gumbel_masks = self.mask_map[nonzero_entries - 1] == 0

        event_type_perms = self.generate_perm_matrices(clean_enc_out, gumbel_masks=gumbel_masks)
        event_types_permed = torch.bmm(event_type.unsqueeze(1).float(), event_type_perms).squeeze(1)

        noise_diff, hinge_term, event_time_perms, noise_vecs = None, None, None, None
        min_hinge, max_hinge = None, None
        extras = {}

        if self.noise_model_choice == NoiseModelChoice.UNIFORM_NOISE:
            # Bypass the model and instead sample from the uniform distribution.
            noise_vecs = (-self.min_ie_time - self.min_ie_time) * \
                torch.rand(event_time.shape, requires_grad=True) + self.min_ie_time
            noise_vecs = noise_vecs.to(self.device)
            event_time_noisy = event_time + noise_vecs

        else:
            if self.sparse_mode:
                noise_perms = self.generate_perm_matrices(clean_enc_out, gumbel_masks=gumbel_masks)
            else:
                noise_perms = None

            if self.same_perm_matrix:
                event_time_perms = event_type_perms
            else:
                event_time_perms = self.generate_perm_matrices(clean_enc_out, gumbel_masks=gumbel_masks)

            event_times_permed = torch.bmm(event_time.unsqueeze(1), event_time_perms).squeeze(1)

            if no_time_noise:
                event_time_noisy = event_times_permed
            else:
                noise_vecs, noise_diff, hinge_term, min_hinge, max_hinge = \
                    self.noise_generator.forward(event_times_permed, event_types_permed,
                                                 noise_perms=noise_perms)
                event_time_noisy = event_times_permed + noise_vecs

            extras.update({
                "noise_perms": noise_perms
            })

        extras.update({
            "event_type_perms": event_type_perms,
            "event_time_perms": event_time_perms,
            "event_types_permed": event_types_permed,
            "noise_vecs": noise_vecs,
            "noise_diff": noise_diff,
            "hinge_term": hinge_term,
            "min_hinge": min_hinge,
            "max_hinge": max_hinge,
            "gumbel_masks": gumbel_masks
        })

        return event_types_permed, event_time_noisy, extras
