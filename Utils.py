import os
import pickle
import pdb
import math
import random
import numpy as np
from numpy import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer.Constants import PAD
from Constants import ArchitectureChoice, AttackRegularizerChoice, BlackBoxSubtype, NoiseModelChoice, ThreatModel

from process import get_dataloader
from transformer.Models import get_non_pad_mask

from loguru import logger

DATA_PATH = os.environ.get("DATA_PATH", "./data")


# XXX: Handle defender models other than THP
def init_defender_models(tpp_model_class, opt, num_types):
    def _get_pickle(path):
        path_pieces = path.split('/')[:-1]
        parent_path = "/".join(path_pieces)

        with open(os.path.join(parent_path, 'config.pkl'), 'rb') as f:
            opt_ret = pickle.load(f)

        return opt_ret

    tpp_model_src = None

    if opt.threat_model in [ThreatModel.WHITE_BOX_SOURCE, ThreatModel.BLACK_BOX]:
        opt_tgt = _get_pickle(opt.defender_tgt_path)
        opt_src = _get_pickle(opt.defender_src_path)

        if opt.arch == ArchitectureChoice.THP:
            tpp_model_tgt = tpp_model_class(
                num_types=num_types,
                d_model=opt_tgt['d_model'],
                d_rnn=opt_tgt['d_rnn'],
                d_inner=opt_tgt['d_inner_hid'],
                n_layers=opt_tgt['n_layers'],
                n_head=opt_tgt['n_head'],
                d_k=opt_tgt['d_k'],
                d_v=opt_tgt['d_v'],
                dropout=opt_tgt['dropout'],
            )
            tpp_model_src = tpp_model_class(
                num_types=num_types,
                d_model=opt_src['d_model'],
                d_rnn=opt_src['d_rnn'],
                d_inner=opt_src['d_inner_hid'],
                n_layers=opt_src['n_layers'],
                n_head=opt_src['n_head'],
                d_k=opt_src['d_k'],
                d_v=opt_src['d_v'],
                dropout=opt_src['dropout'],
            )

        elif opt.arch == ArchitectureChoice.RNN_TPP:
            tpp_model_tgt = tpp_model_class(
                num_types=num_types,
                d_model=opt_tgt['d_model'],
                d_rnn=opt_tgt['d_rnn'],
                pad_max_len=opt.pad_max_len
            )
            tpp_model_src = tpp_model_class(
                num_types=num_types,
                d_model=opt_src['d_model'],
                d_rnn=opt_src['d_rnn'],
                pad_max_len=opt.pad_max_len
            )

        tpp_model_src.to(opt_src['device'])
        tpp_model_tgt.to(opt_tgt['device'])

        tpp_model_src.load_state_dict(torch.load(opt.defender_src_path))
        tpp_model_tgt.load_state_dict(torch.load(opt.defender_tgt_path))

        return tpp_model_tgt, tpp_model_src

    elif opt.threat_model == ThreatModel.WHITE_BOX:
        opt_tgt = _get_pickle(opt.defender_tgt_path)

        if opt.arch == ArchitectureChoice.THP:
            tpp_model_tgt = tpp_model_class(
                num_types=num_types,
                d_model=opt_tgt['d_model'],
                d_rnn=opt_tgt['d_rnn'],
                d_inner=opt_tgt['d_inner_hid'],
                n_layers=opt_tgt['n_layers'],
                n_head=opt_tgt['n_head'],
                d_k=opt_tgt['d_k'],
                d_v=opt_tgt['d_v'],
                dropout=opt_tgt['dropout'],
            )

        elif opt.arch == ArchitectureChoice.RNN_TPP:
            tpp_model_tgt = tpp_model_class(
                num_types=num_types,
                d_model=opt_tgt['d_model'],
                d_rnn=opt_tgt['d_rnn'],
                pad_max_len=opt.pad_max_len
            )

        tpp_model_tgt.to(opt_tgt['device'])
        tpp_model_tgt.load_state_dict(torch.load(opt.defender_tgt_path))

        return tpp_model_tgt, tpp_model_src


def init_OUR_modules(opt, num_types, min_ie_time):
    # avoid circular imports
    from Models import MLP, NoiseGenerator, NoiseTransformerV2, AdversarialGenerator,\
        NoiseRNN, SparseLayer

    if opt.attack_reg in [AttackRegularizerChoice.KLDIV, AttackRegularizerChoice.KLDIV_BETA]:
        assert opt.kl_beta <= opt.kl_alpha

    gphi_mlp = MLP(2 * opt.d_gphi, [2 * opt.d_gphi, 2 * opt.d_gphi, 1], opt.device)

    if opt.noise_model in [NoiseModelChoice.NOISE_RNN, NoiseModelChoice.NOISE_TRANSFORMER]:
        noise_gen = NoiseRNN(opt, num_types)

    elif opt.noise_model == NoiseModelChoice.NOISE_TRANSFORMER_V2:
        noise_gen = NoiseTransformerV2(opt, num_types)

    elif opt.noise_model == NoiseModelChoice.NOISE_SPARSE_NORMAL:
        noise_gen = SparseLayer(
            opt.pad_max_len,
            opt.sparse_hidden,
            opt.kappa,
            opt.device,
            sparse_range=opt.sparse_range
        )

    else:
        noise_gen = NoiseGenerator(opt.pad_max_len, [opt.d_gphi, opt.d_gphi, opt.pad_max_len],
                                   opt.device, min_ie_time, opt.noise_pow_param, opt.noise_act)

    adv_model = AdversarialGenerator(noise_gen, gphi_mlp, opt.gs_iters, opt.gs_tau, min_ie_time,
                                     opt.noise_model, opt.device)
    adv_optimizer = optim.Adam(filter(lambda x: x.requires_grad, adv_model.parameters()),
                               opt.adv_lr)

    adv_model.apply(adv_model.init_weights)

    return {
        "adv_model": adv_model,
        "adv_optimizer": adv_optimizer,
    }


def init_sparsenet_modules(opt):
    from Models import SparseLayer
    sparse_net = SparseLayer(
        opt.pad_max_len,
        opt.sparse_hidden,
        opt.kappa,
        opt.device
    )
    sparse_optim = optim.Adam(sparse_net.parameters(), lr=opt.lr)

    return {
        "adv_model": sparse_net,
        "adv_optimizer": sparse_optim
    }


def set_seed(seed):
    """
    Set random seed for all random generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # set the seed for the current GPU
    torch.cuda.manual_seed_all(seed) # set the seed for all the GPUs
    random.seed(seed)
    np.random.seed(seed)

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def compute_event(event, non_pad_mask):
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result

def compute_integral_biased(all_lambda, time, non_pad_mask):
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result

def _compute_intensity(model, data, time, non_pad_mask=None, type_mask=None,
                       types=None):
    if non_pad_mask is None:
        non_pad_mask = get_non_pad_mask(time).squeeze(2)

    if type_mask is None:
        type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
        for i in range(model.num_types):
            type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    num_samples = 100
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    return all_lambda, diff_time

def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    all_lambda, diff_time = _compute_intensity(model, data, time,
        non_pad_mask=non_pad_mask, type_mask=type_mask)

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral

def log_likelihood(model, data, time, types):
    non_pad_mask = get_non_pad_mask(types).squeeze(2)
    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll

def _markseq_pred(prediction, types):
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]
    pred_type = torch.max(prediction, dim=-1)[1]

    return prediction, truth, pred_type

def type_loss(prediction, types, loss_func):
    prediction, truth, pred_type = _markseq_pred(prediction, types)
    correct_num = torch.sum(pred_type == truth)

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num

def get_mpa_std_error(prediction, types, norm_constant):
    # The inputs contain the complete val/test set, and not batches of it.
    # So the std error computation is across all events of all sequences, i.e.,
    # all events in the test set.
    prediction, truth, pred_type = _markseq_pred(prediction, types)
    correct_preds_per_seq = torch.sum(pred_type == truth, dim=-1)

    std_dev = torch.std(correct_preds_per_seq.float())
    std_error = std_dev / torch.sqrt(norm_constant)
    return std_error

def _timeseq_diff(prediction, event_time):
    prediction.squeeze_(-1)
    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    diff =  true - prediction
    return diff, prediction, true

def time_loss(prediction, event_time, mse=False):
    diff, prediction, true = _timeseq_diff(prediction, event_time)

    if mse:
        se = F.mse_loss(prediction, true, reduction='sum')
    else:
        se = torch.sum(torch.abs(diff))
    return se

def get_mae_std_error(prediction, event_time, norm_constant):
    # The inputs contain the complete val/test set, and not batches of it.
    # So the std error computation is across all events of all sequences, i.e.,
    # all events in the test set.
    seq_diffs, _, _ = _timeseq_diff(prediction, event_time)
    seq_diff = torch.abs(seq_diffs)
    std_dev = torch.std(seq_diff)
    std_error = std_dev / torch.sqrt(norm_constant)

    return std_error

def prepare_dataloader(opt):
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    def process_data(data, standardize=False, scale=1.0):
        """
        If standardize set to True, min-max scale the event time data.
        Else just return the min inter-event time and max sequence length.
        """
        min_ie_time = inf
        max_ie_time = 0
        max_len = 0
        standard_data = []

        for seq in data:
            min_here = min([x['time_since_last_event'] for x in seq])
            max_here = max([x['time_since_last_event'] for x in seq])
            max_len_here = len(seq)

            if min_here < min_ie_time:
                min_ie_time = min_here

            if max_here > max_ie_time:
                max_ie_time = max_here

            if max_len_here > max_len:
                max_len = max_len_here

        for seq in data:
            if standardize:
                new_seq = []

                for x in seq:
                    std_time_since_start = (
                        scale * (x['time_since_start'] - min_ie_time) / (max_ie_time - min_ie_time)
                    )
                    new_seq.append({
                        "time_since_last_event": x['time_since_last_event'],
                        "type_event": x['type_event'],
                        "time_since_start": std_time_since_start
                    })

                standard_data.append(new_seq)

            else:
                standard_data.append(seq)

        return min_ie_time, max_len, standard_data

    print('Loading All Datasets...')
    # XXX: Pickle names are hardcoded for now
    file_path = os.path.join(os.path.join(DATA_PATH, opt.data), f"fold{str(opt.fold)}")
    train_data, num_types = load_data(os.path.join(file_path, 'train.pkl'), 'train')
    try:
        val_data, _ = load_data(os.path.join(file_path, 'val.pkl'), 'dev')
    except FileNotFoundError as e:
        val_data, _ = load_data(os.path.join(file_path, 'dev.pkl'), 'dev')

    test_data, _ = load_data(os.path.join(file_path, 'test.pkl'), 'test')

    # Find smallest inter-event time in training data.
    min_ie_time, max_len, train_data = \
        process_data(train_data, standardize=opt.standardize, scale=opt.std_scale)
    val_min_ie_time, val_max_len, val_data = \
        process_data(val_data, standardize=opt.standardize, scale=opt.std_scale)
    test_min_ie_time, test_max_len, test_data = \
        process_data(test_data, standardize=opt.standardize, scale=opt.std_scale)

    logger.info(f"Number of types: {num_types}")
    logger.info(f"Train: min inter-event time in training data is {min_ie_time}")
    logger.info(f"Train: Max length of sequence is {max_len}")
    logger.info(f"Test: min inter-event time in training data is {test_min_ie_time}")
    logger.info(f"Test: Max length of sequence is {test_max_len}")

    if opt.bb_subtype == BlackBoxSubtype.TRAINSET:
        # Target trainset = 75% full trainset
        # Source trainset = 75% full trainset
        lenn = len(train_data)
        seventy_five = int(0.75 * lenn)

        first_indices = np.random.choice(list(range(lenn)), size=seventy_five, replace=False)
        second_indices = np.random.choice(list(range(lenn)), size=seventy_five, replace=False)
        train_obj = np.array(train_data, dtype=object)

        first_part = train_obj[first_indices].tolist()
        sec_part = train_obj[second_indices].tolist()

        trainloader_tgt = get_dataloader(first_part, opt.batch_size, opt.pad_max_len, shuffle=False)
        trainloader_src = get_dataloader(sec_part, opt.batch_size, opt.pad_max_len, shuffle=False)

    else:
        trainloader_tgt = get_dataloader(train_data, opt.batch_size, opt.pad_max_len, shuffle=False)
        trainloader_src = get_dataloader(train_data, opt.batch_size, opt.pad_max_len, shuffle=False)

    valloader = get_dataloader(val_data, opt.batch_size, opt.pad_max_len, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, opt.pad_max_len, shuffle=False)
    return trainloader_tgt, trainloader_src, valloader, testloader, num_types, min_ie_time, max_len

def freeze_network_weights(network):
    for param in network.parameters():
        param.requires_grad = False

def unfreeze_network_weights(network):
    for param in network.parameters():
        param.requires_grad = True

def hellinger_distance(p, q):
    # Both inputs assumed to be log probabilities.
    _SQRT2 = np.sqrt(2)

    return torch.sqrt(torch.sum(torch.sqrt(torch.exp(p)) - torch.sqrt(torch.exp(q))) ** 2) / _SQRT2

def sort_non_padding(input_data):
    """
    Sorts a batch of event data (time or mark) while excluding the padding values.
    """
    input_data_pre = torch.sort(input_data)[0]
    input_data_srt_indx = torch.sort((input_data_pre != PAD) * 1, dim=-1, descending=True)[1]
    input_data_srt = input_data_pre.gather(1, input_data_srt_indx)

    return input_data_srt

def min_non_padding(input_data, non_pad_mask, max_vals):
    """
    Finds the min. of each sequence in a batch of sequences while excluding padding values.
    """
    tp = input_data.clone()
    nm = max_vals * (1 - non_pad_mask)
    tp = tp + nm
    return torch.min(tp, dim=-1)[0].unsqueeze(-1)

def sequence_extremes(input_data, non_pad_mask, max_factor, min_factor):
    noise_max = max_factor * torch.max(input_data, dim=-1)[0].unsqueeze(-1)
    noise_min = non_pad_mask * min_factor * min_non_padding(input_data, non_pad_mask, noise_max)
    noise_max = noise_max * non_pad_mask

    return noise_min, noise_max

def wasserstein_distance(event_time_adv, event_time_clean, event_type_adv=None, event_type_clean=None,
                         shift_by_min=False):
    """
    Compute the Wasserstein distance between time sequences, and if provided,
    between mark sequences. Sort the adversarial time sequence while taking into account
    the padding values, before calculating the difference. For mark sequences,
    just check how many positions the pair of sequences don't match.
    """
    time_W, mark_W, curr_W = 0, 0, 0
    event_time_adv_srt = sort_non_padding(event_time_adv)
    ev_min = event_time_adv_srt[:, 0].unsqueeze(-1)
    ev_min_clean = event_time_clean[:, 0].unsqueeze(-1)

    if shift_by_min:
        non_pad_mask = get_non_pad_mask(event_time_clean).squeeze(-1)
        time_W = torch.sum(torch.abs(
            ((event_time_adv_srt - ev_min) - (event_time_clean - ev_min_clean)) * non_pad_mask
        ))

    else:
        time_W = torch.sum(torch.abs(event_time_adv_srt - event_time_clean))

    curr_W += time_W

    if event_type_adv is not None and event_type_clean is not None:
        mark_W += torch.sum(torch.round(event_type_adv).long() != event_type_clean)
        curr_W += mark_W

    return curr_W, mark_W, time_W

def serialize_epoch_stats(attack_train_acc_list, attack_train_mae_list,
                          test_acc_list, test_mae_list,
                          robust_test_acc_list, robust_test_mae_list):
    return {
        "attack_metrics": {
            "attack_train_acc_list": attack_train_acc_list,
            "attack_train_mae_list": attack_train_mae_list
        },
        "defense_metrics": {
            "test_acc_list": test_acc_list,
            "test_mae_list": test_mae_list,
            "robust_test_acc_list": robust_test_acc_list,
            "robust_test_mae_list": robust_test_mae_list
        }
    }

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        non_pad_mask = target.ne(self.ignore_index).float()
        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
