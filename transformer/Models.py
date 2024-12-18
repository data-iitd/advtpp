import math
import torch
import torch.nn as nn

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from transformer.Masks import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask
import transformer.Mixins as Mixins


class Encoder(nn.Module, Mixins.TemporalEncMixin):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def forward(self, event_type, event_time, non_pad_mask, remove_sin_cos=False):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type.shape[0], event_type.shape[1],
                                                   event_type.device)
        slf_attn_mask_keypad = get_attn_key_pad_mask(event_type.shape[1],
            ~(non_pad_mask.squeeze().bool()))
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        if remove_sin_cos:
            # Hack to deal with strange gradient backprop issue where setting
            # cos embeddings inplace kills off gradients.
            # The problem does not happen on solitary/non-adversarial training
            # of the THP.
            tem_enc = (event_time.unsqueeze(-1) / self.position_vec) * non_pad_mask
        else:
            tem_enc = self.temporal_enc(event_time, non_pad_mask)

        # torch.embedding layers should be the first layer in a model/sub-model
        # as they take only integer/long inputs and are non-differentiable.
        enc_output = self.event_emb(event_type.long())

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out

class LG(nn.Module):

    def __init__(self):
        super().__init__()
        self.lognorm = torch.distributions.log_normal.LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))

    def forward(self, data, non_pad_mask):
        x = self.lognorm.sample().to("cuda:0")
        out = data * x
        out = out * non_pad_mask
        return out

class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, pad_max_len=None):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)
        self.pad_max_len = pad_max_len

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True,
            total_length=self.pad_max_len)[0]

        out = self.projection(out)
        return out

class RNN_TPP(nn.Module, Mixins.TemporalEncMixin, Mixins.LoglikeLossMixin):
    def __init__(self, num_types, d_model=256, d_rnn=128, pad_max_len=None):
        super().__init__()
        self.num_types = num_types
        self.rnn_layer = RNN_layers(d_model, d_rnn, pad_max_len=pad_max_len)

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0)  )

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

    def forward(self, event_type, event_time, remove_sin_cos=False):
        non_pad_mask = get_non_pad_mask(event_type)
        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        # torch.embedding layers should be the first layer in a model/sub-model
        # as they take only integer/long inputs and are non-differentiable.
        emb = self.event_emb(event_type.long())
        emb = tem_enc + emb

        rnn_output = self.rnn_layer(emb, non_pad_mask)
        rnn_output = rnn_output * non_pad_mask

        time_prediction = self.time_predictor(rnn_output, non_pad_mask)
        type_prediction = self.type_predictor(rnn_output, non_pad_mask)

        return rnn_output, (type_prediction, time_prediction)

class Transformer(nn.Module, Mixins.LoglikeLossMixin):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # Adding log-normal 
        self.time_log = LG()

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time, remove_sin_cos=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask,
                                  remove_sin_cos=remove_sin_cos)
        # enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        # Uncomment to start log-normal distribution sampling
        # time_prediction = self.time_log(time_prediction, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)
