import torch

from transformer.Constants import PAD


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(len_q, padding_mask_k):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    padding_mask = padding_mask_k.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(sz_b, len_s, device):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask