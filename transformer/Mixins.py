import functools
import operator
import torch

import Utils


class LoglikeLossMixin(object):
    def loglike_loss(self, enc_out, event_time, event_type, prediction, pred_loss_func):
        """
        Follows a similar API as the EasyTPP repository of TPP implementations.
        """
        event_ll, non_event_ll = Utils.log_likelihood(self, enc_out, event_time,
                                                      event_type)
        log_ll = event_ll - non_event_ll
        event_loss = torch.sum(log_ll)
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)
        se = Utils.time_loss(prediction[1], event_time)
        mse = Utils.time_loss(prediction[1], event_time, mse=True)

        return {
            "nll": -event_loss,
            "pred_loss": pred_loss,
            "pred_num_event": pred_num_event,
            "se": se,
            "mse": mse
        }


class TemporalEncMixin(object):
    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        # Make vectors of ones and zeros in odd vs even positions
        # of the size of the total shape of `result`.
        # Reshape into the shape of `result` to use them as masks.
        dev = result.device

        shape_total = functools.reduce(operator.mul, result.shape)
        even_ones = [1 if (i % 2) == 0 else 0 for i in range(shape_total)]
        odd_ones = [0 if (i % 2) == 0 else 1 for i in range(shape_total)]

        even_ones = torch.tensor(even_ones, device=dev).reshape(result.shape)
        odd_ones = torch.tensor(odd_ones, device=dev).reshape(result.shape)

        # Replaces the earlier in-place computation where even positions are set to
        # sin() and odd positions are set to cos().
        # Required as in-place operations cause problems during backprop wrt gradients.
        # Tradeoff: Training time more than doubles from ~12 mins to 30 mins per attack iter.
        result = odd_ones * result + even_ones * torch.sin(result)
        result = even_ones * result + odd_ones * torch.cos(result)
        return result * non_pad_mask