import random
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Constants import ThreatModel, TrainTimeAttack, OperationMode
from tqdm import tqdm
import transformer.Constants as TConst
from transformer.Models import get_non_pad_mask
import Utils

from loguru import logger


class PGD(object):
    def __init__(self, defender_model, test_data, pred_loss_func, opt,
                 epsilon=0.031, num_steps=20, step_size=0.0003, momentum_decay_mu=0,
                 sort_sequence=False, defender_model_src=None, to_print=True):
        super().__init__()

        self.defender_model = defender_model
        self.defender_model_src = defender_model_src
        self.test_data = test_data
        self.pred_loss_func = pred_loss_func
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.momentum_decay_mu = momentum_decay_mu
        self.sort_sequence = sort_sequence
        self.to_print = to_print
        self.opt = opt

    def get_random_inits(self, data, non_pad_mask):
        random_noise = torch.FloatTensor(*data.shape).uniform_(0, self.epsilon).to(self.opt.device)
        random_noise = random_noise * non_pad_mask
        adv_var = Variable(data + random_noise, requires_grad=True)

        return adv_var

    def get_pgd_loss(self, event_type_adv, event_time_adv, event_type, event_time):
        if self.opt.threat_model in [ThreatModel.WHITE_BOX, ThreatModel.WHITE_BOX_SOURCE]:
            def_model = self.defender_model
        elif self.opt.threat_model == ThreatModel.BLACK_BOX:
            def_model = self.defender_model_src

        adv_enc_out, adv_pred = def_model(event_type_adv, event_time_adv)
        loss_dict = self.defender_model.loglike_loss(adv_enc_out, event_time, event_type,
            adv_pred, self.pred_loss_func)
        return loss_dict['pred_loss'] + loss_dict['se'] / self.opt.se_time_scale\
            + loss_dict['nll']

    def projection(self, data_adv, data, eta_var, non_pad_mask):
        eta_var = eta_var * non_pad_mask
        data_adv = Variable(data_adv + eta_var, requires_grad=True)
        eta_var = torch.clamp(data_adv - data, -self.epsilon, self.epsilon)
        return Variable(data + eta_var, requires_grad=True), eta_var

    def attack(self):
        """
        Implementation for PGD, FGSM, MI-FGSM and TS-DET attacks.
        """
        self.defender_model.train()

        if self.defender_model_src is not None:
            self.defender_model_src.train()

        predicted_marks_full = []
        predicted_times_full = []
        clean_marks_full = []
        clean_times_full = []

        total_time_se = 0
        total_robust_time_se = 0
        total_event_rate = 0
        total_robust_event_rate = 0
        total_num_pred = 0
        total_num_event = 0
        std_error_sample_size = 0

        W_dist_main = 0
        W_dist_main_shifted = 0
        W_time = 0
        W_time_shifted = 0
        W_mark = 0
        emb_dist_main = 0

        adv_examples = []
        total_avg_Wt = 0
        total_avg_interr = 0

        total_time_taken = 0

        if self.opt.train_time_attack == TrainTimeAttack.FGSM:
            self.num_steps = 1

        for batch in tqdm(self.test_data, mininterval=2, desc=f'  - ({self.opt.train_time_attack} Attack WB)   ',
                          leave=False):
            start_time = time.time()
            event_time, time_gap, event_type = map(lambda x: x.to(self.opt.device), batch)
            non_pad_mask = get_non_pad_mask(event_time).squeeze(-1)

            clean_marks_full.append(event_type.detach().cpu())
            clean_times_full.append(event_time.detach().cpu())

            clean_enc_out, clean_pred = self.defender_model(event_type, event_time)
            clean_loss_dict = self.defender_model.loglike_loss(
                clean_enc_out, event_time, event_type, clean_pred, self.pred_loss_func)
            clean_pred_num = clean_loss_dict['pred_num_event']
            clean_se = clean_loss_dict['se']

            # Random init for PGD
            event_time_adv = self.get_random_inits(event_time, non_pad_mask)
            event_type_adv = event_type
            #event_type_adv = self.get_random_inits(event_type_ip, non_pad_mask_oh)

            # Momentum terms
            g_t_time = torch.zeros(*event_time_adv.shape).to(self.opt.device)
            #g_t_mark = torch.zeros(*event_type_adv.shape).to(self.opt.device)

            for _ in range(self.num_steps):
                # The lr/step size specified in the optimizer isn't actually used.
                # Original code just uses the opt_sgd to zero gradients. Weird.
                opt_sgd = optim.SGD([event_time_adv, event_type_adv], lr=1e-3)
                opt_sgd.zero_grad()

                with torch.enable_grad():
                    adv_pred_loss = self.get_pgd_loss(event_type_adv, event_time_adv,
                                                      event_type, event_time)
                adv_pred_loss.backward()

                if self.opt.train_time_attack == TrainTimeAttack.MI_FGSM:
                    # Momentum Iterative - Fast Gradient Sign Method (MI-FGSM)
                    g_t_time = self.momentum_decay_mu * g_t_time + \
                        (event_time_adv.grad / torch.linalg.vector_norm(event_time_adv.grad, ord=1, dim=-1)
                            .unsqueeze(-1))
                    eta_time = self.step_size * g_t_time.sign()

                    #event_type_norm = torch.linalg.norm(event_type_adv.grad, ord=1,
                    #    dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
                    #g_t_mark = self.momentum_decay_mu * g_t_mark + (event_type_adv.grad / event_type_norm)

                else:
                    # Simple PGD
                    eta_time = self.step_size * event_time_adv.grad.data.sign()
                    #eta_mark = self.step_size * event_type_adv.grad.data.sign()

                event_time_adv, eta_time = self.projection(event_time_adv, event_time, eta_time, non_pad_mask)
                event_type_adv = event_type
                #event_type_adv = self.projection(event_type_adv, event_type_ip, eta_mark, non_pad_mask_oh)

            if self.opt.train_time_attack == TrainTimeAttack.TS_DET:
                # XXX: Apply this to the time sequence only.
                # Save the top-kappa of the (descending) sorted noise vector.
                # Set the rest to zero. Meant to be used with extremely high noise.
                kappa = int((self.opt.kappa/100) * eta_time.shape[1])
                mask = torch.zeros_like(eta_time)
                eta_desc_idx = torch.sort(eta_time, descending=True)[1]

                # Set mask to 1 at those positions where we want to retain our
                # eta/noise values.
                # https://discuss.pytorch.org/t/fill-value-to-matrix-based-on-index/34698/3
                mask[torch.arange(mask.size(0)).unsqueeze(1), eta_desc_idx[:, :kappa]] = 1
                eta = eta_time * mask
                event_time_adv = Variable(event_time + eta, requires_grad=True)

            if self.sort_sequence:
                # XXX: This was meant only as a proof of concept.
                event_time_adv = Utils.sort_non_padding(event_time_adv)

            if self.opt.op_mode == OperationMode.ATTACK_EVAL:
                adv_examples.append((event_type_adv.detach().cpu(), event_time_adv.detach().cpu()))
            else:
                adv_examples.append((event_type_adv, event_time_adv))

            time_taken = time.time() - start_time
            total_time_taken += time_taken

            with torch.no_grad():
                adv_enc_out, adv_pred = self.defender_model(event_type_adv, event_time_adv)
                _, adv_pred_num = Utils.type_loss(adv_pred[0], event_type, self.pred_loss_func)
                adv_se = Utils.time_loss(adv_pred[1], event_time)

                predicted_marks_full.append(adv_pred[0].detach().cpu())
                predicted_times_full.append(adv_pred[1].detach().cpu())

                total_time_se += clean_se.detach()
                total_event_rate += clean_pred_num.detach()
                total_robust_time_se += adv_se.detach()
                total_robust_event_rate += adv_pred_num.detach()
                total_num_event += event_type.ne(TConst.PAD).sum()
                total_num_pred += event_type.ne(TConst.PAD).sum().detach() - event_time.shape[0]
                std_error_sample_size += event_type.ne(TConst.PAD).sum().detach() - self.opt.std_error_subtractor

                #event_type_pert_classes = event_type_adv
                #if self.opt.one_hot_mark:
                #    event_type_pert_classes = torch.argmax(event_type_adv, dim=-1)

                W_dist, mark_W, time_W = Utils.wasserstein_distance(event_time_adv, event_time,
                    event_type_adv=event_type_adv, event_type_clean=event_type)
                W_dist_shifted, _, time_W_shifted = Utils.wasserstein_distance(event_time_adv, event_time,
                    event_type_adv=event_type_adv, event_type_clean=event_type,
                    shift_by_min=True)
                W_dist_main += W_dist
                W_dist_main_shifted += W_dist_shifted
                W_time += time_W
                W_time_shifted += time_W_shifted
                W_mark += mark_W
                emb_dist_main += torch.sum((1 - F.cosine_similarity(adv_enc_out, clean_enc_out)))

                num_events = event_time.ne(TConst.PAD).sum(dim=-1)
                interr_avg = time_gap.sum(dim=-1) / num_events
                event_time_adv_srt = Utils.sort_non_padding(event_time_adv)
                Wt = torch.sum(torch.abs(event_time_adv_srt - event_time), dim=-1)
                Wt_avg = Wt / num_events
                total_avg_Wt += Wt_avg.sum()
                total_avg_interr += interr_avg.sum()

        mae = total_time_se / total_num_pred
        accuracy = total_event_rate / total_num_pred
        robust_mae = total_robust_time_se / total_num_pred
        robust_accuracy = total_robust_event_rate / total_num_pred

        predicted_marks = torch.vstack(predicted_marks_full)
        clean_marks = torch.vstack(clean_marks_full)
        mpa_std_error = Utils.get_mpa_std_error(predicted_marks, clean_marks, std_error_sample_size)

        predicted_times = torch.vstack(predicted_times_full)
        clean_times = torch.vstack(clean_times_full)
        mae_std_error = Utils.get_mae_std_error(predicted_times, clean_times, std_error_sample_size)

        if self.to_print:
            logger.info(f"Clean Stats: Mark Acc = {accuracy}, Time MAE = {mae}")
            logger.info(f"Robust Stats: Mark Acc = {robust_accuracy}, Time MAE = {robust_mae}")
            logger.info(f"MPA std error: {mpa_std_error}, MAE std error: {mae_std_error}")

            logger.info(f"Total number of events: {total_num_event}")
            logger.info(f"(Test) Total Wasserstein Distance: {W_time} (time) +"
                  f" {W_mark} (mark) = {W_dist_main}")
            logger.info(f"Avg W distance at this iteration: "
                  f"{W_time / total_num_event} (time) + {W_mark / total_num_event} (mark) = "
                  f"{W_dist_main / total_num_event}")
            logger.info(f"Avg W distance at this iteration, shifted by min: "
                  f"{W_time_shifted / total_num_event} (time) + {W_mark / total_num_event} (mark) = "
                  f"{W_dist_main_shifted / total_num_event}")
            logger.info(f"(Test) Total embedding cosine distance: {emb_dist_main}")

        logger.info(f"Total time taken for optimization: {total_time_taken} seconds")

        return {
            "adv_examples": adv_examples,
            "mae": mae,
            "accuracy": accuracy,
            "robust_mae": robust_mae,
            "robust_accuracy": robust_accuracy,
            "W_time": W_time,
            "W_mark": W_mark,
            "W_dist_main": W_dist_main,
            "emb_dist_main": emb_dist_main,
            "mpa_std_error": mpa_std_error,
            "mae_std_error": mae_std_error
        }

    def perform_substitution(self, adv_examples):
        test_data = list(self.test_data)
        adv_marks = []
        adv_times = []

        clean_marks = []
        clean_times = []

        for adv_m, adv_t in adv_examples:
            adv_marks.append(adv_m)
            adv_times.append(adv_t)

        for clean_t, _, clean_m in test_data:
            clean_marks.append(clean_m)
            clean_times.append(clean_t)

        clean_marks = torch.vstack(clean_marks)
        clean_times = torch.vstack(clean_times)

        adv_marks = torch.vstack(adv_marks)
        adv_times = torch.vstack(adv_times)

        subs_timeseqs = []
        subs_markseqs = []

        for seq_idx, (timeseq, markseq) in enumerate(tqdm(zip(clean_times, clean_marks), total=len(clean_times))):
            adv_timeseq = adv_times[seq_idx]

            # Pick a random event in the sequence
            adv_event_idx = random.choice(range(len(adv_timeseq)))
            adv_event_time = adv_timeseq[adv_event_idx]

            # Search in all other sequences for an
            # event whose time wasserstein distance is
            # closest to this one. Replace this event with
            # that one.

            min_w_dist_time = None
            min_w_dist_mark = None
            min_w_dist = 100000

            seq_idxs = [(seq_idx + j) % len(clean_times) for j in range(len(clean_times))]
            for seq_check_idx in seq_idxs:
                seq_check_times = clean_times[seq_check_idx]
                seq_check_marks = clean_marks[seq_check_idx]

                for seq_check_e_idx, event_time in enumerate(seq_check_times):
                    w_dist = torch.abs(event_time - adv_event_time)

                    if w_dist < min_w_dist:
                        min_w_dist = w_dist
                        min_w_dist_time = event_time
                        min_w_dist_mark = seq_check_marks[seq_check_e_idx]

            # Replace time and mark
            mark_seq = clean_marks[seq_idx].clone()
            time_seq = clean_times[seq_idx].clone()
            mark_seq[adv_event_idx] = min_w_dist_mark
            time_seq[adv_event_idx] = min_w_dist_time

            subs_timeseqs.append(time_seq)
            subs_markseqs.append(mark_seq)

        subs_timeseqs = torch.vstack(subs_timeseqs)
        subs_markseqs = torch.vstack(subs_markseqs)

        return subs_timeseqs, subs_markseqs, clean_times, clean_marks

    def substitute_attack(self, adv_examples):
        subs_timeseqs, subs_markseqs, clean_times, clean_marks =\
                self.perform_substitution(adv_examples) 
        total_len = subs_timeseqs.shape[0]

        total_time_se = 0
        total_robust_time_se = 0
        total_event_rate = 0
        total_robust_event_rate = 0
        total_num_pred = 0
        total_num_event = 0
        std_error_sample_size = 0

        W_dist_main = 0
        W_dist_main_shifted = 0
        W_time = 0
        W_time_shifted = 0
        W_mark = 0
        emb_dist_main = 0

        total_avg_Wt = 0

        for batch_start in tqdm(range(0, total_len, self.opt.batch_size)):
            batch_end = batch_start + total_len
            event_type = clean_marks[batch_start: batch_end].to(self.opt.device)
            event_time = clean_times[batch_start: batch_end].to(self.opt.device)
            event_type_adv = subs_markseqs[batch_start: batch_end].to(self.opt.device)
            event_time_adv = subs_timeseqs[batch_start: batch_end].to(self.opt.device)

            with torch.no_grad():
                clean_enc_out, clean_pred = self.defender_model(event_type, event_time)
                clean_loss_dict = self.defender_model.loglike_loss(
                    clean_enc_out, event_time, event_type, clean_pred, self.pred_loss_func)
                clean_pred_num = clean_loss_dict['pred_num_event']
                clean_se = clean_loss_dict['se']

                adv_enc_out, adv_pred = self.defender_model(event_type_adv, event_time_adv)
                _, adv_pred_num = Utils.type_loss(adv_pred[0], event_type, self.pred_loss_func)
                adv_se = Utils.time_loss(adv_pred[1], event_time)

                total_time_se += clean_se.detach()
                total_event_rate += clean_pred_num.detach()
                total_robust_time_se += adv_se.detach()
                total_robust_event_rate += adv_pred_num.detach()
                total_num_event += event_type.ne(TConst.PAD).sum()
                total_num_pred += event_type.ne(TConst.PAD).sum().detach() - event_time.shape[0]
                std_error_sample_size += event_type.ne(TConst.PAD).sum().detach() - self.opt.std_error_subtractor

                #event_type_pert_classes = event_type_adv
                #if self.opt.one_hot_mark:
                #    event_type_pert_classes = torch.argmax(event_type_adv, dim=-1)

                W_dist, mark_W, time_W = Utils.wasserstein_distance(event_time_adv, event_time,
                    event_type_adv=event_type_adv, event_type_clean=event_type)
                W_dist_shifted, _, time_W_shifted = Utils.wasserstein_distance(event_time_adv, event_time,
                    event_type_adv=event_type_adv, event_type_clean=event_type,
                    shift_by_min=True)
                W_dist_main += W_dist
                W_dist_main_shifted += W_dist_shifted
                W_time += time_W
                W_time_shifted += time_W_shifted
                W_mark += mark_W
                emb_dist_main += torch.sum((1 - F.cosine_similarity(adv_enc_out, clean_enc_out)))

                num_events = event_time.ne(TConst.PAD).sum(dim=-1)
                event_time_adv_srt = Utils.sort_non_padding(event_time_adv)
                Wt = torch.sum(torch.abs(event_time_adv_srt - event_time), dim=-1)
                Wt_avg = Wt / num_events
                total_avg_Wt += Wt_avg.sum()

        mae = total_time_se / total_num_pred
        accuracy = total_event_rate / total_num_pred
        robust_mae = total_robust_time_se / total_num_pred
        robust_accuracy = total_robust_event_rate / total_num_pred

        logger.info(f"Clean Stats: Mark Acc = {accuracy}, Time MAE = {mae}")
        logger.info(f"Robust Stats: Mark Acc = {robust_accuracy}, Time MAE = {robust_mae}")

        logger.info(f"Total number of events: {total_num_event}")
        logger.info(f"(Test) Total Wasserstein Distance: {W_time} (time) +"
                f" {W_mark} (mark) = {W_dist_main}")
        logger.info(f"Avg W distance at this iteration: "
                f"{W_time / total_num_event} (time) + {W_mark / total_num_event} (mark) = "
                f"{W_dist_main / total_num_event}")
        logger.info(f"Avg W distance at this iteration, shifted by min: "
                f"{W_time_shifted / total_num_event} (time) + {W_mark / total_num_event} (mark) = "
                f"{W_dist_main_shifted / total_num_event}")
        logger.info(f"(Test) Total embedding cosine distance: {emb_dist_main}")