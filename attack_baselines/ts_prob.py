import time

import torch
import torch.nn.functional as F

from Constants import ThreatModel
from tqdm import tqdm
import transformer.Constants as TConst
import Utils

from loguru import logger


def ts_prob_attack(defender_model_tgt, sparse_net, sparse_optim, test_data, pred_loss_func,
                   opt, defender_model_src=None, to_print=True):
    defender_model_tgt.train()

    if defender_model_src is not None:
        defender_model_src.train()

    sparse_net.train()

    W_dist = 0
    emb_dist = 0

    if opt.threat_model in [ThreatModel.WHITE_BOX, ThreatModel.WHITE_BOX_SOURCE]:
        logger.info("---------TS Probabilistic Attack [WHITE BOX]--------")
    elif opt.threat_model == ThreatModel.BLACK_BOX:
        logger.info("---------TS Probabilistic Attack [BLACK BOX]--------")

    # XXX: earlier used opt.epoch
    total_time_taken = 0
    for epoch_i in range(opt.attack_iters):
        epoch = epoch_i + 1
        total_time_se = 0
        total_robust_time_se = 0
        total_event_rate = 0
        total_robust_event_rate = 0
        total_num_pred = 0
        std_error_sample_size = 0

        predicted_marks_full = []
        predicted_times_full = []
        clean_marks_full = []
        clean_times_full = []

        total_num_event = 0

        W_dist_main = 0
        W_dist_main_shifted = 0
        W_time = 0
        W_time_shifted = 0
        W_mark = 0
        emb_dist_main = 0

        total_time_epoch = 0
        total_time_epoch_gen = 0

        for batch in tqdm(test_data, mininterval=2, desc='  - (TS Prob Attack WB)   ', leave=False):
            time_start = time.time()
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            clean_marks_full.append(event_type.detach().cpu())
            clean_times_full.append(event_time.detach().cpu())

            clean_enc_out, clean_pred = defender_model_tgt(event_type, event_time)
            _, clean_pred_num = Utils.type_loss(clean_pred[0], event_type, pred_loss_func)
            clean_se = Utils.time_loss(clean_pred[1], event_time)

            sparse_optim.zero_grad()
            noise = sparse_net(event_time, norm=opt.batch_norm)

            # PGD style loss focusing on mark prediction
            event_time_adv = event_time + noise
            if opt.sort_sequence:
                event_time_adv = Utils.sort_non_padding(event_time_adv)

            if opt.threat_model in [ThreatModel.WHITE_BOX, ThreatModel.WHITE_BOX_SOURCE]:
                adv_enc_out, adv_pred = defender_model_tgt(event_type, event_time_adv)
                loss_dict = defender_model_tgt.loglike_loss(adv_enc_out, event_time, event_type,
                    adv_pred, pred_loss_func)

            elif opt.threat_model == ThreatModel.BLACK_BOX:
                adv_enc_out, adv_pred = defender_model_src(event_type, event_time_adv)
                loss_dict = defender_model_src.loglike_loss(adv_enc_out, event_time, event_type,
                    adv_pred, pred_loss_func)

            adv_pred_loss = loss_dict['pred_loss'] + loss_dict['se'] / opt.se_time_scale\
                                + loss_dict['nll']
            adv_pred_loss = -adv_pred_loss
            adv_pred_loss.backward()
            sparse_optim.step()
            epoch_batch_time = time.time() - time_start
            total_time_epoch += epoch_batch_time

            with torch.no_grad():
                # For inference time counting purpose
                start_time = time.time()
                noise = sparse_net(event_time, norm=opt.batch_norm)
                time_taken_batch = time.time() - start_time
                total_time_epoch_gen += time_taken_batch

                adv_enc_out, adv_pred = defender_model_tgt(event_type, event_time_adv)
                _, adv_pred_num = Utils.type_loss(adv_pred[0], event_type, pred_loss_func)
                adv_se = Utils.time_loss(adv_pred[1], event_time)

                predicted_marks_full.append(adv_pred[0].detach().cpu())
                predicted_times_full.append(adv_pred[1].detach().cpu())

                total_num_event += event_type.ne(TConst.PAD).sum()
                total_time_se += clean_se.item()
                total_event_rate += clean_pred_num.item()
                total_robust_time_se += adv_se.item()
                total_robust_event_rate += adv_pred_num.item()
                total_num_pred += event_type.ne(TConst.PAD).sum().item() - event_time.shape[0]
                std_error_sample_size += event_type.ne(TConst.PAD).sum() - opt.std_error_subtractor

                W_d, mark_W, time_W = Utils.wasserstein_distance(event_time_adv, event_time)
                W_d_shifted, mark_W, time_W_shifted = Utils.wasserstein_distance(event_time_adv, event_time,
                    shift_by_min=True)

                W_dist_main += W_d
                W_dist_main_shifted += W_d_shifted
                W_time += time_W
                W_time_shifted += time_W_shifted
                W_mark += mark_W
                emb_dist_main += torch.sum((1 - F.cosine_similarity(adv_enc_out, clean_enc_out)))

        logger.info(f"Time to generate adv examples: {total_time_epoch_gen} seconds")

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

        if to_print:
            logger.info(f"EPOCH: {epoch}")
            # logger.info(f"Clean Stats: Mark Acc = {accuracy}, Time MAE = {mae}")
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

        total_time_taken += total_time_epoch
        logger.info(f"Epoch {epoch} took {total_time_epoch} seconds")

        W_dist += W_dist_main
        emb_dist += emb_dist_main

    if to_print:
        logger.info(f"(Test) Avg Wasserstein Distance: {W_dist / opt.epoch}")
        logger.info(f"(Test) Avg embedding cosine distance: {emb_dist / opt.epoch}")

    logger.info(f"Total time taken during training: {total_time_taken} seconds")

    return {
        "accuracy": accuracy,
        "mae": mae,
        "robust_acc": robust_accuracy,
        "robust_mae": robust_mae,
        "mpa_std_error": mpa_std_error,
        "mae_std_error": mae_std_error
    }
