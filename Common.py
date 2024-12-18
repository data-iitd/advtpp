from tqdm import tqdm
import time
import torch
import torch.nn.functional as F

from Constants import TrainTimeAttack

import transformer.Constants as Constants
from transformer.Models import get_non_pad_mask
import Utils

from loguru import logger


def eval_epoch(thp_model, test_data, pred_loss_func, opt, adv_model=None, defender_model_src=None,
			   adv_examples=None, to_print=True):
    thp_model.eval()
    if adv_model is not None:
        adv_model.eval()

    predicted_marks_full = []
    predicted_times_full = []
    clean_marks_full = []
    clean_times_full = []

    total_event_ll = 0
    total_time_se = 0
    total_robust_time_se = 0
    total_event_rate = 0
    total_robust_event_rate = 0
    total_num_event = 0
    total_num_pred = 0
    std_error_sample_size = 0
    # total_seqs = 0
    total_loss = 0

    W_distance = 0
    W_distance_shifted = 0
    W_time = 0
    W_time_shifted = 0
    W_mark = 0
    emb_distance = 0
    clean_intensity = 0
    adv_intensity = 0

    total_time_epoch = 0

    with torch.no_grad():
        for bindex, batch in enumerate(tqdm(test_data, mininterval=2, desc='  - (Evaluation) ', leave=False)):
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            clean_marks_full.append(event_type.detach().cpu())
            clean_times_full.append(event_time.detach().cpu())

            # Clean metrics
            clean_enc_out_tgt, clean_prediction = thp_model(event_type, event_time)
            event_ll, non_event_ll = Utils.log_likelihood(thp_model, clean_enc_out_tgt, event_time, event_type)
            # In this case, we're just computing the LL, not optimizing over it. So no need for minus.
            event_loss = torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(clean_prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(clean_prediction[1], event_time)

            # Robust metrics
            if adv_examples is not None or adv_model is not None:
                event_type_pert = event_type

                if adv_examples is not None:
                    event_time_noisy = adv_examples[bindex]

                elif adv_model is not None:
                    start_time = time.time()
                    if opt.train_time_attack == TrainTimeAttack.TS_PROB:
                        noise = adv_model(event_time, norm=opt.batch_norm)
                        event_time_noisy = event_time + noise

                    elif opt.train_time_attack == TrainTimeAttack.OUR:
                        if defender_model_src is not None:
                            clean_enc_out, _ = defender_model_src(event_type, event_time)
                        else:
                            clean_enc_out = clean_enc_out_tgt

                        event_type_pert, event_time_noisy, extras = \
                            adv_model.forward(batch, clean_enc_out, no_time_noise=opt.ablation_notimenoise)

                    time_taken_batch = time.time() - start_time
                    total_time_epoch += time_taken_batch

                    # Hard clip: Maintain min and max of sequence.
                    if not opt.ablation_notimenoise:
                        non_pad_mask = get_non_pad_mask(event_time).squeeze(-1)
                        noise_min, noise_max = Utils.sequence_extremes(event_time, non_pad_mask, opt.max_factor,
                            opt.min_factor)
                        event_time_noisy = torch.clamp(event_time_noisy, min=noise_min, max=noise_max)

                enc_out, prediction = thp_model(event_type_pert, event_time_noisy)
                predicted_marks_full.append(prediction[0].detach().cpu())
                predicted_times_full.append(prediction[1].detach().cpu())

                robust_pred_loss, robust_pred_num = Utils.type_loss(prediction[0],
                        event_type, pred_loss_func)
                robust_se = Utils.time_loss(prediction[1], event_time)
                robust_event_ll, robust_non_event_ll = Utils.log_likelihood(thp_model, enc_out, event_time, event_type)
                robust_event_loss = torch.sum(robust_event_ll - robust_non_event_ll)

                total_robust_time_se += robust_se
                total_robust_event_rate += robust_pred_num

                total_loss += (robust_pred_loss) + (robust_se) + robust_event_loss

            total_event_ll += event_loss
            total_time_se += se
            total_event_rate += pred_num
            total_num_event += event_type.ne(Constants.PAD).sum()
            # total_num_pred += event_type.ne(Constants.PAD).sum() - event_time.shape[0]
            total_num_pred += event_type.ne(Constants.PAD).sum() - opt.std_error_subtractor
            std_error_sample_size += event_type.ne(Constants.PAD).sum() - opt.std_error_subtractor

            if adv_examples is not None or adv_model is not None:
                curr_W, mark_W, time_W = Utils.wasserstein_distance(event_time_noisy, event_time,
                    event_type_adv=event_type_pert, event_type_clean=event_type)
                curr_W_shifted, _, time_W_shifted = Utils.wasserstein_distance(
                    event_time_noisy, event_time, event_type_adv=event_type_pert,
                    event_type_clean=event_type, shift_by_min=True
                )

                W_mark += mark_W
                W_time += time_W
                W_time_shifted += time_W_shifted
                W_distance += curr_W
                W_distance_shifted += curr_W_shifted

                emb_distance += torch.sum((1 - F.cosine_similarity(enc_out, clean_enc_out_tgt)))
                clean_intensity += torch.sum(Utils._compute_intensity(thp_model, clean_enc_out_tgt,
                    event_time, types=event_type)[0])
                adv_intensity += torch.sum(Utils._compute_intensity(thp_model, enc_out,
                    event_time, types=event_type)[0])

        if adv_examples is not None or adv_model is not None:
            predicted_marks = torch.vstack(predicted_marks_full)
            clean_marks = torch.vstack(clean_marks_full)
            mpa_std_error = Utils.get_mpa_std_error(predicted_marks, clean_marks, std_error_sample_size)

            predicted_times = torch.vstack(predicted_times_full)
            clean_times = torch.vstack(clean_times_full)
            mae_std_error = Utils.get_mae_std_error(predicted_times, clean_times, std_error_sample_size)

    logger.info(f"Time taken to generate adv. examples: {total_time_epoch}")
    if to_print:
        logger.info(f"Total number of events: {total_num_event}")
        # logger.info(f"Evaluation loss at this iteration: {total_loss / total_num_event}")

        if adv_examples is not None or adv_model is not None:
            logger.info(f"MPA std error: {mpa_std_error}, MAE std error: {mae_std_error}")
            logger.info(f"Total W distance at this iteration: "
                  f"{W_time} (time) + {W_mark} (mark) = "
                  f"{W_distance}")
            logger.info(f"Avg W distance at this iteration: "
                  f"{W_time / total_num_event} (time) + {W_mark / total_num_event} (mark) = "
                  f"{W_distance / total_num_event}")
            logger.info(f"Avg W distance at this iteration, shifted by min: "
                  f"{W_time_shifted / total_num_event} (time) + {W_mark / total_num_event} (mark) = "
                  f"{W_distance_shifted / total_num_event}")
            logger.info(f"Total cosine embedding at this iteration: {emb_distance}")
            logger.info(f"Total intensity: {clean_intensity} (clean) vs {adv_intensity} (adv)")

    robust_mae = total_robust_time_se / total_num_pred
    mae = total_time_se / total_num_pred
    return (total_event_ll / total_num_event).detach(), (total_event_rate / total_num_pred).detach(),\
        mae.detach(), robust_mae.detach(), (total_robust_event_rate / total_num_pred).detach(), \
        W_distance, emb_distance, clean_intensity, adv_intensity


def train_for_defense_iter(thp_model, test_data, optimizer, scheduler,
                           pred_loss_func, opt, adv_examples=None, adv_model=None):
    thp_model.train()
    if adv_model is not None:
        adv_model.eval()

    total_event_ll = 0
    total_time_se = 0
    total_robust_time_se = 0
    total_event_rate = 0
    total_robust_event_rate = 0
    total_num_event = 0
    total_num_pred = 0

    for bindex, batch in enumerate(tqdm(test_data, mininterval=2, desc='  - (Defense)   ', leave=False)):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        # In the outer optimization problem (ref: Madry), we wish to minimize
        # loss (maximize LL, minimize other components) where hidden state is
        # based on perturbed event data and the next event is from actual event
        # data.
        clean_enc_out, clean_prediction = thp_model(event_type, event_time,
            remove_sin_cos=opt.remove_sin_cos_defense)
        event_type_pert = event_type

        if adv_examples is not None:
            event_type_pert, event_time_noisy = adv_examples[bindex]

        elif adv_model is not None:
            if opt.train_time_attack == TrainTimeAttack.TS_PROB:
                # TODO: Add noise to mark here as well.
                noise = adv_model(event_time, norm=opt.batch_norm)
                event_time_noisy = event_time + noise

            elif opt.train_time_attack == TrainTimeAttack.OUR:
                event_type_pert, event_time_noisy, _ = \
                    adv_model.forward(batch, clean_enc_out)

        enc_out, prediction = thp_model(event_type_pert, event_time_noisy)

        # "Robust" loss components
        # TODO: Replace with loglike_loss()
        event_ll, non_event_ll = Utils.log_likelihood(thp_model, enc_out,
            event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)
        robust_pred_loss, robust_pred_num_event = Utils.type_loss(
            prediction[0], event_type, pred_loss_func)
        robust_se = Utils.time_loss(prediction[1], event_time)

        with torch.no_grad():
            # "Clean" metrics
            pred_loss, pred_num_event = Utils.type_loss(clean_prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(clean_prediction[1], event_time)

        # Scales to stabilize training
        scale_time_loss = 1
        loss = event_loss + robust_pred_loss + robust_se / scale_time_loss
        loss.backward()

        optimizer.step()

        total_event_ll += -event_loss
        total_time_se += se
        total_robust_time_se += robust_se
        total_event_rate += pred_num_event
        total_robust_event_rate += robust_pred_num_event
        total_num_event += event_type.ne(Constants.PAD).sum()
        total_num_pred += event_type.ne(Constants.PAD).sum() - event_time.shape[0]

    mae = total_time_se / total_num_pred
    robust_mae = total_robust_time_se / total_num_pred
    return (total_event_ll / total_num_event).detach(), (total_event_rate / total_num_pred).detach(),\
        mae.detach(), robust_mae.detach(), (total_robust_event_rate / total_num_pred).detach()