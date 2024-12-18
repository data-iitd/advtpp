from tqdm import tqdm
import math
import pickle
import os
import time
import torch
import torch.nn.functional as F
import transformer.Constants as Constants
from transformer.Models import get_non_pad_mask
import Utils

from loguru import logger

from Common import eval_epoch, train_for_defense_iter
from Constants import KL_MAX, AttackRegularizerChoice, TrainMode, NoiseModelChoice

# TODO: Change thp_model to tpp_model

def white_box_attack_source(source_model, defender_model, adv_model,
    test_data, adv_optimizer, pred_loss_func, opt):
    """
    Train the adv_model on the test set, with the help of the source model.
    This is similar to black-box setting, but the loss is still calculated on the
    defender model. Meant to be a sanity check - WB accuracy under this setting
    *should* be worse than black-box accuracy.

    Store the average cosine distance between clean and noisy embeddings when
    computing metrics on the test set. This shall be used to sample noise from
    a range to add to the embedding for each of the baselines, to bring the
    baselines on par with our attack.
    """
    defender_model.eval()
    source_model.eval()

    # First, save the config file
    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)

    W_dist_main = 0
    emb_dist_main = 0
    clean_int_main = 0
    adv_int_main = 0
    total_time_taken = 0

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        logger.info(f'[ WHITE BOX (S) EPOCH {epoch} ]')

        test_acc_list = []
        test_mae_list = []
        robust_test_acc_list = []
        robust_test_mae_list = []

        logger.info("\n----ADVERSARY TRAINING----\n")
        start_time = time.time()
        train_event, train_type, train_time = train_with_adversary_iter(
            defender_model, adv_model, test_data, adv_optimizer, pred_loss_func, opt,
            source_model_sanity=source_model)
        epoch_time = time.time() - start_time
        total_time_taken += epoch_time
        logger.info('(Training) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}, Average Loss/LL: {loss: 8.5f}'
            .format(type=train_type, mae=train_time, loss=train_event))
        logger.info(f"Epoch {epoch} took {epoch_time} seconds to train")

        torch.save(adv_model.state_dict(),
            os.path.join(opt.ckpt_dir,
            f'adv_model_epoch_{epoch}.pkl'))

        logger.info("\n----DEFENDER EVALUATION----\n")
        test_event, test_type, test_time, robust_test_time, robust_test_type,\
            W_distance, emb_distance, clean_intensity, adv_intensity = eval_epoch(defender_model,
                test_data, pred_loss_func, opt, adv_model=adv_model)

        W_dist_main += (W_distance / len(test_data))
        emb_dist_main += (emb_distance / len(test_data))
        clean_int_main += (clean_intensity / len(test_data))
        adv_int_main += (adv_intensity / len(test_data))

        logger.info('(Test) Clean Acc: {type: 8.5f}, Clean MAE: {mae: 8.5f}'
              .format(type=test_type, mae=test_time))
        logger.info('(Test) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}'
              .format(type=robust_test_type, mae=robust_test_time))

        test_acc_list += [test_type]
        test_mae_list += [test_time]
        robust_test_acc_list += [robust_test_type]
        robust_test_mae_list += [robust_test_time]

    W_dist_main /= opt.epoch
    emb_dist_main /= opt.epoch
    clean_int_main /= opt.epoch
    adv_int_main /= opt.epoch

    logger.info('(Test) Best Clean ACC: {pred: 8.5f}, Best Clean MAE: {mae: 8.5f}'
          .format(pred=max(test_acc_list), mae=min(test_mae_list)))
    logger.info('(Test) Best Robust ACC: {pred: 8.5f}, Best Robust MAE: {mae: 8.5f}'
          .format(pred=max(robust_test_acc_list), mae=min(robust_test_mae_list)))
    logger.info(f"(Test) Avg Wasserstein Distance: {W_dist_main}")
    logger.info(f"(Test) Avg embedding cosine distance: {emb_dist_main}")
    logger.info(f"(Test) Avg intensity: {clean_int_main} (clean) vs {adv_int_main} (adv)")
    logger.info(f"Total training time: {total_time_taken} seconds")


def white_box_attack(defender_model, adv_model, test_data, adv_optimizer,
                     pred_loss_func, opt):
    """
    Train the adv_model on the test set.
    Then use the adv_model to craft adversarial examples on the test set which
    the defender must defend against.
    Store the average cosine distance between clean and noisy embeddings when
    computing metrics on the test set. This shall be used to sample noise from
    a range to add to the embedding for each of the baselines, to bring the
    baselines on par with our attack.
    """
    defender_model.eval()

    # First, save the config file
    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)

    W_dist_main = 0
    emb_dist_main = 0
    clean_int_main = 0
    adv_int_main = 0

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        logger.info(f'[ WHITE BOX EPOCH {epoch} ]')

        test_acc_list = []
        test_mae_list = []
        robust_test_acc_list = []
        robust_test_mae_list = []

        logger.info("\n----ADVERSARY TRAINING----\n")
        train_event, train_type, train_time = train_with_adversary_iter(
            defender_model, adv_model, test_data, adv_optimizer, pred_loss_func, opt)
        logger.info('(Training) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}, Average Loss/LL: {loss: 8.5f}'
            .format(type=train_type, mae=train_time, loss=train_event))

        torch.save(adv_model.state_dict(),
            os.path.join(opt.ckpt_dir,
            f'adv_model_epoch_{epoch}.pkl'))

        logger.info("\n----DEFENDER EVALUATION----\n")
        test_event, test_type, test_time, robust_test_time, robust_test_type, W_distance, emb_distance, \
            clean_intensity, adv_intensity = eval_epoch(defender_model,
                test_data, pred_loss_func, opt, adv_model=adv_model)

        W_dist_main += (W_distance / len(test_data))
        emb_dist_main += (emb_distance / len(test_data))
        clean_int_main += (clean_intensity / len(test_data))
        adv_int_main += (adv_intensity / len(test_data))

        logger.info('(Test) Clean Acc: {type: 8.5f}, Clean MAE: {mae: 8.5f}'
              .format(type=test_type, mae=test_time))
        logger.info('(Test) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}'
              .format(type=robust_test_type, mae=robust_test_time))

        test_acc_list += [test_type]
        test_mae_list += [test_time]
        robust_test_acc_list += [robust_test_type]
        robust_test_mae_list += [robust_test_time]

    W_dist_main /= opt.epoch
    emb_dist_main /= opt.epoch
    clean_int_main /= opt.epoch
    adv_int_main /= opt.epoch

    logger.info('(Test) Best Clean ACC: {pred: 8.5f}, Best Clean MAE: {mae: 8.5f}'
          .format(pred=max(test_acc_list), mae=min(test_mae_list)))
    logger.info('(Test) Best Robust ACC: {pred: 8.5f}, Best Robust MAE: {mae: 8.5f}'
          .format(pred=max(robust_test_acc_list), mae=min(robust_test_mae_list)))
    logger.info(f"(Test) Avg Wasserstein Distance: {W_dist_main}")
    logger.info(f"(Test) Avg embedding cosine distance: {emb_dist_main}")
    logger.info(f"(Test) Avg intensity: {clean_int_main} (clean) vs {adv_int_main} (adv)")


def black_box_attack(defender_model_src, defender_model_tgt, adv_model,
                     test_data, adv_optimizer, pred_loss_func, opt):
    """
    The adversary model (assuming oracle) is trained on the test data, using
    a trained source defender model.
    We then use the trained adv_model to craft adversarial examples on the
    test set, which the target defender must defend against.
    Store the average cosine distance between clean and noisy embeddings when
    computing metrics on the test set. This shall be used to sample noise from
    a range to add to the embedding for each of the baselines, to bring the
    baselines on par with our attack.
    """
    defender_model_src.eval()
    defender_model_tgt.eval()

    # First, save the config file
    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)

    W_dist_main = 0
    emb_dist_main = 0
    clean_int_main = 0
    adv_int_main = 0

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        logger.info(f'[ BLACK BOX EPOCH {epoch} ]')

        test_acc_list = []
        test_mae_list = []
        robust_test_acc_list = []
        robust_test_mae_list = []

        logger.info("\n----ADVERSARY TRAINING----\n")
        train_event, train_type, train_time = train_with_adversary_iter(
            defender_model_src, adv_model, test_data, adv_optimizer, pred_loss_func, opt)
        logger.info('(Training) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}, Average Loss/LL: {loss: 8.5f}'
            .format(type=train_type, mae=train_time, loss=train_event))

        torch.save(adv_model.state_dict(),
            os.path.join(opt.ckpt_dir,
            f'adv_model_epoch_{epoch}.pkl'))

        logger.info("\n----DEFENDER EVALUATION----\n")
        test_event, test_type, test_time, robust_test_time, robust_test_type,\
            W_distance, emb_distance, clean_intensity, adv_intensity = eval_epoch(defender_model_tgt,
                test_data, pred_loss_func, opt, adv_model=adv_model, defender_model_src=defender_model_src)

        W_dist_main += (W_distance / len(test_data))
        emb_dist_main += (emb_distance / len(test_data))
        clean_int_main += (clean_intensity / len(test_data))
        adv_int_main += (adv_intensity / len(test_data))

        logger.info('(Test) Clean Acc: {type: 8.5f}, Clean MAE: {mae: 8.5f}'
              .format(type=test_type, mae=test_time))
        logger.info('(Test) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}'
              .format(type=robust_test_type, mae=robust_test_time))

        test_acc_list += [test_type]
        test_mae_list += [test_time]
        robust_test_acc_list += [robust_test_type]
        robust_test_mae_list += [robust_test_time]

    W_dist_main /= opt.epoch
    emb_dist_main /= opt.epoch
    clean_int_main /= opt.epoch
    adv_int_main /= opt.epoch

    logger.info('(Test) Best Clean ACC: {pred: 8.5f}, Best Clean MAE: {mae: 8.5f}'
          .format(pred=max(test_acc_list), mae=min(test_mae_list)))
    logger.info('(Test) Best Robust ACC: {pred: 8.5f}, Best Robust MAE: {mae: 8.5f}'
          .format(pred=max(robust_test_acc_list), mae=min(robust_test_mae_list)))
    logger.info(f"(Test) Avg Wasserstein Distance: {W_dist_main}")
    logger.info(f"(Test) Avg embedding cosine distance: {emb_dist_main}")
    logger.info(f"(Test) Avg intensity: {clean_int_main} (clean) vs {adv_int_main} (adv)")


def train_with_adversary_interleaved(thp_model, adv_model, training_data,
    val_data, test_data, adv_optimizer, thp_optimizer, scheduler,
    pred_loss_func, opt):
    """
    Train the adversarial generator for opt.attack_iters steps and then train
    the THP model for opt.defense_iters steps. Repeat this for opt.epoch epochs.
    """
    # TODO: At some point, include early stopping based on the validation metrics.
    # After training is complete, evaluate trained model on test set against
    # known strong attacks in the literature.
    epoch_stats = {}

    # First, save the config file
    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        logger.info('[ EPOCH', epoch, ']')

        val_acc_list = []
        val_mae_list = []
        robust_val_acc_list = []
        robust_val_mae_list = []
        val_acc_max = 0

        attack_train_acc_list = []
        attack_train_mae_list = []
        attack_train_acc_min = 1

        logger.info("----ADVERSARY TRAINING----\n")

        Utils.unfreeze_network_weights(adv_model)
        Utils.freeze_network_weights(thp_model)

        for attack_iter_i in range(opt.attack_iters):
            attack_iter = attack_iter_i + 1
            logger.info('[Attack Iteration', attack_iter, ']')

            train_event, train_type, train_time = train_with_adversary_iter(
                thp_model, adv_model, training_data, adv_optimizer, pred_loss_func, opt)
            logger.info('(Training) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}, Average Loss/LL: {loss: 8.5f}'
                .format(type=train_type, mae=train_time, loss=train_event))

            attack_train_acc_list += [train_type]
            attack_train_mae_list += [train_time]
            logger.info('Lowest robust attack train ACC: {pred: 8.5f}, max robust MAE: {mae: 8.5f}'.format(
                  pred=min(attack_train_acc_list),
                  mae=max(attack_train_mae_list)))

            if attack_train_acc_min > train_time:
                attack_train_acc_min = train_time

                torch.save(adv_model.state_dict(),
                    os.path.join(opt.ckpt_dir,
                    f'adv_model_epoch_{epoch}_best.pkl'))

        logger.info("\n----DEFENDER TRAINING----\n")

        Utils.unfreeze_network_weights(thp_model)
        Utils.freeze_network_weights(adv_model)

        for defense_iter_i in range(opt.defense_iters):
            defense_iter = defense_iter_i + 1
            logger.info('[ Defense Iteration', defense_iter, ']')

            train_event, train_type, train_time, robust_train_time, robust_train_type = \
                train_for_defense_iter(thp_model, adv_model, training_data, test_data,
                    thp_optimizer, scheduler, pred_loss_func, opt)
            logger.info('(Training) Clean Acc: {type: 8.5f}, Clean MAE: {mae: 8.5f}, Average Loss/LL: {loss: 8.5f}'
                .format(type=train_type, mae=train_time, loss=train_event))
            logger.info('(Training) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}'
                  .format(type=robust_train_type, mae=robust_train_time))

            scheduler.step()

        # Check validation metrics and save defender model accordingly.
        val_event, val_type, val_time, robust_val_time, robust_val_type, _, _ = \
            eval_epoch(thp_model, val_data, pred_loss_func, opt, adv_model=adv_model)
        logger.info('(Validation) Clean Acc: {type: 8.5f}, Clean MAE: {mae: 8.5f}'
              .format(type=val_type, mae=val_time))
        logger.info('(Validation) Robust Acc: {type: 8.5f}, Robust MAE: {mae: 8.5f}'
              .format(type=robust_val_type, mae=robust_val_time))

        val_acc_list += [val_type]
        val_mae_list += [val_time]
        robust_val_acc_list += [robust_val_type]
        robust_val_mae_list += [robust_val_time]
        logger.info('(Validation) Best Clean ACC: {pred: 8.5f}, Best Clean MAE: {mae: 8.5f}'
              .format(pred=max(val_acc_list), mae=min(val_mae_list)))
        logger.info('(Validation) Best Robust ACC: {pred: 8.5f}, Best Robust MAE: {mae: 8.5f}'
              .format(pred=max(robust_val_acc_list), mae=min(robust_val_mae_list)))

        if val_acc_max < val_type:
            val_acc_max = val_type

            torch.save(thp_model.state_dict(),
                os.path.join(opt.ckpt_dir,
                    f'thp_model_epoch_{epoch}_best.pkl'))
            logger.info(f"Saved best defender model (acc. to validation set) at epoch {epoch}")

        epoch_stats[epoch] = \
            Utils.serialize_epoch_stats(attack_train_acc_list, attack_train_mae_list,
                val_acc_list, val_mae_list, robust_val_acc_list, robust_val_mae_list)

        torch.save(epoch_stats, os.path.join(opt.ckpt_dir, f'epoch_stats.pkl'))

def train_with_adversary_iter(thp_model, adv_model, training_data, optimizer, pred_loss_func, opt,
                              source_model_sanity=None, to_print=True):
    """
    Trains one epoch of the adversarial sequence generator model's components. The THP model's
    weights are assumed to have been frozen (before the model is input here) while the
    adversarial model is trained.
    """
    adv_model.train()
    # Base models should not have to be set on train mode, but it turns out if we're computing
    # a loss using a base model, and that loss is part of backward, we need to set to train mode.
    # We're doing this only because of torch RNN. Either way only adv_model's weights will be updated.
    thp_model.train()
    if source_model_sanity is not None:
        source_model_sanity.train()

    total_event_ll = 0
    total_time_se = 0
    total_event_rate = 0
    total_num_event = 0
    total_num_pred = 0

    inner_optim_sum = 0
    # Permutation
    perm_diag_loss = 0
    # Time noise
    noise_diff_loss = 0
    sorted_loss = 0
    max_hinge_loss = 0
    min_hinge_loss = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Adversarial Generation)   ', leave=False):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        if source_model_sanity is not None:
            clean_enc_out, _ = source_model_sanity(event_type, event_time, remove_sin_cos=opt.remove_sin_cos)
        else:
            clean_enc_out, _ = thp_model(event_type, event_time, remove_sin_cos=opt.remove_sin_cos)

        event_type_permed, event_time_noisy, extras = adv_model.forward(batch, clean_enc_out,
            no_time_noise=opt.ablation_notimenoise)

        event_type_perms = extras["event_type_perms"]
        event_time_perms = extras["event_time_perms"]
        noise_diff = extras["noise_diff"]
        hinge_term = extras["hinge_term"]
        min_hinge = extras["min_hinge"]
        max_hinge = extras["max_hinge"]
        gumbel_masks = extras["gumbel_masks"]

        if not opt.ablation_notimenoise:
            # Hard clip: Maintain min and max of sequence.
            non_pad_mask = get_non_pad_mask(event_time).squeeze(-1)
            noise_min, noise_max = Utils.sequence_extremes(event_time, non_pad_mask, opt.max_factor, opt.min_factor)
            event_time_noisy = torch.clamp(event_time_noisy, min=noise_min, max=noise_max)

        enc_out, prediction = thp_model(event_type_permed, event_time_noisy,
                                        remove_sin_cos=opt.remove_sin_cos)
        # As part of the inner optimization problem (ref: Madry), we want to
        # minimize log likelihood of _actual_ next event t_j given the hidden state/history
        # of _perturbed_ events upto that point. Following similar logic, the XEnt and MAE
        # losses are with respect to actual event data.
        loss_dict = thp_model.loglike_loss(enc_out, event_time, event_type, prediction, pred_loss_func)

        inner_optim_loss = loss_dict['pred_loss'] + loss_dict['se'] / opt.se_time_scale + loss_dict['nll']
        inner_optim_sum += inner_optim_loss
        loss = -inner_optim_loss

        if opt.noise_model == NoiseModelChoice.NOISE_MLP:
            # XXX: Unused
            loss += torch.nn.functional.relu(adv_model.noise_generator.min_ie_time / 2 - noise_vecs) + \
                torch.nn.functional.relu(noise_vecs - adv_model.noise_generator.min_ie_time / 2)

        elif opt.noise_model in [NoiseModelChoice.NOISE_RNN, NoiseModelChoice.NOISE_TRANSFORMER,
                                 NoiseModelChoice.NOISE_TRANSFORMER_V2]:

            if opt.ablation_notimenoise:
                pass

            else:
                noise_diff_norm = torch.norm(noise_diff)
                hinge_term_sum = torch.sum(hinge_term) # sortedness inequalities
                max_hinge_sum = torch.sum(max_hinge)
                min_hinge_sum = torch.sum(min_hinge)

                if opt.ablation_nohinge:
                    loss += (noise_diff_norm + min_hinge_sum + max_hinge_sum)
                else:
                    loss += (noise_diff_norm + opt.gamma * hinge_term_sum +
                             min_hinge_sum + max_hinge_sum)

                noise_diff_loss += noise_diff_norm
                sorted_loss += hinge_term_sum
                max_hinge_loss += max_hinge_sum
                min_hinge_loss += min_hinge_sum

        if opt.train_mode == TrainMode.ADV_LLH_DIAG:
            eye = torch.eye(*event_type_perms.shape[1:]).to(opt.device)
            eye = eye * ~gumbel_masks
            type_diag = (event_type_perms - eye)

            # Get the diagonal entries of each perm matrix in the batch as an array.
            # Then square for squared error.
            type_diag = torch.diagonal(type_diag, dim1=-2, dim2=-1) ** 2
            type_diag_sum = type_diag.sum()

            if not opt.ablation_nopermloss:
                loss += type_diag_sum

            perm_diag_loss += type_diag_sum

            if event_time_perms is not None and not opt.same_perm_matrix:
                eye = torch.eye(*event_time_perms.shape[1:]).to(opt.device)
                eye = eye * ~gumbel_masks

                time_diag = event_time_perms - eye
                time_diag = torch.diagonal(time_diag, dim1=-2, dim2=-1) ** 2

                if not opt.ablation_nopermloss:
                    loss += time_diag.sum()

        if opt.attack_reg != AttackRegularizerChoice.NONE:
            event_ll_clean, non_event_ll_clean = Utils.log_likelihood(
                thp_model, clean_enc_out, event_time, event_type)
            log_ll_clean = event_ll_clean - non_event_ll_clean

            kl_div = torch.min(torch.tensor([F.kl_div(log_ll, log_ll_clean, log_target=True), KL_MAX]))

            if opt.attack_reg == AttackRegularizerChoice.KLDIV:
                loss += (opt.kl_alpha * log_ll.sum() + (1 - opt.kl_alpha) * kl_div.sum())

            elif opt.attack_reg == AttackRegularizerChoice.HELLINGER:
                # XXX: Unused, as converting from log probs to actual probs can lead to nan issues.
                hellinger = Utils.hellinger_distance(log_ll, log_ll_clean)
                loss += (opt.kl_alpha * log_ll.sum() + (1 - opt.kl_alpha) * hellinger.sum())

            elif opt.attack_reg == AttackRegularizerChoice.KLDIV_BETA:
                loss += (opt.kl_alpha * log_ll.sum() +
                         (1 - opt.kl_alpha - opt.kl_beta) * log_ll_clean.sum() +
                         opt.kl_beta * kl_div.sum())

        loss.backward()
        optimizer.step()

        total_event_ll += loss_dict['nll'].item() # store the magnitude of LL
        total_time_se += loss_dict['se'].item()
        total_event_rate += loss_dict['pred_num_event'].item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - opt.std_error_subtractor

    if to_print:
        # logger.info(f"Inner optim. loss on this iteration: {inner_optim_sum / total_num_event}")
        logger.info(f"Permutation loss: {perm_diag_loss / total_num_event}")
        logger.info(f"Time losses: norm = {noise_diff_loss / total_num_event}, sortedness = {sorted_loss / total_num_event}")
        logger.info(f"Time losses: max hinge = {max_hinge_loss / total_num_event}, min hinge = {min_hinge_loss / total_num_event}")

    mae = total_time_se / total_num_pred
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, mae

def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    model.train()

    total_event_ll = 0
    total_time_se = 0
    total_event_rate = 0
    total_num_event = 0
    total_num_pred = 0
    total_mse = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, remove_sin_cos=opt.remove_sin_cos)

        loss_dict = model.loglike_loss(enc_out, event_time, event_type, prediction, pred_loss_func)
        # Scales to stabilize training
        scale_time_loss = 1
        loss = loss_dict['nll'] + loss_dict['pred_loss'] + loss_dict['se'] / scale_time_loss
        loss.backward()

        optimizer.step()

        total_event_ll += -(loss_dict['nll']).item() # store the magnitude of LL
        total_time_se += loss_dict['se'].item()
        total_mse += loss_dict['mse'].item()
        total_event_rate += loss_dict['pred_num_event'].item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_se / total_num_pred
    rmse = math.sqrt(total_mse / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, mae, rmse

def train(model, training_data, test_data, optimizer, scheduler, pred_loss_func, opt):
    test_acc_list = []
    test_mae_list = []

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        logger.info(f'[ Epoch {epoch} ]')

        start = time.time()
        train_event, train_type, train_time, rmse = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        logger.info('(Training) Acc: {type: 8.5f}, MAE: {mae: 8.5f}'.format(type=train_type, mae=train_time))
        logger.info(f'(Training) RMSE: {rmse: 8.5f}')

        start = time.time()
        test_event, test_type, test_time, _, _, _, _, _, _ = eval_epoch(model, test_data, pred_loss_func, opt)
        logger.info('(Test) Acc: {type: 8.5f}, MAE: {mae: 8.5f}'.format(type=test_type, mae=test_time))

        test_acc_list += [test_type]
        test_mae_list += [test_time]
        logger.info('Best ACC: {pred: 8.5f}, MAE: {mae: 8.5f}'.format(pred=max(test_acc_list), mae=min(test_mae_list)))

        scheduler.step()

        torch.save(model.state_dict(),
            os.path.join(opt.ckpt_dir, f'thp_model_{epoch}.pkl'))
        torch.save([test_acc_list, test_mae_list],
            os.path.join(opt.ckpt_dir, f'normal_metrics_{epoch}.pkl'))

    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)