from tqdm import tqdm
import math
import pickle
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import get_non_pad_mask
import Utils

from attack_baselines.pgd import PGD
from attack_baselines.ts_prob import ts_prob_attack

from Common import eval_epoch, train_for_defense_iter
from Constants import TrainMode, TrainTimeAttack
from Train import train_with_adversary_iter as our_train_iter

from loguru import logger

# TODO: Put all this logic, including the choice logic in Main, into parent and sub-classes.


def train_with_adversary_interleaved_free(defender_model_tgt, train_data, test_data, thp_optimizer,
                                          scheduler, pred_loss_func, opt):
    """
    Run the adversarial method for given number of steps, and then obtain the adversarial examples using the frozen
    defender model. Then unfreeze and train the defender model on these adversarial examples.
    """
    epoch_stats = {}

    # First, save the config file
    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)

    test_acc_list = []
    test_mae_list = []

    # placeholders
    robust_test_acc_list = []
    robust_test_mae_list = []

    attack_train_acc_list = []
    attack_train_mae_list = []

    total_time = time.time()

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ EPOCH', epoch, ']')

        print("----ADVERSARY TURN----\n")

        Utils.freeze_network_weights(defender_model_tgt)

        attack_time = time.time()
        pgd = PGD(defender_model_tgt, train_data, pred_loss_func, opt,
                  momentum_decay_mu=opt.momentum_decay_mu,
                  sort_sequence=opt.sort_sequence, epsilon=opt.baseline_epsilon,
                  to_print=False, num_steps=opt.attack_iters)
        res = pgd.attack()

        attack_time = time.time() - attack_time
        adv_examples = res['adv_examples']
        robust_acc = res['robust_accuracy']
        robust_mae = res['robust_mae']

        attack_train_acc_list += [robust_acc]
        attack_train_mae_list += [robust_mae]

        print(f"Epoch {epoch} attack time: {attack_time}")

        print("\n----DEFENDER TURN----\n")

        Utils.unfreeze_network_weights(defender_model_tgt)

        def_time = time.time()
        for defense_iter_i in range(opt.defense_iters):
            defense_iter = defense_iter_i + 1
            print('[ Defense Training iter', defense_iter, ']')

            loss, acc, mae, robust_acc, robust_mae = \
                train_for_defense_iter(defender_model_tgt, train_data,
                    thp_optimizer, scheduler, pred_loss_func, opt, adv_examples=adv_examples)

            scheduler.step()

        def_time = time.time() - def_time

        print(f"Epoch {epoch} defense time: {def_time}")

        # EVAL SECTION
        # Check metrics and save defender model accordingly.
        _, test_acc, test_mae, _, _, _, _, _, _ = \
            eval_epoch(defender_model_tgt, test_data, pred_loss_func, opt,
                       to_print=False)
        logger.info('(Eval) Acc on natural test examples: {acc: 8.5f}, MAE: {mae: 8.5f}'
              .format(acc=test_acc, mae=test_mae))

        test_acc_list.append(test_acc)
        test_mae_list.append(test_mae)

        if epoch % opt.save_freq == 0:
            torch.save(defender_model_tgt.state_dict(), os.path.join(opt.ckpt_dir, f'defender_model_tgt_epoch_{epoch}_best.pkl'))
            print(f"Saved best defender model (acc. to test set: {test_acc:8.5f}) at epoch {epoch}")

        epoch_stats[epoch] = \
            Utils.serialize_epoch_stats(attack_train_acc_list, attack_train_mae_list,
                test_acc_list, test_mae_list, robust_test_acc_list, robust_test_mae_list)

        torch.save(epoch_stats, os.path.join(opt.ckpt_dir, f'epoch_stats.pkl'))

    total_time = time.time() - total_time
    print(f"Time taken overall: {total_time}s")
    print('(Evaluation) Best Natural Acc: {pred: 8.5f}, Best Natural MAE: {mae: 8.5f}'
          .format(pred=max(test_acc_list), mae=min(test_mae_list)))


def train_with_adversary_interleaved_model(defender_model_tgt, train_data, test_data, thp_optimizer,
                                           scheduler, pred_loss_func, adv_model, adv_optimizer, opt):
    """
    Run the adversarial method for given number of steps, and then obtain the adversarial examples using the frozen
    defender model. Then unfreeze and train the defender model on these adversarial examples.

    This method is used specifically for model-based attacks.
    """
    epoch_stats = {}

    # First, save the config file
    with open(os.path.join(opt.ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt.__dict__, f)

    test_acc_list = []
    test_mae_list = []

    # placeholders
    robust_test_acc_list = []
    robust_test_mae_list = []

    attack_train_acc_list = []
    attack_train_mae_list = []

    total_time = time.time()

    for epoch_i in range(opt.epoch):
        epoch = epoch_i +   1
        print('[ EPOCH', epoch, ']')
        print("----ADVERSARY TURN----\n")

        Utils.unfreeze_network_weights(adv_model)
        Utils.freeze_network_weights(defender_model_tgt)

        attack_time = time.time()
        if opt.train_time_attack == TrainTimeAttack.TS_PROB:
            res = ts_prob_attack(defender_model_tgt, adv_model, adv_optimizer, train_data, pred_loss_func, opt,
                                 to_print=False)
            robust_acc = res["robust_acc"]
            robust_mae = res["robust_mae"]

        elif opt.train_time_attack == TrainTimeAttack.OUR:
            for attack_iter_i in range(opt.attack_iters):
                logger.info(f"[ ATTACK ITER: {attack_iter_i + 1} ]")
                _, train_type, train_time = our_train_iter(
                    defender_model_tgt, adv_model, train_data, adv_optimizer,
                    pred_loss_func, opt, to_print=False)

            robust_acc = train_type
            robust_mae = train_time

        attack_time = time.time() - attack_time
        attack_train_acc_list += [robust_acc]
        attack_train_mae_list += [robust_mae]

        print(f"Epoch {epoch} attack time: {attack_time}")

        print("\n----DEFENDER TURN----\n")

        Utils.freeze_network_weights(adv_model)
        Utils.unfreeze_network_weights(defender_model_tgt)

        def_time = time.time()
        for defense_iter_i in range(opt.defense_iters):
            defense_iter = defense_iter_i + 1
            logger.info(f'[ DEFENSE ITER: {defense_iter} ]')

            loss, acc, mae, robust_acc, robust_mae = \
                train_for_defense_iter(defender_model_tgt, train_data,
                    thp_optimizer, scheduler, pred_loss_func, opt, adv_model=adv_model)

            scheduler.step()

        def_time = time.time() - def_time

        # EVAL SECTION
        # Check metrics and save defender model accordingly.
        _, test_acc, test_mae, _, _, _, _, _, _ = \
            eval_epoch(defender_model_tgt, test_data, pred_loss_func, opt,
                       to_print=False)
        logger.info('(Eval) Acc on natural test examples: {acc: 8.5f}, MAE: {mae: 8.5f}'
              .format(acc=test_acc, mae=test_mae))

        test_acc_list.append(test_acc)
        test_mae_list.append(test_mae)

        if epoch % opt.save_freq == 0:
            torch.save(defender_model_tgt.state_dict(), os.path.join(opt.ckpt_dir, f'defender_model_tgt_epoch_{epoch}_best.pkl'))
            print(f"Saved best defender model (acc. to test set: {test_acc:8.5f}) at epoch {epoch}")

        epoch_stats[epoch] = \
            Utils.serialize_epoch_stats(attack_train_acc_list, attack_train_mae_list,
                test_acc_list, test_mae_list, robust_test_acc_list, robust_test_mae_list)

        torch.save(epoch_stats, os.path.join(opt.ckpt_dir, f'epoch_stats.pkl'))

    total_time = time.time() - total_time
    print(f"Time taken overall: {total_time}s")
    print('(Evaluation) Best Natural Acc: {pred: 8.5f}, Best Natural MAE: {mae: 8.5f}'
          .format(pred=max(test_acc_list), mae=min(test_mae_list)))
