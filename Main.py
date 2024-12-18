import argparse
import os
import time
import sys

import torch
from torch.profiler import profiler, record_function, ProfilerActivity
import torch.nn as nn
import torch.optim as optim

from attack_baselines import adv_train as advt
from attack_baselines.train_trades import train_trades
from attack_baselines.pgd import PGD
from attack_baselines.ts_prob import ts_prob_attack
from Constants import ArchitectureChoice, AttackRegularizerChoice, BlackBoxSubtype, TrainMode, TrainTimeAttack,\
    NoiseModelChoice, NoiseActivationFunction, OperationMode, ThreatModel, CleanTrainsetChoice
from transformer.Models import Transformer, RNN_TPP
import Utils
import Train

from loguru import logger
DATA_PATH = os.environ.get("DATA_PATH", "./data")

# Notes -:
# Target model params: 8 heads, 128-dim d_k, 128-dim d_v
# Source model params: 8 heads, 128-dim d_k, 128-dim d_v
# Target and source assumed to be trained on full trainset.
# Trainset not used by attacker in either WB/BB. Attacker learns to generate noise on test set.

@logger.catch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True, help="Name of the dataset folder")
    parser.add_argument('-fold', type=int, default=1)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-attack_iters', type=int, default=20)
    parser.add_argument('-defense_iters', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=32)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-shuffle_trainset', action='store_true')
    parser.add_argument('-standardize', action='store_true',
        help="If data has not already been min-max scaled before use, then"
             " do so before training.")
    parser.add_argument('-std_scale', type=float, default=1.0,
        help="Scale to normalize time data to during standardization.")
    parser.add_argument('-same_perm_matrix', action="store_true",
        help="Whether to use the same permutation matrix for mark and time data.")
    parser.add_argument('-se_time_scale', type=float, default=1.0)

    parser.add_argument('-log_path')

    parser.add_argument('-std_error_subtractor', type=int,
        help="When computing the norm constant for MPA/MAE std error, what number to subtract on each batch by")
    parser.add_argument('-std_error_max_allowed_len', type=int, default=20,
        help="How many elements of each sequence count during std_error calc")

    parser.add_argument("-arch", type=ArchitectureChoice.from_string,
        default=ArchitectureChoice.THP, choices=list(ArchitectureChoice),
        help="Choice of base TPP model on which to perform attacks")

    # Ablations and other experiment options
    parser.add_argument('-ablation_nopermloss', action="store_true")
    parser.add_argument('-ablation_nohinge', action="store_true")
    parser.add_argument('-ablation_notimenoise', action='store_true')

    # Note: remove_sin_cos is for the attacker phase of adversarial training,
    # as well as normal THP training. remove_sin_cos_defense is for
    # the defense phase in adversarial training.
    # XXX: Unused
    parser.add_argument('-remove_sin_cos', action='store_true')
    parser.add_argument('-remove_sin_cos_defense', action='store_true')

    parser.add_argument('-adv_lr', type=float, default=1e-3)
    parser.add_argument('-pad_max_len', type=int, default=200)
    parser.add_argument('-d_noise_rnn', type=int, default=64)
    parser.add_argument('-noise_rnn_layers', type=int, default=5)
    parser.add_argument('-d_gphi', type=int, default=64)
    parser.add_argument('-gs_iters', type=int, default=20)
    parser.add_argument('-gs_tau', type=float, default=0.1)
    parser.add_argument('-noise_pow_param', type=int, default=10)
    parser.add_argument("-noise_model", type=NoiseModelChoice.from_string,
        default=NoiseModelChoice.UNIFORM_NOISE, choices=list(NoiseModelChoice))
    parser.add_argument("-noise_act", type=NoiseActivationFunction.from_string,
        default=NoiseActivationFunction.TANH, choices=list(NoiseActivationFunction))
    parser.add_argument('-kl_alpha', type=float, default=0.3)
    parser.add_argument('-kl_beta', type=float, default=0.2)
    parser.add_argument('-gamma', type=float, default=0.8, help="Noise hinge loss coefficient")
    parser.add_argument('-sparse_mode', action='store_true',
        help="(NoiseTransformerV2) If we want to sparsify the noise in our attack")
    parser.add_argument("-train_mode", type=TrainMode.from_string,
        default=TrainMode.ADV_LLH_DIAG, choices=list(TrainMode))
    parser.add_argument("-train_time_attack", type=TrainTimeAttack.from_string,
        default=TrainTimeAttack.OUR, choices=list(TrainTimeAttack))
    parser.add_argument("-op_mode", type=OperationMode.from_string,
        default=OperationMode.ATTACK_EVAL, choices=list(OperationMode))
    parser.add_argument("-threat_model", type=ThreatModel.from_string,
        default=ThreatModel.WHITE_BOX, choices=list(ThreatModel))
    parser.add_argument("-bb_subtype", type=BlackBoxSubtype.from_string,
        default=BlackBoxSubtype.RANDOMNESS, choices=list(BlackBoxSubtype))
    parser.add_argument("-clean_trainset", type=CleanTrainsetChoice.from_string,
        default=CleanTrainsetChoice.TARGET, choices=list(CleanTrainsetChoice),
        help="Train a clean model on certain parts of the training set")

    parser.add_argument("-attack_reg", type=AttackRegularizerChoice.from_string,
        default=AttackRegularizerChoice.NONE, choices=list(AttackRegularizerChoice),
        help="Whether to add KL/Hellinger regularization during attack phase (our method)")
    parser.add_argument("-momentum_decay_mu", type=float, default=0.033,
        help="(MI-FGSM Attack) The mu parameter for momentum calculation")
    parser.add_argument("-baseline_epsilon", type=float, default=0.031,
        help="(Attack baselines) How much noise to add [higher variance]")
    parser.add_argument("-sparse_hidden", type=int, default=32,
        help="(Attack baselines) Hidden dimension for SparseLayer")
    parser.add_argument("-batch_norm", action="store_true",
        help="Batchnorm for noise generator models")
    parser.add_argument("-sparse_range", type=float, default=3,
        help="Parameter controlling noise range of SparseLayer")
    parser.add_argument("-sparse_reg", action="store_true",
        help="Whether to regularize the perm matrix that is learned for sparse noise")
    parser.add_argument("-sort_sequence", action="store_true",
        help="To show that differentiability breaks in baselines if sequence is sorted")
    parser.add_argument('-kappa', type=float, default=40,
        help="(Attack baselines) For robust TS and SparseLayer attacks, how much of the noise vector to sparsify")
    parser.add_argument('-min_factor', type=float, default=0.5,
        help="(Sparse noise transformer) Minimum noise values will be min_factor times the min of each sequence")
    parser.add_argument('-max_factor', type=float, default=3.5,
        help="(Sparse noise transformer) Max noise values will be max_factor times the max of each sequence")

    parser.add_argument('-ckpt', type=str, default='./CKPT')
    parser.add_argument('-defender_src_path', type=str, help="Defender src model loaded during attack eval")
    parser.add_argument('-defender_tgt_path', type=str, help="Defender target model loaded during attack eval")
    parser.add_argument('-save_freq', type=int, default=5)
    # TODO: Implement this option and other loading options
    parser.add_argument('-load_best_attacker', action='store_true')
    parser.add_argument('-profile', action='store_true', help="If set, measure time and memory taken.")

    parser.add_argument('-pgd_substitute_attack', action='store_true')

    opt = parser.parse_args()
    Utils.set_seed(opt.seed)
    timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())

    dataset_name = opt.data.strip('/')
    tgt_name = ""

    if opt.defender_tgt_path:
        tgt_name = opt.defender_tgt_path.split('/')[-2]

    if opt.threat_model in [ThreatModel.WHITE_BOX_SOURCE, ThreatModel.BLACK_BOX]:
        threat_model = f"{opt.threat_model}_variant:{opt.bb_subtype}"
    else:
        threat_model = f"{opt.threat_model}"

    if opt.op_mode == OperationMode.ATTACK_EVAL:
        if opt.train_time_attack == TrainTimeAttack.NONE:
            if opt.clean_trainset == CleanTrainsetChoice.TARGET:
                opt.ckpt_dir = os.path.join(opt.ckpt, f'{dataset_name}_{opt.arch}_fold{opt.fold}_CLEAN_{opt.clean_trainset}_{timestamp}')
            elif opt.clean_trainset == CleanTrainsetChoice.SOURCE:
                opt.ckpt_dir = os.path.join(opt.ckpt, f'{dataset_name}_{opt.arch}_fold{opt.fold}_CLEAN_{opt.clean_trainset}_variant:{opt.bb_subtype}_{timestamp}')

        elif opt.train_time_attack == TrainTimeAttack.OUR:
            opt.ckpt_dir = os.path.join(opt.ckpt, f'{dataset_name}_{opt.arch}_fold{opt.fold}_{opt.train_time_attack}_{threat_model}_{opt.noise_model}_{opt.noise_act}_Sparse:{opt.sparse_mode}_{tgt_name}_{timestamp}')
        else:
            opt.ckpt_dir = os.path.join(opt.ckpt, f'{dataset_name}_{opt.arch}_fold{opt.fold}_{opt.train_time_attack}_{threat_model}_{tgt_name}_{timestamp}')

    elif opt.op_mode == OperationMode.ATTACK_PLUS_DEFENSE:
        if opt.train_time_attack == TrainTimeAttack.OUR:
            opt.ckpt_dir = os.path.join(opt.ckpt, f'{dataset_name}_{opt.arch}_ADVTRAIN_fold{opt.fold}_{opt.train_time_attack}_{threat_model}_{opt.clean_trainset}_{opt.bb_subtype}_{opt.noise_model}_{opt.noise_act}_Sparse:{opt.sparse_mode}_attack:{opt.attack_iters}_def:{opt.defense_iters}_{timestamp}')

        else:
            opt.ckpt_dir = os.path.join(opt.ckpt, f'{dataset_name}_{opt.arch}_ADVTRAIN_fold{opt.fold}_{opt.train_time_attack}_{threat_model}_{opt.clean_trainset}_{opt.bb_subtype}_attack:{opt.attack_iters}_def:{opt.defense_iters}_{timestamp}')

    os.makedirs(opt.ckpt_dir)

    print(f"COMMANDLINE ARGS: {' '.join(sys.argv[1:])}")
    if opt.log_path:
        logger.add(opt.log_path, level="DEBUG")

    logger.info(f"COMMANDLINE ARGS: {' '.join(sys.argv[1:])}")
    logger.info(f"CHECKPOINT PATH: {opt.ckpt_dir}")

    opt.device = torch.device('cuda')
    trainloader_tgt, trainloader_src, valloader, testloader, num_types,\
        min_ie_time, max_len = Utils.prepare_dataloader(opt)

    if opt.arch == ArchitectureChoice.THP:
        tpp_model = Transformer(
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
        )
        tpp_model_class = Transformer
        tpp_model.to(opt.device)

    elif opt.arch == ArchitectureChoice.RNN_TPP:
        tpp_model = RNN_TPP(
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            pad_max_len=opt.pad_max_len
        )
        tpp_model_class = RNN_TPP
        tpp_model.to(opt.device)

    tpp_optimizer = optim.Adam(filter(lambda x: x.requires_grad, tpp_model.parameters()), opt.lr,
            betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(tpp_optimizer, 10, gamma=0.5)

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    if min_ie_time == 0:
        logger.info("Smoothing min_ie_time from 0 to 1e-5")
        min_ie_time = 1e-5

    if opt.op_mode == OperationMode.ATTACK_EVAL and opt.train_time_attack == TrainTimeAttack.NONE:
        if opt.clean_trainset == CleanTrainsetChoice.TARGET:
            Train.train(tpp_model, trainloader_tgt, testloader, tpp_optimizer, scheduler, pred_loss_func, opt)
        elif opt.clean_trainset == CleanTrainsetChoice.SOURCE:
            Train.train(tpp_model, trainloader_src, testloader, tpp_optimizer, scheduler, pred_loss_func, opt)

    elif opt.op_mode == OperationMode.ATTACK_PLUS_DEFENSE:
        if opt.train_time_attack in [TrainTimeAttack.PGD, TrainTimeAttack.MI_FGSM, TrainTimeAttack.TS_DET]:
            advt.train_with_adversary_interleaved_free(tpp_model, trainloader_tgt, testloader,
                tpp_optimizer, scheduler, pred_loss_func, opt)

        elif opt.train_time_attack == TrainTimeAttack.OUR:
            res = Utils.init_OUR_modules(opt, num_types, min_ie_time)
            adv_model = res["adv_model"]
            adv_optimizer = res["adv_optimizer"]

            advt.train_with_adversary_interleaved_model(tpp_model, trainloader_tgt, testloader,
                tpp_optimizer, scheduler, pred_loss_func, adv_model, adv_optimizer, opt)

        elif opt.train_time_attack == TrainTimeAttack.TS_PROB:
            res = Utils.init_sparsenet_modules(opt)
            adv_model = res["adv_model"]
            adv_optimizer = res["adv_optimizer"]

            advt.train_with_adversary_interleaved_model(tpp_model, trainloader_tgt, testloader,
                tpp_optimizer, scheduler, pred_loss_func, adv_model, adv_optimizer, opt)

    elif opt.op_mode == OperationMode.ATTACK_EVAL:
        if opt.train_time_attack == TrainTimeAttack.OUR:
            res = Utils.init_OUR_modules(opt, num_types, min_ie_time)
            adv_model = res["adv_model"]
            adv_optimizer = res["adv_optimizer"]

            if opt.profile:
                # If profiling, just run our attack method directly.
                with profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("adversarial_loop"):
                        Train.train_with_adversary_interleaved(tpp_model, adv_model,
                            trainloader, valloader, testloader, adv_optimizer, tpp_optimizer,
                            scheduler, pred_loss_func, opt)

                logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                return

            if opt.threat_model == ThreatModel.WHITE_BOX_SOURCE:
                # During attack phase, generate permutation and noise using the source network.
                tpp_model_tgt, tpp_model_src = Utils.init_defender_models(tpp_model_class, opt, num_types)
                Train.white_box_attack_source(tpp_model_src, tpp_model_tgt, adv_model,
                    testloader, adv_optimizer, pred_loss_func, opt)

            elif opt.threat_model == ThreatModel.WHITE_BOX:
                tpp_model = Utils.init_defender_models(Transformer, opt, num_types)
                Train.white_box_attack(tpp_model, adv_model, testloader, adv_optimizer,
                                       pred_loss_func, opt)

            elif opt.threat_model == ThreatModel.BLACK_BOX:
                tpp_model_tgt, tpp_model_src = Utils.init_defender_models(tpp_model_class, opt, num_types)
                Train.black_box_attack(tpp_model_src, tpp_model_tgt, adv_model,
                                       testloader, adv_optimizer, pred_loss_func, opt)

        elif opt.train_time_attack == TrainTimeAttack.TRADES:
            # XXX: unused
            train_trades(tpp_model, trainloader, valloader, tpp_optimizer,
                         scheduler, pred_loss_func, opt)

        elif opt.train_time_attack in [TrainTimeAttack.PGD, TrainTimeAttack.TS_DET,
                                       TrainTimeAttack.FGSM, TrainTimeAttack.MI_FGSM]:

            tpp_model_tgt, tpp_model_src = Utils.init_defender_models(tpp_model_class, opt, num_types)
            pgd = PGD(tpp_model_tgt, testloader, pred_loss_func, opt,
                      sort_sequence=opt.sort_sequence, epsilon=opt.baseline_epsilon,
                      momentum_decay_mu=opt.momentum_decay_mu, num_steps=opt.attack_iters,
                      defender_model_src=tpp_model_src)
            res = pgd.attack()
            logger.debug(f"Saving {opt.train_time_attack} adversarial examples in pickle file.")
            adv_examples = res["adv_examples"]
            torch.save(adv_examples, os.path.join(opt.ckpt_dir,
                f'{opt.train_time_attack}_adv_examples.pkl'))

            if opt.pgd_substitute_attack:
                pgd.substitute_attack(adv_examples)

        elif opt.train_time_attack == TrainTimeAttack.TS_PROB:
            res = Utils.init_sparsenet_modules(opt)
            adv_model = res["adv_model"]
            adv_optimizer = res["adv_optimizer"]
            tpp_model_tgt, tpp_model_src = Utils.init_defender_models(tpp_model_class, opt, num_types)

            ts_prob_attack(tpp_model_tgt, adv_model, adv_optimizer, testloader, pred_loss_func, opt,
                           defender_model_src=tpp_model_src)


if __name__ == '__main__':
    main()
