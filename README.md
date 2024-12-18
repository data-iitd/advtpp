# Differentiable Adversarial Attacks for Marked Temporal Point Processes (AAAI 2025)

### Requirements

Run the following command in a suitable virtual environment to install requirements.

```
pip install -r requirements.txt
```

### Environment Variables

We use one variable - `DATA_PATH` - in some places in code. It should be set before running any of the code.

### Commands

We list the commands for training the THP model on clean examples, and for evaluating the clean-trained THP model using various attacks. In each experiment, the log file contains the path to the current checkpoint folder.

#### Clean Training

1. Taobao

```
python Main.py -data <path to taobao dataset> -pad_max_len 64 -train_time_attack NONE -batch_size 128 -epoch 20 -clean_trainset TARGET -bb_subtype RANDOMNESS -std_error_subtractor 128 -standardize -std_scale 1e-4 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to taobao dataset> -pad_max_len 64 -train_time_attack NONE -batch_size 128 -epoch 20 -clean_trainset SOURCE -bb_subtype RANDOMNESS -seed 100 -std_error_subtractor 128 -standardize -std_scale 1e-4 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to taobao dataset> -pad_max_len 64 -train_time_attack NONE -batch_size 128 -epoch 20 -clean_trainset SOURCE -bb_subtype TRAINSET -std_error_subtractor 128 -standardize -std_scale 1e-4 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>
```

2. Health

```
python Main.py -data <path to Health dataset> -pad_max_len 512 -train_time_attack NONE -batch_size 16 -epoch 20 -clean_trainset TARGET -bb_subtype RANDOMNESS -std_error_subtractor 16 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to Health dataset> -pad_max_len 512 -train_time_attack NONE -batch_size 16 -epoch 20 -clean_trainset SOURCE -bb_subtype RANDOMNESS -seed 100 -std_error_subtractor 16 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to Health dataset> -pad_max_len 512 -train_time_attack NONE -batch_size 16 -epoch 20 -clean_trainset SOURCE -bb_subtype TRAINSET -std_error_subtractor 16 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>
```

3. Twitter

```
python Main.py -data <path to Health dataset> -pad_max_len 512 -train_time_attack NONE -batch_size 16 -epoch 20 -clean_trainset TARGET -bb_subtype RANDOMNESS -std_error_subtractor 16 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to Health dataset> -pad_max_len 512 -train_time_attack NONE -batch_size 16 -epoch 20 -clean_trainset SOURCE -bb_subtype RANDOMNESS -seed 100 -std_error_subtractor 16 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to Health dataset> -pad_max_len 512 -train_time_attack NONE -batch_size 16 -epoch 20 -clean_trainset SOURCE -bb_subtype TRAINSET -std_error_subtractor 16 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>
```

4. Electricity
```
python Main.py -data <path to Electricity dataset> -pad_max_len 300 -train_time_attack NONE -batch_size 128 -epoch 20 -clean_trainset TARGET -bb_subtype RANDOMNESS -std_error_subtractor 128 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to Electricity dataset> -pad_max_len 300 -train_time_attack NONE -batch_size 128 -epoch 20 -clean_trainset SOURCE -bb_subtype RANDOMNESS -seed 100 -std_error_subtractor 128 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>

python Main.py -data <path to Electricity dataset> -pad_max_len 300 -train_time_attack NONE -batch_size 128 -epoch 20 -clean_trainset SOURCE -bb_subtype TRAINSET -std_error_subtractor 128 -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file>
```

#### Evaluating Clean Models

The first command is in the white box setting, while the second is in the black box setting.

1. Taobao
```
# PermTPP
python Main.py -data ./taobao -pad_max_len 64 -train_time_attack OUR -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -standardize -std_scale 1e-4 -d_k 128 -d_v 128 -n_head 8 -batch_norm -std_error_subtractor 128 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data ./taobao -pad_max_len 64 -train_time_attack OUR -batch_size 128 -epoch 20 -threat_model BLACK_BOX -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -standardize -std_scale 1e-4 -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 128

# PGD
python Main.py -data ./taobao -pad_max_len 64 -train_time_attack PGD -baseline_epsilon 3.1 -batch_size 128 -threat_model WHITE_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data taobao -pad_max_len 64 -train_time_attack PGD -baseline_epsilon 3.1 -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128

# MI-FGSM
python Main.py -data taobao -pad_max_len 64 -train_time_attack MI_FGSM -momentum_decay_mu 0.4 -baseline_epsilon 3.2 -batch_size 128 -threat_model WHITE_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data taobao -pad_max_len 64 -train_time_attack MI_FGSM -momentum_decay_mu 0.4 -baseline_epsilon 3.2 -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128

# RTS-D
python Main.py -data taobao -pad_max_len 64 -train_time_attack TS_DET -kappa 50 -baseline_epsilon 3.5 -batch_size 128 -threat_model WHITE_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data taobao -pad_max_len 64 -train_time_attack TS_DET -kappa 50 -baseline_epsilon 3.5 -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128

# RTS-P
python Main.py -data taobao -pad_max_len 64 -train_time_attack TS_PROB -kappa 65 -sparse_hidden 1600 -batch_norm -batch_size 128 -threat_model WHITE_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data taobao -pad_max_len 64 -train_time_attack TS_PROB -kappa 65 -sparse_hidden 1600 -batch_norm -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/taobao_fold1_CLEAN_TARGET_Apr-26-2024_172736/thp_model_20.pkl -defender_src_path ./CKPT/taobao_fold1_CLEAN_SOURCE_Apr-26-2024_172946/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -standardize -std_scale 1e-4 -log_path <path to log file> -std_error_subtractor 128

```

2. Health
```
# PermTPP
python Main.py -data Health -pad_max_len 512 -train_time_attack OUR -batch_size 16 -epoch 20 -threat_model WHITE_BOX_SOURCE -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 16
python Main.py -data Health -pad_max_len 512 -train_time_attack OUR -batch_size 16 -epoch 20 -threat_model BLACK_BOX -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 16

# PGD
python Main.py -data Health -pad_max_len 512 -train_time_attack PGD -baseline_epsilon 4.2 -batch_size 16 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16
python Main.py -data Health -pad_max_len 512 -train_time_attack PGD -baseline_epsilon 5.2 -batch_size 16 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16

# MI-FGSM
python Main.py -data Health -pad_max_len 512 -train_time_attack MI_FGSM -momentum_decay_mu 0.5 -baseline_epsilon 4.6 -batch_size 16 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16
python Main.py -data Health -pad_max_len 512 -train_time_attack MI_FGSM -momentum_decay_mu 0.6 -baseline_epsilon 5.4 -batch_size 16 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16

# RTS-D
python Main.py -data Health -pad_max_len 512 -train_time_attack TS_DET -baseline_epsilon 5 -kappa 55 -batch_size 64 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16
python Main.py -data Health -pad_max_len 512 -train_time_attack TS_DET -baseline_epsilon 5.6 -kappa 55 -batch_size 16 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16

# RTS-P
python Main.py -data Health -pad_max_len 512 -train_time_attack TS_PROB -kappa 65 -batch_size 30 -sparse_hidden 275 -batch_norm -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16
python Main.py -data Health -pad_max_len 512 -train_time_attack TS_PROB -kappa 20 -batch_size 16 -sparse_hidden 100 -batch_norm -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Health_fold1_CLEAN_TARGET_Sep-07-2023_232109/thp_model_20.pkl -defender_src_path ./CKPT/Health_fold1_CLEAN_SOURCE_Sep-08-2023_003936/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 16
```

3. Electricity
```
# PermTPP
python Main.py -data Electricity -pad_max_len 300 -train_time_attack OUR -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Electricity -pad_max_len 300 -train_time_attack OUR -batch_size 128 -epoch 20 -threat_model BLACK_BOX -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 128

# PGD
python Main.py -data Electricity -pad_max_len 300 -train_time_attack PGD -baseline_epsilon 2.9 -batch_size 128 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Electricity -pad_max_len 300 -train_time_attack PGD -baseline_epsilon 2 -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128

# MI-FGSM
python Main.py -data Electricity -pad_max_len 300 -train_time_attack MI_FGSM -momentum_decay_mu 0.5 -baseline_epsilon 2.7 -batch_size 128 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Electricity -pad_max_len 300 -train_time_attack MI_FGSM -momentum_decay_mu 0.5 -baseline_epsilon 2.1 -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128

# RTS-D
python Main.py -data Electricity -pad_max_len 300 -train_time_attack TS_DET -baseline_epsilon 3.7 -kappa 65 -batch_size 128 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Electricity -pad_max_len 300 -train_time_attack TS_DET -baseline_epsilon 4.6 -kappa 70 -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128

# RTS-P
python Main.py -data Electricity -pad_max_len 300 -train_time_attack TS_PROB -sparse_hidden 64 -kappa 60 -epoch 20 -batch_norm -batch_size 128 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Electricity -pad_max_len 300 -train_time_attack TS_PROB -sparse_hidden 50 -epoch 20 -kappa 60 -batch_norm -batch_size 128 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Electricity_fold1_CLEAN_TARGET_Sep-11-2023_165512/thp_model_20.pkl -defender_src_path ./CKPT/Electricity_fold1_CLEAN_SOURCE_Sep-11-2023_165525//thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log files> -std_error_subtractor 128
```

4. Twitter
```
# PermTPP
python Main.py -data Twitter -pad_max_len 265 -train_time_attack OUR -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Twitter -pad_max_len 265 -train_time_attack OUR -batch_size 128 -epoch 20 -threat_model BLACK_BOX -noise_model NOISE_TRANSFORMER_V2 -train_mode ADV_LLH_DIAG -attack_reg NONE -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -batch_norm -log_path <path to log file> -std_error_subtractor 128


# PGD
python Main.py -data Twitter -pad_max_len 265 -train_time_attack PGD -baseline_epsilon 1.7 -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Twitter -pad_max_len 265 -train_time_attack PGD -baseline_epsilon 2.1 -batch_size 128 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128

# MI-FGSM
python Main.py -data Twitter -pad_max_len 265 -train_time_attack MI_FGSM -momentum_decay_mu 0.4 -baseline_epsilon 1.8 -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Twitter -pad_max_len 265 -train_time_attack MI_FGSM -momentum_decay_mu 0.4 -baseline_epsilon 2.3 -batch_size 128 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128

# RTS-D
python Main.py -data Twitter -pad_max_len 265 -train_time_attack TS_DET -baseline_epsilon 1.9 -kappa 40 -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Twitter -pad_max_len 265 -train_time_attack TS_DET -baseline_epsilon 2.3 -kappa 40 -batch_size 128 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128

# RTS-P
python Main.py -data Twitter -pad_max_len 265 -train_time_attack TS_PROB -sparse_hidden 300 -batch_norm -kappa 40 -batch_size 128 -epoch 20 -threat_model WHITE_BOX_SOURCE -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
python Main.py -data Twitter -pad_max_len 265 -train_time_attack TS_PROB -sparse_hidden 350 -batch_norm -kappa 40 -batch_size 128 -epoch 20 -threat_model BLACK_BOX -defender_tgt_path ./CKPT/Twitter_fold1_CLEAN_TARGET_Sep-07-2023_163819/thp_model_20.pkl -defender_src_path ./CKPT/Twitter_fold1_CLEAN_SOURCE_Sep-07-2023_163811/thp_model_20.pkl -d_k 128 -d_v 128 -n_head 8 -log_path <path to log file> -std_error_subtractor 128
```

## Citing
If you use this code in your research, please cite:
```
@inproceedings{aaai25,
 author = {Pritish Chakraborty and Vinayak Gupta and Rahul R and Srikanta Bedathur and Abir De},
 booktitle = {Proc. of the 39th AAAI Conference on Artificial Intelligence (AAAI)},
 title = {Differentiable Adversarial Attacks for Marked Temporal Point Processes},
 year = {2025}
}
```

## Contact
In case of any issues, please reach out at:
```pritish (at) cse.iitb.ac.in``` or ```guptavinayak51 (at) gmail.com```
