from glob import glob

import torch

from code_base.augmentations.transforms import BackgroundNoise, OneOf
from code_base.callbacks import ROC_AUC_Score
from code_base.datasets import WaveAllFileDataset, WaveDataset
from code_base.forwards import MultilabelClsForwardLongShort
from code_base.models import RandomFiltering, WaveCNNAttenClasifier
from code_base.train_functions.train_lightning import lightning_training

B_S = 64
TRAIN_PERIOD = 5.0
N_EPOCHS = 50
ROOT_PATH = "/gpfs/space/projects/BetterMedicine/volodymyr1/exps/bird_clef_2025/xeno_canto/train_audio"
LATE_NORMALIZE = True
MAXIMIZE_METRIC = True
MAIN_METRIC = "valid_roc_auc"
PATH_TO_JSON_MAPPING = "/gpfs/space/projects/BetterMedicine/volodymyr1/exps/bird_clef_2025/birdclef_2025/bird2int_pretraintrain_prev_comps_xc_alltaxonomy_nosmall10sp_and_2025.json"
PRECOMPUTE = False
REPLACE_PATHES = ("pretrain_audio", "pretrain_features")
DEBUG = False
N_CORES = 12

CONFIG = {
    "seed": 1243,
    "df_path": "/gpfs/space/projects/BetterMedicine/volodymyr1/exps/bird_clef_2025/xeno_canto/data_all_years_and_more/dataset/train_and_prev_comps_extendedv1_pruneSL_XCallyearstaxonomy_snipet03042025_hdf5_nosmall10sp_no2025.csv",
    "split_path": "/gpfs/space/projects/BetterMedicine/volodymyr1/exps/bird_clef_2025/xeno_canto/data_all_years_and_more/dataset/cv_split_20_folds_train_and_prev_comps_extendedv1_pruneSL_XCallyearstaxonomy_snipet03042025_hdf5_nosmall10sp_no2025.npy",
    "exp_name": "tf_efficientnetv2_s_in21k_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_AdamW2d5e4_CosBatchLR1e6_Epoch50_BackGroundSoundScapeORESC50P05_SpecAugV1_FocalBCELoss_Full_PretrainPrevCompSp10_XCAllTaxonomy",
    "files_to_save": (glob("code_base/**/*.py") + [__file__] + ["scripts/main_train.py"]),
    "folds": [0],
    "train_function": lightning_training,
    "train_function_args": {
        "train_dataset_class": WaveDataset,
        "train_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": PRECOMPUTE,
            "n_cores": N_CORES,
            "debug": DEBUG,
            "do_mixup": True,
            "mixup_params": {"prob": 0.5, "alpha": None},
            "segment_len": TRAIN_PERIOD,
            "late_normalize": LATE_NORMALIZE,
            "sampler_col": "primary_label",
            "use_sampler": True,
            "shuffle": True,
            "use_h5py": True,
            "replace_pathes": REPLACE_PATHES,
            "filename_change_mapping": {
                "base": "train_audio",
                "pretrain_audio_from_prev_comps": "pretrain_audio_from_prev_comps",
                "add_pretrain_audio_from_xeno_canto_03042025": "add_pretrain_audio_from_xeno_canto_03042025",
            },
            "ignore_setting_dataset_value": True,
            "check_all_files_exist": False,
            "late_aug": OneOf(
                [
                    BackgroundNoise(
                        p=0.5,
                        esc50_root="data/soundscapes_nocall/train_audio",
                        esc50_df_path="data/v1_no_call_meta.csv",
                        normalize=LATE_NORMALIZE,
                        precompute=False,
                    ),
                    BackgroundNoise(
                        p=0.5,
                        esc50_root="data/esc50/esc50/audio",
                        esc50_df_path="data/esc50_background.csv",
                        esc50_cats_to_include=[
                            "dog",
                            "rain",
                            "insects",
                            "hen",
                            "engine",
                            "hand_saw",
                            "pig",
                            "rooster",
                            "sea_waves",
                            "cat",
                            "crackling_fire",
                            "thunderstorm",
                            "chainsaw",
                            "train",
                            "sheep",
                            "wind",
                            "footsteps",
                            "frog",
                            "cow",
                            "crickets",
                        ],
                        normalize=LATE_NORMALIZE,
                    ),
                ]
            ),
        },
        "val_dataset_class": WaveAllFileDataset,
        "val_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": False,
            "n_cores": N_CORES,
            "debug": DEBUG,
            "segment_len": 5,
            "sample_id": None,
            "late_normalize": LATE_NORMALIZE,
            "use_h5py": True,
            "replace_pathes": REPLACE_PATHES,
            "filename_change_mapping": {
                "base": "train_audio",
                "pretrain_audio_from_prev_comps": "pretrain_audio_from_prev_comps",
                "add_pretrain_audio_from_xeno_canto_03042025": "add_pretrain_audio_from_xeno_canto_03042025",
            },
            "ignore_setting_dataset_value": True,
            "check_all_files_exist": False,
        },
        "train_dataloader_config": {
            "batch_size": B_S,
            "shuffle": False,
            "drop_last": True,
            "num_workers": N_CORES,
            "pin_memory": True,
        },
        "val_dataloader_config": {
            "batch_size": B_S,
            "shuffle": False,
            "drop_last": False,
            "num_workers": N_CORES,
            "pin_memory": True,
        },
        "nn_model_class": WaveCNNAttenClasifier,
        "nn_model_config": dict(
            backbone="tf_efficientnetv2_s_in21k",
            mel_spec_paramms={
                "sample_rate": 32000,
                "n_mels": 128,
                "f_min": 20,
                "n_fft": 2048,
                "hop_length": 512,
                "normalized": True,
            },
            spec_augment_config={
                "freq_mask": {
                    "mask_max_length": 10,
                    "mask_max_masks": 3,
                    "p": 0.3,
                    "inplace": True,
                },
                "time_mask": {
                    "mask_max_length": 20,
                    "mask_max_masks": 3,
                    "p": 0.3,
                    "inplace": True,
                },
            },
            head_config={
                "p": 0.5,
                "hidden_chans": 1024,
                "num_class": 7544,
                "train_period": TRAIN_PERIOD,
                "infer_period": TRAIN_PERIOD,
            },
            exportable=True,
            fixed_amplitude_to_db=True,
        ),
        "optimizer_init": lambda model: torch.optim.AdamW(model.parameters(), lr=2.5e-4, eps=1e-8, betas=(0.9, 0.999)),
        "scheduler_init": lambda optimizer, len_train: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=N_EPOCHS * len_train, T_mult=1, eta_min=1e-6, last_epoch=-1
        ),
        "scheduler_params": {"interval": "step", "monitor": MAIN_METRIC},
        "forward": lambda: MultilabelClsForwardLongShort(
            loss_type="baseline",
            use_weights=False,
            batch_aug=RandomFiltering(min_db=-20, is_wave=True, normalize_wave=LATE_NORMALIZE),
            use_bce_focal_loss=True,
        ),
        "callbacks": lambda: [
            ROC_AUC_Score(
                pred_key="clipwise_pred_long",
                loader_names=("valid",),
                aggr_key="dfidx",
                use_sigmoid=False,
                label_str2int_mapping_path=PATH_TO_JSON_MAPPING,
                scored_bird_path="/gpfs/space/projects/BetterMedicine/volodymyr1/exps/bird_clef_2025/birdclef_2025/sb_pretraintrain_prev_comps_xc_alltaxonomy_speciesfrom100.json",
            )
        ],
        "n_epochs": N_EPOCHS,
        "main_metric": MAIN_METRIC,
        "metric_mode": "max" if MAXIMIZE_METRIC else "min",
        "checkpoint_callback_params": dict(
            save_last=True,
            auto_insert_metric_name=True,
            save_weights_only=True,
            save_on_train_epoch_end=True,
            filename="{epoch}-{step}-{valid_roc_auc:.3f}",
        ),
        "wandb_logger_params": {"project": "birdclef_2025", "id": None, "log_model": False},
        # Possible options
        # "16-mixed", "bf16-mixed", "32-true", "64-true"
        "precision_mode": "32-true",
        "train_strategy": "ddp_find_unused_parameters_true",
        "n_checkpoints_to_save": 3,
        "log_every_n_steps": None,
        "debug": DEBUG,
    },
}
