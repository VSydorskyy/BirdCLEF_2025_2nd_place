from glob import glob
import torch

from code_base.augmentations.transforms import BackgroundNoise
from code_base.callbacks import ROC_AUC_Score
from code_base.datasets import WaveAllFileDataset, WaveDataset
from code_base.forwards import MultilabelClsForwardLongShort
from code_base.models import RandomFiltering, WaveCNNAttenClasifier
from code_base.train_functions.train_lightning import lightning_training

B_S = 64
TRAIN_PERIOD = 5.0
N_EPOCHS = 40
ROOT_PATH = "/home/vova/data/exps/birdclef_2024/birdclef_2024/train_features/"
LATE_NORMALIZE = True
MAXIMIZE_METRIC = True
MAIN_METRIC = "valid_roc_auc"
PATH_TO_JSON_MAPPING = "/home/vova/data/exps/birdclef_2024/class_mappings/bird2int_2024.json"
PRECOMPUTE = False
DEBUG = False

CONFIG = {
    "seed": 1243,
    "df_path": "/home/vova/data/exps/birdclef_2024/birdclef_2024/train_metadata_extended_noduplv1.csv",
    "split_path": "/home/vova/data/exps/birdclef_2024/cv_splits/birdclef_2024_5_folds_split_nodupl.npy",
    "exp_name": "convnext_small_fb_in22k_ft_in1k_384_Exp_noamp_64bs_5sec_PrevCompXCScoredDataNoSecLab_BackGroundSoundScapeP05_mixupP05_RandomFiltering_balancedSampler_Adamlr1e4_TailCosBatchLR1e6_Epoch40_SpecAugV1_FocalLoss_Full_NoDuplsV1",
    "files_to_save": (glob("code_base/**/*.py") + [__file__] + ["scripts/main_train.py"]),
    "folds": None,
    "train_function": lightning_training,
    "train_function_args": {
        "train_dataset_class": WaveDataset,
        "train_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": PRECOMPUTE,
            "n_cores": 32,
            "debug": DEBUG,
            "do_mixup": True,
            "mixup_params": {"prob": 0.5, "alpha": None},
            "segment_len": TRAIN_PERIOD,
            "late_normalize": LATE_NORMALIZE,
            "sampler_col": "stratify_col",
            "use_sampler": True,
            "shuffle": True,
            "use_h5py": True,
            "add_df_paths": [
                "/home/vova/data/exps/birdclef_2024/dfs/full_noduplsV3_scored_meta_prev_comps_extended_2024SecLabels.csv",
                "/home/vova/data/exps/birdclef_2024/xeno_canto/dataset_2024_classes/train_metadata_noduplV3_extended_2024SecLabels.csv"
            ],
            "filename_change_mapping": {
                "base": "birdclef_2024/train_features/",
                "comp_2021":"birdclef_2021/train_features/",
                "comp_2023":"birdclef_2023/train_features/",
                "comp_2022":"birdclef_2022/train_features/",
                "comp_2020":"birdsong_recognition/train_features/",
                "a_m_2020": "xeno_canto_bird_recordings_extended_a_m/train_features/",
                "n_z_2020": "xeno_canto_bird_recordings_extended_n_z/train_features/",
                "xc_2024_classes": "xeno_canto/dataset_2024_classes/train_features/"
            },
            "late_aug": BackgroundNoise(
                p=0.5,
                esc50_root="/home/vova/data/exps/birdclef_2024/my_2023_data/soundscapes_nocall/train_audio",
                esc50_df_path="/home/vova/data/exps/birdclef_2024/my_2023_data/soundscapes_nocall/v1_no_call_meta.csv",
                normalize=LATE_NORMALIZE,
            ),
        },
        
        "val_dataset_class": WaveAllFileDataset,
        "val_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": False,
            "n_cores": 32,
            "debug": DEBUG,
            "segment_len": 5,
            "sample_id": None,
            "late_normalize": LATE_NORMALIZE,
            "use_h5py": True,
        },
        "train_dataloader_config": {
            "batch_size": B_S,
            "shuffle": False,
            "drop_last": True,
            "num_workers": 8,
            "pin_memory": True,
        },
        "val_dataloader_config": {
            "batch_size": B_S,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 8,
            "pin_memory": True,
        },
        "nn_model_class": WaveCNNAttenClasifier,
        "nn_model_config": dict(
            backbone="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
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
                "num_class": 188,
                "train_period": TRAIN_PERIOD,
                "infer_period": TRAIN_PERIOD,
            },
            exportable=True,
        ),
        "optimizer_init": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4),
        "scheduler_init": lambda optimizer, len_train: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(N_EPOCHS*len_train*1.1), T_mult=1, eta_min=1e-6, last_epoch=-1
        ),
        "scheduler_params": {"interval": "step", "monitor": MAIN_METRIC},
        "forward": lambda: MultilabelClsForwardLongShort(
            loss_type="baseline",
            use_weights=False,
            batch_aug=RandomFiltering(
                min_db=-20, is_wave=True, normalize_wave=LATE_NORMALIZE
            ),
            use_focal_loss=True,
        ),
        "callbacks": lambda: [
            ROC_AUC_Score(
                pred_key="clipwise_pred_long",
                loader_names=("valid",),
                aggr_key="dfidx",
                use_sigmoid=False,
                label_str2int_mapping_path=PATH_TO_JSON_MAPPING,
                scored_bird_path="/home/vova/data/exps/birdclef_2024/scored_birds/sb_2024.json",
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
        # Possible options
        # "16-mixed", "bf16-mixed", "32-true", "64-true"
        "precision_mode": "16-mixed",
        "train_strategy": "ddp_find_unused_parameters_true",
        "n_checkpoints_to_save": 3,
        "log_every_n_steps": None,
        "debug": DEBUG,

        "label_str2int_path": PATH_TO_JSON_MAPPING,
        "class_weights_path": "/home/vova/data/exps/birdclef_2024/sample_weights/sw_2024_add_data_v1.json",
        "use_sampler": True,
    },
}