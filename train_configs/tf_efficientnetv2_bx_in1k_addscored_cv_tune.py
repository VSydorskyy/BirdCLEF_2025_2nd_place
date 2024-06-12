from glob import glob

import torch
from transformers import get_cosine_schedule_with_warmup

from code_base.augmentations.transforms import TimeFlip
from code_base.callbacks import ROC_AUC_Score
from code_base.datasets import WaveAllFileDataset, WaveDataset
from code_base.forwards import MultilabelClsForwardLongShort
from code_base.models import RandomFiltering, WaveCNNAttenClasifier
from code_base.train_functions.train_lightning import lightning_training

B_S = 64
TRAIN_PERIOD = 5.0
N_EPOCHS = 30
ROOT_PATH = "/home/vova/data/exps/birdclef_2024/birdclef_2024/train_features/"
LATE_NORMALIZE = True
MAXIMIZE_METRIC = True
MAIN_METRIC = "valid_roc_auc"
PATH_TO_JSON_MAPPING = "/home/vova/data/exps/birdclef_2024/class_mappings/bird2int_2024_PrevComp.json"
PRECOMPUTE = False
DEBUG = False
MIXUP_DIV_FACTOR = 10

CONFIG = {
    "seed": 1243,
    "df_path": "/home/vova/data/exps/birdclef_2024/birdclef_2024/merged_train_metadata_extended_noduplv2.csv",
    "split_path": "/home/vova/data/exps/birdclef_2024/cv_splits/merged_5_folds_split_noduplV1.npy",
    "exp_name": "tf_efficientnetv2_b1_in1k_Exp_noamp_FixedAmp2Db_64bs_5sec_MergedData_TuneStartF3em4_TimeFlip05_FormixupAlpha05NormedBinTgt10X_RealbalSamplWithRep_Radamlr1e3_TrCosBatchLRWarm1Ep_Epoch30_SpecAugV207_FocalBCELoss_5Folds_NoDuplsV2",
    "files_to_save": (glob("code_base/**/*.py") + [__file__] + ["scripts/main_train.py"]),
    "folds": [0, 1, 2, 3, 4],
    "train_function": lightning_training,
    "train_function_args": {
        "train_dataset_class": WaveDataset,
        "train_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": PRECOMPUTE,
            "n_cores": 32,
            "debug": DEBUG,
            "segment_len": TRAIN_PERIOD,
            "late_normalize": LATE_NORMALIZE,
            "use_h5py": True,
            "late_aug": TimeFlip(p=0.5),
            "ignore_setting_dataset_value": True,
            "filename_change_mapping": {
                "base": "birdclef_2024/train_features/",
                "comp_2021": "birdclef_2021/train_features/",
                "comp_2023": "birdclef_2023/train_features/",
                "comp_2022": "birdclef_2022/train_features/",
                "comp_2020": "birdsong_recognition/train_features/",
                "a_m_2020": "xeno_canto_bird_recordings_extended_a_m/train_features/",
                "n_z_2020": "xeno_canto_bird_recordings_extended_n_z/train_features/",
                "xc_2024_classes": "xeno_canto/dataset_2024_classes/train_features/",
            },
            "shuffle": True,
            "use_sampler": True,
            "sampler_col": "primary_label",
            "dataset_repeat": MIXUP_DIV_FACTOR,
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
            "ignore_setting_dataset_value": True,
            "filename_change_mapping": {
                "base": "birdclef_2024/train_features/",
                "comp_2021": "birdclef_2021/train_features/",
                "comp_2023": "birdclef_2023/train_features/",
                "comp_2022": "birdclef_2022/train_features/",
                "comp_2020": "birdsong_recognition/train_features/",
                "a_m_2020": "xeno_canto_bird_recordings_extended_a_m/train_features/",
                "n_z_2020": "xeno_canto_bird_recordings_extended_n_z/train_features/",
                "xc_2024_classes": "xeno_canto/dataset_2024_classes/train_features/",
            },
        },
        "train_dataloader_config": {
            "batch_size": B_S * MIXUP_DIV_FACTOR,
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
            backbone="tf_efficientnetv2_b1.in1k",
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
                    "mask_max_length": 20,
                    "mask_max_masks": 3,
                    "p": 0.7,
                    "inplace": True,
                },
                "time_mask": {
                    "mask_max_length": 30,
                    "mask_max_masks": 3,
                    "p": 0.7,
                    "inplace": True,
                },
            },
            head_config={
                "p": 0.5,
                "num_class": 998,
                "train_period": TRAIN_PERIOD,
                "infer_period": TRAIN_PERIOD,
            },
            exportable=True,
            fixed_amplitude_to_db=True,
        ),
        "optimizer_init": lambda model: torch.optim.RAdam(model.parameters(), lr=3e-4),
        "scheduler_init": lambda optimizer, len_train: get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len_train, num_training_steps=int((N_EPOCHS * len_train) * 1.2)
        ),
        "scheduler_params": {"interval": "step", "monitor": MAIN_METRIC},
        "forward": lambda: MultilabelClsForwardLongShort(
            loss_type="baseline",
            use_weights=False,
            batch_aug=None,
            use_bce_focal_loss=True,
            mixup_alpha=0.5,
            mixup_inf_norm=True,
            mixup_binarized_tgt=True,
            mixup_equal_data_w=False,
            binirize_labels=True,
            mixup_div_factor=MIXUP_DIV_FACTOR,
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
        "precision_mode": "32-true",
        "train_strategy": "ddp_find_unused_parameters_true",
        "n_checkpoints_to_save": 3,
        "log_every_n_steps": None,
        "debug": DEBUG,
        "class_weights_path": "sqrt",
        "use_sampler": True,
        "sampler_with_replacement": True,
        "pretrain_config": [
            {
                "model_chkp_root": f"logdirs/tf_efficientnetv2_b1_in1k_Exp_noamp_FixedAmp2Db_64bs_5sec_MergedData_AllPrevCompData_TimeFlip05_FormixupAlpha05NormedBinTgt10X_RealbalSamplWithRep_Radamlr1e3_CosBatchLR1e6_Epoch30_SpecAugV207_FocalBCELoss_5Folds_NoDuplsV2/fold_{fold_id}/checkpoints",
                "model_chkp_basename": None,
                "model_chkp_regex": r"(?P<key>\w+)=(?P<value>[\d.]+)(?=\.ckpt|$)",
                "swa_sort_rule": lambda x: -float(x["valid_roc_auc"]),
                "n_swa_to_take": 3,
                "delete_prefix": "model.",
            }
            for fold_id in range(5)
        ],
    },
}
