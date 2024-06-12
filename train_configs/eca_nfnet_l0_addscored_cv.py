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
N_EPOCHS = 50
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
    "exp_name": "eca_nfnet_l0_Exp_FromPretrainEnc_noamp_FixedAmp2Db_64bs_5sec_TimeFlip05_Montage0til5A05_UnlabeledP05_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_FullData_NoDuplsV1",
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
            "segment_len": TRAIN_PERIOD,
            "late_normalize": LATE_NORMALIZE,
            "use_h5py": True,
            "late_aug": TimeFlip(p=0.5),
            "do_montage": True,
            "montage_params": {"montage_samples": (0, 5), "alpha": 0.5},
            "unlabeled_glob": "/home/vova/data/exps/birdclef_2024/birdclef_2024/unlabeled_soundscapes_features/*.hdf5",
            "unlabeled_params": {"mode": "mixup", "prob": 0.5, "alpha": None},
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
            "shuffle": True,
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
            backbone="eca_nfnet_l0",
            mel_spec_paramms={
                "sample_rate": 32000,
                "n_mels": 128,
                "f_min": 20,
                "n_fft": 2048,
                "hop_length": 512,
                "normalized": True,
            },
            head_config={
                "p": 0.5,
                "num_class": 188,
                "train_period": TRAIN_PERIOD,
                "infer_period": TRAIN_PERIOD,
            },
            exportable=True,
            fixed_amplitude_to_db=True,
        ),
        "optimizer_init": lambda model: torch.optim.RAdam(model.parameters(), lr=1e-3),
        "scheduler_init": lambda optimizer, len_train: get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len_train, num_training_steps=int((N_EPOCHS * len_train) * 1.3)
        ),
        "scheduler_params": {"interval": "step", "monitor": MAIN_METRIC},
        "forward": lambda: MultilabelClsForwardLongShort(
            loss_type="baseline", use_weights=False, batch_aug=None, use_bce_focal_loss=True, use_masked_loss=True
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
        "pretrain_config": {
            "backbone_path": "/home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_addones/pretrained_backbones/eca_nfnet_l0_Pretrainversion3.pth",
        },
    },
}
