from code_base.models import WaveCNNAttenClasifier

EXP_NAME = "tf_efficientnetv2_s_in21k_Exp_noamp_64bs_5sec_BasicAug_EqualBalancing_AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromPrebs1_PseudoF2PT05MT01P04I2MOOFRev_AddRareBirdsNoLeak"
TRAIN_PERIOD = 5

CONFIG = {
    # Inference Class
    "use_sigmoid": False,
    "aggregate_preds": True,
    # Data config
    "train_df_path": "data/train_and_prev_comps_extendedv1_pruneSL_XConly2025_snipet28032025_hdf5_fixedaudiometa_h5pyDur.csv",
    "split_path": "data/cv_split_base_and_prev_comps_XCsnipet28032025_group_allbirds_hdf5.npy",
    "n_folds": 5,
    "train_data_root": "data/train_audio",
    "label_map_data_path": "data/bird2int_2025.json",
    "scored_birds_path": "data/sb_2025.json",
    "lookback": None,
    "lookahead": None,
    "segment_len": 5,
    "step": None,
    "late_normalize": True,
    "add_dataset_config": {
        "filename_change_mapping": {
            "base": "train_audio",
            "train_audio": "train_audio",
            "add_train_audio_from_prev_comps": "add_train_audio_from_prev_comps",
            "add_train_audio_from_xeno_canto_28032025": "add_train_audio_from_xeno_canto_28032025",
        },
        "ignore_setting_dataset_value": True,
        "duration_col": "duration_s_h5py",
    },
    # Model config
    "exp_name": EXP_NAME,
    "model_class": WaveCNNAttenClasifier,
    "model_config": dict(
        backbone="tf_efficientnetv2_s_in21k",
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
            "num_class": 206,
            "train_period": TRAIN_PERIOD,
            "infer_period": TRAIN_PERIOD,
            "output_type": "clipwise_pred_long",
        },
        exportable=True,
        fixed_amplitude_to_db=True,
    ),
    "chkp_name": "last.ckpt",
    "swa_checkpoint_regex": r"(?P<key>\w+)=(?P<value>[\d.]+)(?=\.ckpt|$)",
    "swa_sort_rule": lambda x: -float(x["valid_roc_auc"]),
    "delete_prefix": "model.",
    "n_swa_models": 1,
    "model_output_key": None,
    # Compilation config
    "use_openvino": True,
    "use_fp16": True,
    "folds_to_onnx": [0, 1, 2, 3, 4],
    "final_activation": None,
}
