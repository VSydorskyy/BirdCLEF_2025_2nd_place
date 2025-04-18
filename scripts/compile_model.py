import os
import re
from copy import deepcopy
from itertools import chain
from pprint import pprint

import torch

from code_base.models import WaveCNNAttenClasifier, WaveCNNClasifier
from code_base.utils.onnx_utils import ONNXEnsemble, convert_to_onnx
from code_base.utils.swa import avarage_weights, delete_prefix_from_chkp


def create_model_and_upload_chkp(
    model_class,
    model_config,
    model_device,
    model_chkp_root,
    model_chkp_basename=None,
    model_chkp_regex=None,
    delete_prefix=None,
    swa_sort_rule=None,
    n_swa_to_take=3,
    prune_checkpoint_func=None,
):
    if model_chkp_basename is None:
        basenames = os.listdir(model_chkp_root)
        checkpoints = []
        for el in basenames:
            matches = re.findall(model_chkp_regex, el)
            if not matches:
                continue
            parsed_dict = {key: value for key, value in matches}
            parsed_dict["name"] = el
            checkpoints.append(parsed_dict)
        print("SWA checkpoints")
        pprint(checkpoints)
        checkpoints = sorted(checkpoints, key=swa_sort_rule)
        checkpoints = checkpoints[:n_swa_to_take]
        print("SWA sorted checkpoints")
        pprint(checkpoints)
        if len(checkpoints) > 1:
            checkpoints = [
                torch.load(os.path.join(model_chkp_root, el["name"]), map_location="cpu")["state_dict"]
                for el in checkpoints
            ]
            t_chkp = avarage_weights(nn_weights=checkpoints, delete_prefix=delete_prefix)
        else:
            chkp_path = os.path.join(model_chkp_root, checkpoints[0]["name"])
            print("vanilla model")
            print("Loading", chkp_path)
            t_chkp = torch.load(chkp_path, map_location="cpu")["state_dict"]
            if delete_prefix is not None:
                t_chkp = delete_prefix_from_chkp(t_chkp, delete_prefix)
    else:
        chkp_path = os.path.join(model_chkp_root, model_chkp_basename)
        print("vanilla model")
        print("Loading", chkp_path)
        t_chkp = torch.load(chkp_path, map_location="cpu")["state_dict"]
        if delete_prefix is not None:
            t_chkp = delete_prefix_from_chkp(t_chkp, delete_prefix)

    if prune_checkpoint_func is not None:
        t_chkp = prune_checkpoint_func(t_chkp)
    t_model = model_class(**model_config, device=model_device)
    print("Missing keys: ", set(t_model.state_dict().keys()) - set(t_chkp))
    print("Extra keys: ", set(t_chkp) - set(t_model.state_dict().keys()))
    t_model.load_state_dict(t_chkp, strict=False)
    t_model.eval()
    return t_model


def main():
    MODEL_CLASS = WaveCNNAttenClasifier
    TRAIN_PERIOD = 5

    MODELS = [
        {
            "model_config": dict(
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
                    "num_class": 206,
                    "train_period": TRAIN_PERIOD,
                    "infer_period": TRAIN_PERIOD,
                    "output_type": "clipwise_pred_long",
                },
                exportable=True,
                fixed_amplitude_to_db=True,
            ),
            "exp_name": "eca_nfnet_l0_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_BackGroundSoundScapeORESC50P05_SpecAugV1_FocalBCELoss_5Folds_ScoredPrevCompsAndXCsnipet28032025",
            "fold": [0, 1, 2, 3, 4],
            "chkp_name": "last.ckpt",
            "swa_checkpoint_regex": r"(?P<key>\w+)=(?P<value>[\d.]+)(?=\.ckpt|$)",
            "swa_sort_rule": lambda x: -float(x["valid_roc_auc"]),
            "delete_prefix": "model.",
            "n_swa_models": 1,
            "model_output_key": None,
            # "prune_checkpoint_func": prune_checkpoint_rule
        },
        {
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
            "exp_name": "tf_efficientnetv2_s_in21k_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_BackGroundSoundScapeORESC50P05_SpecAugV1_FocalBCELoss_5Folds_ScoredPrevCompsAndXCsnipet28032025",
            "fold": [0, 1, 2, 3, 4],
            "chkp_name": "last.ckpt",
            "swa_checkpoint_regex": r"(?P<key>\w+)=(?P<value>[\d.]+)(?=\.ckpt|$)",
            "swa_sort_rule": lambda x: -float(x["valid_roc_auc"]),
            "delete_prefix": "model.",
            "n_swa_models": 1,
            "model_output_key": None,
            # "prune_checkpoint_func": prune_checkpoint_rule
        },
    ]

    model = []
    for config in MODELS:
        model.extend(
            [
                create_model_and_upload_chkp(
                    model_class=MODEL_CLASS,
                    model_config=config["model_config"],
                    model_device="cpu",
                    model_chkp_root=f"logdirs/{config['exp_name']}/fold_{m_i}/checkpoints",
                    model_chkp_basename=config["chkp_name"] if config["swa_checkpoint_regex"] is None else None,
                    model_chkp_regex=config.get("swa_checkpoint_regex"),
                    swa_sort_rule=config.get("swa_sort_rule"),
                    n_swa_to_take=config.get("n_swa_models", 3),
                    delete_prefix=config.get("delete_prefix"),
                    prune_checkpoint_func=config.get("prune_checkpoint_func"),
                )
                for m_i in config["fold"]
            ]
        )

    exportable_ensem = ONNXEnsemble(
        model_class=MODEL_CLASS,
        configs=list(chain(*[[deepcopy(config["model_config"]) for _ in config["fold"]] for config in MODELS])),
    )
    assert len(exportable_ensem.models) == len(model)
    for model_id in range(len(model)):
        exportable_ensem.models[model_id].load_state_dict(model[model_id].state_dict())
        exportable_ensem.models[model_id].eval()
    exportable_ensem.eval()
    convert_to_onnx(
        model_to_convert=exportable_ensem,
        sample_input=torch.randn(5, TRAIN_PERIOD * 32_000),
        base_path="logdirs/tf_efficientnetv2_s_in21k_and_eca_nfnet_l0_with_add_data_and_background_noise/onnx_ensem_5first_folds",
        use_fp16=True,
        use_openvino=True,
        opset_version=12,
    )


if __name__ == "__main__":
    main()
