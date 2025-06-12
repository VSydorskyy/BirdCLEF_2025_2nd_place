import argparse
import importlib.util
import os
import re
from copy import deepcopy
from pprint import pprint

import numpy as np
import pandas as pd
import torch

from code_base.datasets import WaveAllFileDataset
from code_base.inefernce import BirdsInference
from code_base.utils import load_json
from code_base.utils.main_utils import get_device
from code_base.utils.metrics import score_numpy
from code_base.utils.onnx_utils import ONNXEnsemble, convert_to_onnx
from code_base.utils.swa import avarage_weights, delete_prefix_from_chkp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to .py file with CONFIG dict")
    args = parser.parse_args()

    # Import CONFIG file
    spec = importlib.util.spec_from_file_location(name="module.name", location=args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    CONFIG = config_module.CONFIG

    TRAIN_PERIOD = CONFIG["model_config"]["head_config"]["train_period"]

    if CONFIG.get("use_sed_mode", False):
        assert CONFIG["step"] is not None
    else:
        assert CONFIG["step"] is None

    if "folds" not in CONFIG:
        CONFIG["folds"] = list(range(CONFIG["n_folds"]))

    bird2id = load_json(CONFIG["label_map_data_path"])

    df = pd.read_csv(CONFIG["train_df_path"])
    split = np.load(CONFIG["split_path"], allow_pickle=True)
    val_df = [df.iloc[split[i][1]].reset_index(drop=True) for i in CONFIG["folds"]]

    val_ds_conig = {
        "root": CONFIG["train_data_root"],
        "label_str2int_mapping_path": CONFIG["label_map_data_path"],
        "use_audio_cache": True,
        "n_cores": 64,
        "verbose": False,
        "segment_len": CONFIG["segment_len"],
        "lookback": CONFIG["lookback"],
        "lookahead": CONFIG["lookahead"],
        "sample_id": None,
        "late_normalize": CONFIG["late_normalize"],
        "step": CONFIG["step"],
        # "validate_sr": 32_000,
    }
    if CONFIG.get("add_dataset_config") is not None:
        val_ds_conig.update(CONFIG["add_dataset_config"])

    loader_config = {
        "batch_size": 64,
        "drop_last": False,
        "shuffle": False,
        "num_workers": 0,
    }

    ds_val = [WaveAllFileDataset(df=df, **val_ds_conig) for df in val_df]

    loader_val = [
        torch.utils.data.DataLoader(
            ds,
            **loader_config,
        )
        for ds in ds_val
    ]

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

    model = [
        create_model_and_upload_chkp(
            model_class=CONFIG["model_class"],
            model_config=CONFIG["model_config"],
            model_device=get_device(),
            model_chkp_root=f"logdirs/{CONFIG['exp_name']}/fold_{m_i}/checkpoints",
            model_chkp_basename=CONFIG["chkp_name"] if CONFIG["swa_checkpoint_regex"] is None else None,
            model_chkp_regex=CONFIG.get("swa_checkpoint_regex"),
            swa_sort_rule=CONFIG.get("swa_sort_rule"),
            n_swa_to_take=CONFIG.get("n_swa_models", 3),
            delete_prefix=CONFIG.get("delete_prefix"),
            prune_checkpoint_func=CONFIG.get("prune_checkpoint_func"),
        )
        for m_i in range(CONFIG["n_folds"])
    ]

    # os.environ["OMP_NUM_THREADS"]="10"

    if CONFIG.get("folds_to_onnx") is not None:
        if isinstance(CONFIG["folds_to_onnx"], list):
            print("Mutli Fold Model")
            exportable_ensem = ONNXEnsemble(
                model_class=CONFIG["model_class"],
                configs=[deepcopy(CONFIG["model_config"]) for _ in CONFIG["folds_to_onnx"]],
                final_activation=CONFIG.get("final_activation", None),
                extract_spec_ones=True,
            )
            assert len(exportable_ensem.models) == len(model)
            for model_id in range(len(exportable_ensem.models)):
                exportable_ensem.models[model_id].load_state_dict(model[model_id].state_dict())
                exportable_ensem.models[model_id].eval()
            exportable_ensem.eval()
            POSTFIX = ""
            convert_to_onnx(
                model_to_convert=exportable_ensem,
                sample_input=torch.randn(5, TRAIN_PERIOD * 32_000),
                base_path=f"logdirs/{CONFIG['exp_name']}/onnx_ensem_5first_folds" + POSTFIX,
                use_fp16=CONFIG.get("use_fp16", False),
                use_openvino=CONFIG.get("use_openvino", False),
                opset_version=12,
            )

        elif isinstance(CONFIG["folds_to_onnx"], int):
            print("Solo Fold Model")
            for fold_id in CONFIG["folds_to_onnx"]:
                exportable_ensem = ONNXEnsemble(
                    model_class=CONFIG["model_class"],
                    configs=[deepcopy(CONFIG["model_config"])],
                    final_activation=CONFIG.get("final_activation", None),
                    extract_spec_ones=True,
                )
                # for model_id in range(len(exportable_ensem.models)):
                exportable_ensem.models[0].load_state_dict(model[fold_id].state_dict())

                exportable_ensem.eval()
                convert_to_onnx(
                    model_to_convert=exportable_ensem,
                    sample_input=torch.randn(5, TRAIN_PERIOD * 32_000),
                    base_path=f"logdirs/{CONFIG['exp_name']}/onnx_ensem_fold{fold_id}",
                    use_fp16=CONFIG.get("use_fp16", False),
                    use_openvino=CONFIG.get("use_openvino", False),
                    opset_version=12,
                )

        else:
            raise ValueError(f"{type(CONFIG['folds_to_onnx'])} - unsupported type")

    inference_class = BirdsInference(
        device="cuda",
        verbose_tqdm=True,
        use_sigmoid=CONFIG["use_sigmoid"],
        model_output_key=CONFIG["model_output_key"],
        aggregate_preds=CONFIG.get("aggregate_preds", True),
    )

    val_tgts, val_preds = inference_class.predict_val_loaders(nn_models=model, data_loaders=loader_val)
    if CONFIG.get("scored_birds_path", None):
        print("Extracting scored_bird_ids indices")
        scored_bird = load_json(CONFIG["scored_birds_path"])
        scored_bird_ids = [bird2id[el] for el in scored_bird]
        val_tgts = [el[:, scored_bird_ids] for el in val_tgts]
        val_preds = [el[:, scored_bird_ids] for el in val_preds]
    cmaps = [score_numpy(gt, pr) for gt, pr in zip(val_tgts, val_preds)]

    print(f"Folds Roc Auc: {cmaps}\n" f"Mean Roc Auc: {np.mean(cmaps)}")


if __name__ == "__main__":
    main()
