import argparse
import importlib.util
import os
from os.path import join as pjoin
from shutil import copy

import numpy as np
import pandas as pd
from prompt_toolkit import prompt

from code_base.utils import write_json


def dump_code(target_path: str, filename: str):
    filename_splitted = filename.split("/")

    if len(filename_splitted) > 1:
        gen_filename = "___".join(filename_splitted)
    else:
        gen_filename = filename_splitted[0]

    copy(filename, pjoin(target_path, gen_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to .py file with CONFIG dict")
    parser.add_argument(
        "--check_exist_in_train_f",
        default=False,
        action="store_true",
        help="Check if logdir exists in train function",
    )

    args = parser.parse_args()

    # Import CONFIG file
    spec = importlib.util.spec_from_file_location(name="module.name", location=args.config)
    config_dict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_dict)
    print(f"Your config:\n{config_dict.CONFIG}")

    # Extract data
    if config_dict.CONFIG["df_path"] is not None:
        train_df = pd.read_csv(config_dict.CONFIG["df_path"])
    else:
        train_df = None
    if config_dict.CONFIG["split_path"] is not None:
        splits = np.load(config_dict.CONFIG["split_path"], allow_pickle=True)
    else:
        splits = None

    # Create logdir
    logdir_name = pjoin("logdirs", config_dict.CONFIG["exp_name"])
    logdir_exists = os.path.exists(logdir_name)
    if logdir_exists and not args.check_exist_in_train_f:
        answer = prompt("Logdir already exists. Do you want to continue (y/n)? ")
        if answer.lower() != "y":
            raise RuntimeError(f"Folder {logdir_name} already exists!")
    else:
        os.makedirs(logdir_name, exist_ok=True)
        print("Folder created!")

    if not logdir_exists:
        code_folder_path = pjoin(logdir_name, "code")
        os.makedirs(code_folder_path)
        for f in config_dict.CONFIG["files_to_save"]:
            dump_code(target_path=code_folder_path, filename=f)
        print("Code dumped!")

    seed = config_dict.CONFIG["seed"]

    # Choose OOF training or CV training or pretrain
    if isinstance(config_dict.CONFIG["folds"], int):  # OOF mode
        print("OOF training preparation")
        split = splits[config_dict.CONFIG["folds"]]
        train_train_df = train_df.iloc[split[0]]
        train_val_df = train_df.iloc[split[1]]
        print("Going into train function")
        config_dict.CONFIG["train_function"](
            train_df=train_train_df,
            val_df=train_val_df,
            exp_name=logdir_name,
            fold_id=0,
            seed=seed,
            check_exp_exists=args.check_exist_in_train_f,
            **config_dict.CONFIG["train_function_args"],
        )
        print("OOF training completed!")

    elif isinstance(config_dict.CONFIG["folds"], list):  # CV mode
        fold_status = dict()
        print("CV Training")
        for fold_num in config_dict.CONFIG["folds"]:
            print(f"Fold {fold_num} preparation")
            fold_seed = seed if isinstance(seed, int) else seed[fold_num]
            split = splits[fold_num]
            train_train_df = train_df.iloc[split[0]]
            train_val_df = train_df.iloc[split[1]]
            print("Going into train function")
            config_dict.CONFIG["train_function"](
                train_df=train_train_df,
                val_df=train_val_df,
                exp_name=pjoin(logdir_name, f"fold_{fold_num}"),
                fold_id=fold_num,
                seed=fold_seed,
                check_exp_exists=args.check_exist_in_train_f,
                **config_dict.CONFIG["train_function_args"],
            )
            fold_status[f"fold_{fold_num}"] = "Success"
            print(f"Fold {fold_num} training completed!")
        print("CV Training completed")

    elif (
        isinstance(config_dict.CONFIG["folds"], str) and config_dict.CONFIG["folds"] == "train_valid"
    ):  # Train and valid mode
        print("Train and valid training preparation")
        train_train_df = train_df[train_df["subset"] == "train"]
        train_val_df = train_df[train_df["subset"] == "valid"]
        print("Going into train function")
        config_dict.CONFIG["train_function"](
            train_df=train_train_df,
            val_df=train_val_df,
            exp_name=logdir_name,
            fold_id=None,
            seed=seed,
            check_exp_exists=args.check_exist_in_train_f,
            **config_dict.CONFIG["train_function_args"],
        )
        print("Train and valid training completed!")

    elif config_dict.CONFIG["folds"] is None:  # Train on all data
        print("Pretrain trainig")
        print("Going into train function")
        config_dict.CONFIG["train_function"](
            train_df=train_df.copy(),
            val_df=None,
            exp_name=logdir_name,
            fold_id=None,
            seed=seed,
            check_exp_exists=args.check_exist_in_train_f,
            **config_dict.CONFIG["train_function_args"],
        )
        print("Pretrain trainig completed")

    else:
        raise ValueError(f"""{type(config_dict.CONFIG['folds'])} is wrong type for `folds`""")

    print("END of main train script")
