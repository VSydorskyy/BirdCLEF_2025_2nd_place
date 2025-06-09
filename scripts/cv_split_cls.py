import argparse
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RS = 42
SHUFFLE = True
UNKNOWN_GROUPS = ("unknown",)
RARE_SPECIES_COUNT = 10


def collect_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("df_path", type=str, help="Path to train dataframe")
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to .npy file to save split",
        default="",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        help="Name of stratification column",
        default="primary_label",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        help="Name of stratification column",
        default="author",
    )
    parser.add_argument("--n_folds", type=int, help="Number of CV folds", default=5)
    parser.add_argument(
        "--all_species_in_basic_folds",
        default=False,
        action="store_true",
        help="Whether to include all species in basic folds",
    )
    parser.add_argument(
        "--all_species_in_train_folds",
        default=False,
        action="store_true",
        help="Whether to include all species in train folds",
    )
    parser.add_argument(
        "--duplicate_rare_species_in_train_folds",
        default=False,
        action="store_true",
        help="Whether to duplicate rare species in train folds",
    )
    parser.add_argument(
        "--remove_duplicates_from_val",
        default=False,
        action="store_true",
        help="Whether to remove duplicates from validation set. Should be used with --all_species_in_train_folds or --duplicate_rare_species_in_train_folds",
    )
    parser.add_argument(
        "--extra_rare_birds",
        type=str,
        nargs="*",
        default=[],
        help="List of extra classes to treat as rare birds (space separated)",
    )

    args = parser.parse_args()

    return args


def stratified_k_fold(X, y, g, k):
    return StratifiedGroupKFold(n_splits=k, shuffle=SHUFFLE, random_state=RS).split(X, y, g)


if __name__ == "__main__":
    args = collect_args()

    print(f"Received args: {args}")

    assert not (
        args.all_species_in_train_folds
        and args.duplicate_rare_species_in_train_folds
        and args.all_species_in_basic_folds
    ), "Cannot use all_species_in_train_folds and duplicate_rare_species_in_train_folds and all_species_in_basic_folds at the same time"
    assert not (
        args.remove_duplicates_from_val
        and not (args.all_species_in_train_folds or args.duplicate_rare_species_in_train_folds)
    ), "Option --remove_duplicates_from_val should be used with --all_species_in_train_folds or --duplicate_rare_species_in_train_folds"

    if args.save_path == "":
        save_path = pjoin(os.path.dirname(args.df_path), "cv_split.npy")
    else:
        save_path = args.save_path

    df = pd.read_csv(args.df_path)

    df["group"] = df[args.group_col].str.lower()
    df.loc[df["group"].isin(UNKNOWN_GROUPS), "group"] = df.loc[df["group"].isin(UNKNOWN_GROUPS), "filename"]

    split = list(stratified_k_fold(X=df, y=df[args.split_col], g=df["group"], k=args.n_folds))

    # Sanity check
    for train_idx, val_idx in split:
        assert not set(df["filename"].iloc[val_idx]) & set(df["filename"].iloc[train_idx])
        assert not set(df["group"].iloc[val_idx]) & set(df["group"].iloc[train_idx])

    if args.all_species_in_train_folds or args.duplicate_rare_species_in_train_folds or args.all_species_in_basic_folds:
        assert list(df.index) == list(range(len(df))), "Index should be arranged"

        all_unique_birds = set(df[args.split_col])
        split = [list(el) for el in split]
        if args.all_species_in_basic_folds:
            basic_folds = [fold[1] for fold in split]
            for fold_i in range(len(basic_folds)):
                fold_df = df.iloc[basic_folds[fold_i]]
                missing_birds = all_unique_birds - set(fold_df[args.split_col])
                if missing_birds:
                    print(f"Fold {fold_i} missing species: {missing_birds}")
                    index_to_add = (
                        df.loc[df[args.split_col].isin(missing_birds)]
                        .sample(frac=1)
                        .drop_duplicates(args.split_col)
                        .index.to_list()
                    )
                    print("Adding next filenames:\n", df["filename"].iloc[index_to_add])
                    basic_folds[fold_i] = np.concatenate([basic_folds[fold_i], np.array(index_to_add)])
            split = []
            for fold_i in range(len(basic_folds)):
                split.append(
                    (
                        np.concatenate([basic_folds[i] for i in range(len(basic_folds)) if i != fold_i]),
                        basic_folds[fold_i],
                    )
                )
        if args.all_species_in_train_folds:
            for fold_i in range(len(split)):
                fold_train = df.iloc[split[fold_i][0]]
                missing_birds = all_unique_birds - set(fold_train[args.split_col])
                if missing_birds:
                    print(f"Fold {fold_i} missing species: {missing_birds}")
                    index_to_add = (
                        df.loc[df[args.split_col].isin(missing_birds)]
                        .sample(frac=1)
                        .drop_duplicates(args.split_col)
                        .index.to_list()
                    )
                    print("Adding next filenames:\n", df["filename"].iloc[index_to_add])
                    split[fold_i][0] = np.concatenate([split[fold_i][0], np.array(index_to_add)])
                    if args.remove_duplicates_from_val:
                        split[fold_i][1] = np.array([idx for idx in split[fold_i][1] if idx not in index_to_add])
        if args.duplicate_rare_species_in_train_folds:
            birds_vc = df[args.split_col].value_counts()
            rare_birds = set(birds_vc[birds_vc <= RARE_SPECIES_COUNT].index)
            if args.extra_rare_birds:
                rare_birds = rare_birds.union(set(args.extra_rare_birds))
            rare_bird_ids = df[df[args.split_col].isin(rare_birds)].index
            for fold_i in range(len(split)):
                add_ids = list(set(rare_bird_ids) - set(split[fold_i][0]))
                if len(add_ids) > 0:
                    print(f"Adding next filenames to {fold_i}:\n", df["filename"].iloc[add_ids])
                    split[fold_i][0] = np.concatenate([split[fold_i][0], np.array(add_ids)])
                    if args.remove_duplicates_from_val:
                        split[fold_i][1] = np.array([idx for idx in split[fold_i][1] if idx not in add_ids])
        split = [tuple(el) for el in split]

        # Sanity check
        for train_idx, _ in split:
            assert set(df[args.split_col].iloc[train_idx]) == all_unique_birds, all_unique_birds - set(
                df[args.split_col].iloc[train_idx]
            )
        if args.remove_duplicates_from_val:
            for train_idx, val_idx in split:
                assert not set(df["filename"].iloc[val_idx]) & set(df["filename"].iloc[train_idx])

    split = np.array(split, dtype=object)

    np.save(save_path, split)

    print("Done!")
