import argparse
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RS = 42
SHUFFLE = True
UNKNOWN_GROUPS = ("unknown",)


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
        "--all_species_in_all_folds",
        default=False,
        action="store_true",
        help="Whether to include all species in all folds",
    )

    args = parser.parse_args()

    return args


def stratified_k_fold(X, y, g, k):
    return StratifiedGroupKFold(n_splits=k, shuffle=SHUFFLE, random_state=RS).split(X, y, g)


if __name__ == "__main__":
    args = collect_args()

    print(f"Received args: {args}")

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

    if args.all_species_in_all_folds:
        assert list(df.index) == list(range(len(df))), "Index should be arranged"

        all_unique_birds = set(df[args.split_col])
        split = [list(el) for el in split]
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
        split = [tuple(el) for el in split]

        # Sanity check
        for train_idx, _ in split:
            assert set(df[args.split_col].iloc[split[fold_i][0]]) == all_unique_birds

    split = np.array(split, dtype=object)

    np.save(save_path, split)

    print("Done!")
