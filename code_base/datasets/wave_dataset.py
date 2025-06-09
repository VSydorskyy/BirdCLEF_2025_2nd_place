import gc
import math
import os
import warnings
from glob import glob
from os.path import join as pjoin
from os.path import relpath, splitext
from time import time

import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..augmentations.transforms import BackgroundNoise
from ..utils import load_json, load_pp_audio, parallel_librosa_load
from ..utils.main_utils import create_oversampled_df_for_class, sample_uniformly_np_index, standardize_audio_filters

tqdm.pandas()

DEFAULT_TARGET = 0
EPS = 1e-5
MAX_RATING = 5.0


def get_curate_id(filename, with_extension=False):
    if with_extension:
        return "/".join(filename.split("/")[-2:])
    else:
        return "/".join(os.path.splitext(filename)[0].split("/")[-2:])


def get_weighted_probability_vector(
    start_second: float, pseudo_probs: np.ndarray, chunk_duration: int = 5, rounding_precision: int = 3
) -> np.ndarray:
    """
    Calculate a weighted probability vector for a given start time within a pseudo-probabilities array.

    Args:
        start_second (float): The starting second of the audio segment.
        pseudo_probs (np.ndarray): A 2D array where each row represents probabilities for a chunk.
        chunk_duration (int, optional): Duration of each chunk in seconds. Defaults to 5.
        rounding_precision (int, optional): Precision for rounding checks. Defaults to 3.

    Returns:
        np.ndarray: A weighted probability vector for the given start time.

    Raises:
        RuntimeError: If the start time exceeds the length of the pseudo-probabilities array.
    """
    # Calculate the ID of the first chunk based on the start time
    first_chunk_id = int(start_second // chunk_duration)

    # Calculate the duration of the first chunk that overlaps with the start time
    first_chunk_duration = ((first_chunk_id + 1) * chunk_duration) - start_second
    assert first_chunk_duration <= chunk_duration, "First chunk duration exceeds chunk duration"

    # Calculate the weight of the first chunk
    first_chunk_weight = first_chunk_duration / chunk_duration

    # If the first chunk is the last one, return its probabilities
    if (first_chunk_id + 1) == pseudo_probs.shape[0]:
        assert (
            round(first_chunk_duration, rounding_precision) == chunk_duration
        ), "First chunk duration does not match chunk duration"
        return pseudo_probs[first_chunk_id]

    # If the first chunk ID exceeds the array length, raise an error
    elif (first_chunk_id + 1) > pseudo_probs.shape[0]:
        raise RuntimeError("Run out of pseudo_probs length")

    # Otherwise, calculate the weighted sum of probabilities from the first and second chunks
    else:
        return pseudo_probs[first_chunk_id] * first_chunk_weight + pseudo_probs[first_chunk_id + 1] * (
            1 - first_chunk_weight
        )


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        label_str2int_mapping_path,
        replace_pathes=None,
        df=None,
        add_df_paths=None,
        filename_change_mapping=None,
        target_col="primary_label",
        sec_target_col="secondary_labels",
        name_col="filename",
        duration_col="duration_s",
        timewise_col=None,
        timewise_slice_pad=2.5,
        rating_col=None,
        sample_rate=32_000,
        segment_len=5.0,
        precompute=False,
        early_aug=None,
        late_aug=None,
        do_mixup=False,
        mixup_params={"prob": 0.5, "alpha": 1.0},
        do_montage=False,
        montage_params={"montage_samples": (0, 5), "alpha": 0.5},
        n_cores=None,
        debug=False,
        df_filter_rule=None,
        do_noisereduce=False,
        late_normalize=False,
        load_normalize=False,
        use_sampler=False,
        shuffle=False,
        res_type="soxr_hq",
        pos_dtype=None,
        sampler_col=None,
        use_h5py=False,
        empty_soundscape_config=None,
        soundscape_pseudo_df_path=None,
        soundscape_pseudo_config=None,
        soundscape_total_n_samples=9726,
        soundscape_fold_idx=None,
        oof_df_path=None,
        oof_sampling_config=None,
        start_from_zero=False,
        ignore_setting_dataset_value=False,
        dataset_repeat=1,
        unlabeled_glob=None,
        unlabeled_params={"mode": "return", "prob": 0.5, "alpha": None},
        right_bound_main=None,
        right_bound_unlabeled=None,
        check_all_files_exist=True,
        curation_json_path=None,
        curation_chunks=None,
        curation_chunks_json_path=None,
        label_smoothing=None,
        use_label_smoothing_fix=False,
        label_smoothing_treshold=0.5,
        oversample_classes_dict=None,
    ):
        super().__init__()

        assert unlabeled_params["mode"] in ["return", "mixup"]
        assert timewise_slice_pad < segment_len
        if curation_chunks_json_path is not None and curation_chunks is not None:
            raise ValueError("curation_chunks_json_path and curation_chunks can not be used together")
        if timewise_col is not None and not use_h5py:
            raise ValueError("timewise_col can be used only with h5py")
        if use_h5py and precompute:
            raise ValueError("h5py files can not be used with `precompute`")
        if use_h5py and load_normalize:
            raise RuntimeError("load_normalize is not supported with h5py")
        if df is None and add_df_paths is None:
            raise ValueError("`df` OR/AND `add_df_paths` should be defined")
        if soundscape_pseudo_df_path is not None and not use_h5py:
            raise ValueError("Soundscape pseudo df can be used only with h5py")
        if not ignore_setting_dataset_value:
            df["data_root_id"] = "base"
        if add_df_paths is not None:
            cols_to_take = [
                target_col,
                sec_target_col,
                name_col,
                "duration_s",
                "data_root_id",
                "url",
            ]
            cols_to_take = [el for el in cols_to_take if el is not None]
            if rating_col is not None and rating_col not in cols_to_take:
                cols_to_take.append(rating_col)
            if sampler_col is not None and sampler_col not in cols_to_take:
                cols_to_take.append(sampler_col)
            if timewise_col is not None and timewise_col not in cols_to_take:
                cols_to_take.append(timewise_col)
            # Create fake `df`
            if df is None:
                df = pd.DataFrame()
            else:
                df = df[cols_to_take]
            add_merged_df = pd.concat(
                [pd.read_csv(el)[cols_to_take] for el in add_df_paths],
                axis=0,
            ).reset_index(drop=True)
            add_merged_df[target_col] = add_merged_df[target_col].astype(str)
            df = pd.concat([df, add_merged_df], axis=0).reset_index(drop=True)
        if oversample_classes_dict is not None:
            add_oversampled_df = pd.concat(
                [
                    create_oversampled_df_for_class(original_df=df, class_name=class_name, n_add_samples=n_add_samples)
                    for class_name, n_add_samples in oversample_classes_dict.items()
                ],
                axis=0,
            ).reset_index(drop=True)
            df = pd.concat([df, add_oversampled_df], axis=0).reset_index(drop=True)
        if soundscape_pseudo_df_path is not None:
            if not isinstance(soundscape_pseudo_df_path, list):
                soundscape_pseudo_df_path = [soundscape_pseudo_df_path]

            for index, one_df_path in enumerate(soundscape_pseudo_df_path):
                soundscape_pseudo_df = pd.read_csv(one_df_path)
                if "data_root_id" not in soundscape_pseudo_df.columns:
                    soundscape_pseudo_df["data_root_id"] = "soundscape_" + str(index)
                # Dirty but let it be
                soundscape_pseudo_df["final_second"] = soundscape_pseudo_df["row_id"].apply(
                    lambda x: int(x.split("_")[-1])
                )
                soundscape_pseudo_df[sec_target_col] = "[]"
                soundscape_pseudo_df[name_col] = soundscape_pseudo_df["row_id"].apply(
                    lambda x: "_".join(x.split("_")[:-1]) + ".ogg"
                )
                df = pd.concat([df, soundscape_pseudo_df], axis=0).reset_index(drop=True)
            self.soundscape_pseudo_config = soundscape_pseudo_config
        if df_filter_rule is not None:
            df = df_filter_rule(df)
        if debug:
            self.df = df.iloc[:1200]
        else:
            self.df = df
        self.df = self.df.reset_index(drop=True)
        if label_str2int_mapping_path is not None:
            self.label_str2int = load_json(label_str2int_mapping_path)
            if do_montage:
                self.all_birds = set(self.label_str2int.keys())
        else:
            if do_montage:
                raise ValueError("Montage requires label_str2int_mapping_path")
            self.label_str2int = None
        try:
            self.df["secondary_labels"] = self.df["secondary_labels"].apply(eval)
        except:
            print("secondary_labels is not found in df. Maybe test or nocall mode")
        if timewise_col is not None:
            self.df[timewise_col] = self.df[timewise_col].apply(eval)
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df[f"{name_col}_with_root"] = self.df[name_col].apply(lambda x: pjoin(root, x))
        if filename_change_mapping is not None:
            self.df[f"{name_col}_with_root"] = self.df.apply(
                lambda row: row[f"{name_col}_with_root"].replace(
                    filename_change_mapping["base"], filename_change_mapping[row["data_root_id"]]
                ),
                axis=1,
            )
        if replace_pathes is not None:
            self.df[f"{name_col}_with_root"] = self.df[f"{name_col}_with_root"].apply(
                lambda x: x.replace(replace_pathes[0], replace_pathes[1])
            )

        self.duration_col = duration_col
        self.target_col = target_col
        self.sec_target_col = sec_target_col
        self.timewise_col = timewise_col
        self.name_col = f"{name_col}_with_root"
        self.rating_col = rating_col
        self.late_normalize = late_normalize
        self.empty_soundscape_config = empty_soundscape_config

        self.precompute = precompute
        self.use_h5py = use_h5py
        self.load_normalize = load_normalize

        self.df["original_name_col"] = self.df[name_col].copy()

        if self.use_h5py:
            self.df[self.name_col] = self.df[self.name_col].apply(lambda x: splitext(x)[0] + ".hdf5")

        # Validate that all files exist
        if check_all_files_exist:
            print("Validating that all files exist")
            filenames_exists = self.df[self.name_col].progress_apply(lambda x: os.path.exists(x))
            if not filenames_exists.all() and not self.use_h5py:
                print("Some files do not exist. Trying to twick extensions")
                # Try to twick extensions
                self.df.loc[~filenames_exists, self.name_col] = self.df.loc[~filenames_exists, self.name_col].apply(
                    lambda x: splitext(x)[0] + ".mp3"
                )
                filenames_exists = self.df[self.name_col].apply(lambda x: os.path.exists(x))
            if not filenames_exists.all():
                raise FileNotFoundError(
                    "Some files do not exist. Failed List:\n", self.df[~filenames_exists][self.name_col].to_list()
                )

        self.sample_rate = sample_rate
        self.do_noisereduce = do_noisereduce
        # save segment len in points (not in seconds)
        self.segment_len = int(self.sample_rate * segment_len)
        self.start_from_zero = start_from_zero
        self.timewise_slice_pad = timewise_slice_pad

        self.early_aug = early_aug
        self.late_aug = late_aug
        if mixup_params is not None and mixup_params.get("weights_path", None):
            if not use_sampler:
                raise ValueError("Mixup with weighted sampling requires `use_sampler=True`")
        self.do_mixup = do_mixup
        self.mixup_params = mixup_params
        self.do_montage = do_montage
        self.montage_params = montage_params

        self.pos_dtype = pos_dtype
        self.res_type = res_type

        self.dataset_repeat = dataset_repeat

        self.right_bound_main = right_bound_main
        self.right_bound_unlabeled = right_bound_unlabeled

        self.label_smoothing = label_smoothing
        self.use_label_smoothing_fix = use_label_smoothing_fix
        self.label_smoothing_treshold = label_smoothing_treshold

        if unlabeled_glob is not None:
            self.unlabeled_files = glob(unlabeled_glob)
        else:
            self.unlabeled_files = None
        self.unlabeled_params = unlabeled_params

        if self.precompute:
            if n_cores is not None:
                self.audio_cache = parallel_librosa_load(
                    audio_pathes=self.df[self.name_col].tolist(),
                    n_cores=n_cores,
                    return_sr=False,
                    sr=self.sample_rate,
                    do_normalize=(not self.late_normalize) and self.load_normalize,
                    do_noisereduce=do_noisereduce,
                    res_type=self.res_type,
                    pos_dtype=self.pos_dtype,
                )
                assert all(au is not None for au in self.audio_cache)
                self.audio_cache = {i: el for i, el in enumerate(self.audio_cache)}
            else:
                print("NOT Parallel load")
                self.audio_cache = {
                    # Extract only audio, without sample_rate
                    i: load_pp_audio(
                        im_name,
                        sr=self.sample_rate,
                        do_noisereduce=do_noisereduce,
                        normalize=(not self.late_normalize) and self.load_normalize,
                        res_type=self.res_type,
                        pos_dtype=self.pos_dtype,
                    )
                    for i, im_name in tqdm(
                        enumerate(self.df[self.name_col].tolist()),
                        total=len(self.df),
                    )
                }

        if soundscape_pseudo_df_path is not None:
            soundscape_dfs = [
                self.df[self.df["data_root_id"] == "soundscape_" + str(idx)].reset_index(drop=True)
                for idx in range(len(soundscape_pseudo_df_path))
            ]
            self.df = self.df[~self.df["data_root_id"].apply(lambda x: x.startswith("soundscape"))].reset_index(
                drop=True
            )

            label_int2str = {v: k for k, v in self.label_str2int.items()}
            self.indices_for_soundscapes = [label_int2str[i] for i in range(len(self.label_str2int))]
            self.soundscape_df = []
            used_sample_ids = set()
            for idx, (ss_path, ss_df) in enumerate(zip(soundscape_pseudo_df_path, soundscape_dfs)):
                assert len(ss_df.columns) >= len(self.label_str2int)
                ss_df["sample_id"] = ss_df["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
                ss_df["soundscape_path"] = ss_path
                if isinstance(self.soundscape_pseudo_config["primary_label_min_prob"], list):
                    primary_label_min_prob = self.soundscape_pseudo_config["primary_label_min_prob"][idx]
                else:
                    primary_label_min_prob = self.soundscape_pseudo_config["primary_label_min_prob"]
                ss_df = ss_df[ss_df["primary_label_prob"] > primary_label_min_prob].reset_index(drop=True)

                if self.soundscape_pseudo_config.get("use_oof", False):
                    ss_df = ss_df[ss_df["fold_id"].astype(int) != soundscape_fold_idx].reset_index(drop=True)
                else:
                    ss_df = ss_df[~ss_df["sample_id"].isin(used_sample_ids)].reset_index(drop=True)
                print(
                    f"Selected {len(set(ss_df['sample_id']))} samples from {soundscape_total_n_samples} from {ss_path}"
                )
                print(f"Selected {len(ss_df)} rows")
                used_sample_ids.update(set(ss_df["sample_id"].tolist()))
                self.soundscape_df.append(ss_df)

            self.soundscape_df = pd.concat(self.soundscape_df, axis=0).reset_index(drop=True)
            assert (
                self.soundscape_df["row_id"].value_counts().max() == 1
            ), "Some samples are duplicated in soundscape df"
            assert (
                self.soundscape_df["sample_id"].value_counts().max() <= 12
            ), "Some samples are duplicated in soundscape df"

            pseudo_probs = self.soundscape_df[list(self.label_str2int.keys())].values
            pseudo_probs[pseudo_probs < self.soundscape_pseudo_config["trim_min_prob"]] = 0
            self.soundscape_df[list(self.label_str2int.keys())] = pseudo_probs
            self.pseudo_birds = set(self.soundscape_df[self.target_col])
        else:
            self.soundscape_df = None

        if self.empty_soundscape_config is not None:
            self.empty_sampler = BackgroundNoise(**self.empty_soundscape_config["sampler_config"])

        if curation_json_path is not None:
            self.curation_json = load_json(curation_json_path)
            self.df["temp_col"] = self.df["original_name_col"].apply(get_curate_id)
            assert set(self.df["temp_col"]) <= set(self.curation_json.keys()), "Some files are not in curation json"
            ignore_ids = [key for key, value in self.curation_json.items() if (value[0] == value[1] == 0)]
            print("Curation json: Ignoring ids", ignore_ids)
            self.df = self.df[~self.df["temp_col"].isin(ignore_ids)].reset_index(drop=True)
            self.df = self.df.drop(columns=["temp_col"])
        else:
            self.curation_json = None

        self.curation_chunks = curation_chunks
        if self.curation_chunks is not None:
            self.curation_chunks = standardize_audio_filters(self.curation_chunks)
            self.df["temp_col"] = self.df["original_name_col"].apply(get_curate_id)
            tempid2length = self.df.set_index("temp_col")[self.duration_col].to_dict()
            # drop some samples
            ignore_ids = [key for key, value in self.curation_chunks.items() if value[0] == "i"]
            print("Curation chunks: Ignoring ids", ignore_ids)
            self.df = self.df[~self.df["temp_col"].isin(ignore_ids)].reset_index(drop=True)
            self.df = self.df.drop(columns=["temp_col"])
            self.curation_chunks = {key: value for key, value in self.curation_chunks.items() if key in tempid2length}
            for key in self.curation_chunks.keys():
                self.curation_chunks[key] = np.array(
                    [
                        (
                            int(el[0] * self.sample_rate),
                            int(tempid2length[key] * self.sample_rate if el[1] is None else el[1] * self.sample_rate),
                        )
                        for el in self.curation_chunks[key][1:]
                    ]
                )

        if curation_chunks_json_path is not None:
            self.curation_chunks = load_json(curation_chunks_json_path)
            self.df["temp_col"] = self.df["original_name_col"].apply(get_curate_id)
            # drop some samples
            ignore_ids = [key for key, value in self.curation_chunks.items() if len(value) == 0]
            print("Curation chunks: Ignoring ids", ignore_ids)
            self.df = self.df[~self.df["temp_col"].isin(ignore_ids)].reset_index(drop=True)
            self.df = self.df.drop(columns=["temp_col"])
            for key in self.curation_chunks.keys():
                self.curation_chunks[key] = np.array(
                    [
                        (
                            (int(el[0] * self.sample_rate) if el[0] != -1 else -1),
                            (int(el[1] * self.sample_rate) if el[1] != -1 else -1),
                        )
                        for el in self.curation_chunks[key]
                    ]
                )

        if use_sampler:
            self.targets = (
                self.df[sampler_col].tolist() if sampler_col is not None else self.df[self.target_col].tolist()
            )
        if mixup_params.get("weights_path", None):
            self.weights = load_json(mixup_params["weights_path"])
            self.weights = torch.FloatTensor([self.weights[el] for el in self.targets])

        self.oof_sampling_config = oof_sampling_config
        if oof_df_path is not None:
            assert self.timewise_col is None, "OOF sampling is not supported with timewise_col"
            assert segment_len == 5.0, "OOF sampling is not supported with segment_len != 5.0"

            oof_df = pd.read_csv(oof_df_path)
            oof_df["sample_id"] = oof_df["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
            oof_df["end_second"] = oof_df["row_id"].apply(lambda x: int(x.split("_")[-1]))
            label_int2str = {v: k for k, v in self.label_str2int.items()}
            arranged_birds = [label_int2str[idx] for idx in range(len(label_int2str))]

            oof_probs = oof_df[arranged_birds].values
            oof_probs[oof_probs < self.oof_sampling_config["trim_min_prob"]] = 0
            oof_df[arranged_birds] = oof_probs

            self.sample2oofprob = (
                oof_df.groupby("sample_id")
                .apply(lambda sample_id_df: sample_id_df.sort_values("end_second")[arranged_birds].values)
                .to_dict()
            )

            self.df["sample_id"] = self.df[self.name_col].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        else:
            self.sample2oofprob = None

    def turn_off_all_augs(self):
        print("All augs Turned Off")
        self.do_mixup = False
        self.early_aug = None
        self.late_aug = None

    def __len__(self):
        return len(self.df) * self.dataset_repeat

    def _prepare_sample_piece(
        self, input, end_second=None, sample_slices=None, max_sampling_second=None, trim_indices=None
    ):
        if end_second is not None and trim_indices is not None:
            raise ValueError("end_second and trim_indices can not be used together")
        if sample_slices is not None:
            if input.shape[0] < self.segment_len:
                pad_len = self.segment_len - input.shape[0]
                assert len(sample_slices) == 1
                return (
                    np.pad(np.array(input) if self.use_h5py else input, ((pad_len, 0))),
                    0,
                )
            sample_slices_padded = [
                [
                    int((el[0] - self.timewise_slice_pad) * self.sample_rate),
                    int((el[1] + self.timewise_slice_pad) * self.sample_rate),
                ]
                for el in sample_slices
            ]
            sample_slices_padded[0][0] = max(0, sample_slices_padded[0][0])
            sample_slices_padded[-1][1] = min(input.shape[0] - self.segment_len, sample_slices_padded[-1][1])
            if sample_slices_padded[-1][1] - sample_slices_padded[-1][0] < 1:
                if len(sample_slices_padded) > 1:
                    sample_slices_padded.pop()
                else:
                    sample_slices_padded[0][1] = sample_slices_padded[0][0] + 1
            # print("Padded sample slices", sample_slices_padded)
            start, chunk_idx = sample_uniformly_np_index(sample_slices_padded)
            cutted_slice = (
                np.array(input[start : start + self.segment_len])
                if self.use_h5py
                else input[start : start + self.segment_len]
            )
            # print("Picked start", start)
            if cutted_slice.shape[0] < self.segment_len:
                pad_len = self.segment_len - cutted_slice.shape[0]
                cutted_slice = np.pad(np.array(cutted_slice), ((pad_len, 0)))
            return cutted_slice, chunk_idx
        if end_second is not None:
            end = int(end_second * self.sample_rate)
            input = (
                np.array(input[end - self.segment_len : end]) if self.use_h5py else input[end - self.segment_len : end]
            )
        input_len = input.shape[0]
        return_start = 0
        if trim_indices is not None:
            input_len = trim_indices[1] - trim_indices[0]
            return_start = trim_indices[0]
        if input_len < self.segment_len:
            input = input[trim_indices[0] : trim_indices[1]] if trim_indices is not None else np.array(input)
            pad_len = self.segment_len - len(input)
            return np.pad(input if self.use_h5py else input, ((pad_len, 0))), return_start
        elif input_len > self.segment_len:
            if self.start_from_zero:
                start = 0
            else:
                left_bound = 0
                if max_sampling_second is not None:
                    right_bound = min(input.shape[0], int(max_sampling_second * self.sample_rate)) - self.segment_len
                else:
                    if trim_indices is not None:
                        right_bound = trim_indices[1] - self.segment_len
                        left_bound = trim_indices[0]
                    else:
                        right_bound = input.shape[0] - self.segment_len
                start = np.random.randint(left_bound, right_bound)
                return_start = start
            input = (
                np.array(input[start : start + self.segment_len])
                if self.use_h5py
                else input[start : start + self.segment_len]
            )
            # TODO: Fast and dirty fix for curation length unaligned
            if trim_indices is not None:
                pad_len = self.segment_len - len(input)
                if pad_len > 0:
                    input = np.pad(input if self.use_h5py else input, ((pad_len, 0)))
            return input, return_start
        else:
            if trim_indices is not None:
                input = input[trim_indices[0] : trim_indices[1]]
                # TODO: Fast and dirty fix for curation length unaligned
                pad_len = self.segment_len - len(input)
                if pad_len > 0:
                    input = np.pad(input if self.use_h5py else input, ((pad_len, 0)))
            return np.array(input) if self.use_h5py else input, return_start

    def _prepare_target(self, main_tgt, sec_tgt, all_labels=None):
        if all_labels is not None:
            if all_labels == "nocall":
                return torch.zeros(len(self.label_str2int)).float()
            else:
                all_labels = all_labels.split() if isinstance(all_labels, str) else all_labels
                all_tgt = [self.label_str2int[el] for el in all_labels]
        else:
            if main_tgt == "nocall":
                all_tgt = []
            else:
                all_tgt = [self.label_str2int[main_tgt]] + [self.label_str2int[el] for el in sec_tgt if el != ""]
        all_tgt = torch.nn.functional.one_hot(torch.LongTensor(all_tgt), len(self.label_str2int)).float()
        all_tgt = torch.clamp(all_tgt.sum(0), 0.0, 1.0)
        if self.label_smoothing is not None:
            all_tgt = all_tgt * (1 - self.label_smoothing) + self.label_smoothing * all_tgt.sum() / all_tgt.shape[-1]
        return all_tgt

    def _prepare_sample_target_from_idx(self, idx: int):
        is_pseudo_sample = False
        if self.soundscape_df is not None and np.random.binomial(1, self.soundscape_pseudo_config["sampling_prob"]):
            initial_class = self.df[self.target_col].iloc[idx]
            if initial_class in self.pseudo_birds:
                idx = np.random.choice(np.where(self.soundscape_df[self.target_col] == initial_class)[0])
                is_pseudo_sample = True
        if self.empty_soundscape_config is not None and np.random.binomial(1, self.empty_soundscape_config["prob"]):
            wave = self.empty_sampler.sample(sample_length=self.segment_len)
            target = torch.zeros(len(self.label_str2int)).float()
        else:
            if is_pseudo_sample:
                row = self.soundscape_df.iloc[idx]
            else:
                row = self.df.iloc[idx]
            if self.use_h5py:
                with h5py.File(row[self.name_col], "r", swmr=True) as f:
                    # Very messy but let it be
                    if (
                        row["data_root_id"] is not None
                        and row["data_root_id"].startswith("soundscape")
                        and self.timewise_col is None
                    ):
                        end_second = row["final_second"]
                    else:
                        end_second = None
                    if self.timewise_col is not None:
                        sample_slices = row[self.timewise_col]
                        # print("Input sample slices", sample_slices)
                        wave, chunk_idx = self._prepare_sample_piece(
                            f["au"],
                            end_second=end_second,
                            sample_slices=sample_slices,
                            max_sampling_second=self.right_bound_main,
                        )
                    else:
                        sample_slices = None
                        trim_indices = None
                        if not is_pseudo_sample:
                            if self.curation_chunks is not None:
                                curate_id = get_curate_id(row["original_name_col"])
                                if curate_id in self.curation_chunks:
                                    all_trim_indices = self.curation_chunks[curate_id]
                                    if all_trim_indices[0][0] == -1:
                                        trim_indices = None
                                    else:
                                        lengths = all_trim_indices[:, 1] - all_trim_indices[:, 0]
                                        if lengths.sum() == 0:
                                            print("All trim indices are empty")
                                            trim_indices = None
                                        else:
                                            chosen_index = np.random.choice(
                                                range(all_trim_indices.shape[0]), p=lengths / lengths.sum()
                                            )
                                        trim_indices = (
                                            all_trim_indices[chosen_index][0],
                                            all_trim_indices[chosen_index][1],
                                        )
                            elif self.curation_json is not None:
                                trim_indices = self.curation_json[get_curate_id(row[self.name_col])]
                            if trim_indices is not None and trim_indices[0] == -1:
                                trim_indices = None
                            # print("Trim Indices", trim_indices[0] / self.sample_rate, trim_indices[1] / self.sample_rate)
                        wave, start_point = self._prepare_sample_piece(
                            f["au"],
                            end_second=end_second,
                            max_sampling_second=self.right_bound_main,
                            trim_indices=trim_indices,
                        )
            else:
                if self.precompute:
                    wave = self.audio_cache[idx]
                else:
                    # Extract only audio, without sample_rate
                    wave = load_pp_audio(
                        row[self.name_col],
                        sr=self.sample_rate,
                        do_noisereduce=self.do_noisereduce,
                        normalize=(not self.late_normalize) and self.load_normalize,
                        res_type=self.res_type,
                        pos_dtype=self.pos_dtype,
                    )

                if self.pos_dtype is not None:
                    wave = wave.astype(np.float32)

                wave, start_point = self._prepare_sample_piece(wave, max_sampling_second=self.right_bound_main)

            # print("Start Point in seconds:", start_point / self.sample_rate)

            main_tgt = row[self.target_col]
            if self.sec_target_col is not None:
                sec_tgt = row[self.sec_target_col]
            else:
                sec_tgt = [""]
            if self.timewise_col is not None:
                # print("Chunk", self.df[self.timewise_col].iloc[idx][chunk_idx])
                target = self._prepare_target(main_tgt, sec_tgt, all_labels=row[self.timewise_col][chunk_idx][-1])
            elif is_pseudo_sample:
                target = torch.from_numpy(row[self.indices_for_soundscapes].values.astype(np.float32)).float()
            else:
                target = None
                if (
                    self.sample2oofprob is not None
                    and not is_pseudo_sample
                    and row["sample_id"] in self.sample2oofprob
                    and np.random.binomial(1, self.oof_sampling_config["sampling_prob"])
                ):
                    start_second = start_point / self.sample_rate
                    try:
                        target = get_weighted_probability_vector(
                            start_second,
                            self.sample2oofprob[row["sample_id"]],
                        )
                        current_bird_names = [el for el in [main_tgt] + sec_tgt if el != ""]
                        selected_probs = [target[self.label_str2int[el]] for el in current_bird_names]
                        if max(selected_probs) < self.oof_sampling_config["acceptance_prob"]:
                            # If model predicts other species with high probability - than we should not rely on predictions
                            if (
                                self.oof_sampling_config.get("use_zero_samples", False)
                                and target.max() < self.oof_sampling_config["zero_sample_prob"]
                            ):
                                target = np.zeros(len(self.label_str2int))
                            else:
                                target = None
                    except Exception as e:
                        print("Failed on sample", row["sample_id"])
                        raise e

                if target is None:
                    target = self._prepare_target(main_tgt, sec_tgt)
                else:
                    target = torch.from_numpy(target.astype(np.float32)).float()
            if self.rating_col is not None:
                rating = row[self.rating_col] / MAX_RATING
                assert 0.0 <= rating <= 1.0
                target = (target * rating).float()

        if self.early_aug is not None:
            raise RuntimeError("Not implemented")

        if self.late_normalize:
            wave = librosa.util.normalize(wave)

        return wave, target

    def _get_mixup_idx(self):
        if self.mixup_params.get("weights_path", None):
            mixup_idx = torch.multinomial(self.weights, 1, replacement=True).item()
        else:
            mixup_idx = np.random.randint(0, self.__len__())
        return mixup_idx

    def __getitem__(self, index: int):
        if self.dataset_repeat > 1:
            index = index % len(self.df)

        wave, target = self._prepare_sample_target_from_idx(index)

        # Mixup/Cutmix/Fmix
        # .....
        if self.do_mixup and np.random.binomial(n=1, p=self.mixup_params["prob"]):
            n_samples = self.mixup_params.get("n_samples", 1)
            target_aggregation = self.mixup_params.get("target_aggregation", "sum")
            assert n_samples >= 1
            if n_samples == 1:
                mixup_idx = self._get_mixup_idx()
                (
                    mixup_wave,
                    mixup_target,
                ) = self._prepare_sample_target_from_idx(mixup_idx)
                multimix = False
            else:
                n_samples = np.random.randint(1, n_samples + 1)
                mixup_wave, mixup_target = [], []
                for _ in range(n_samples):
                    mixup_idx = self._get_mixup_idx()
                    (
                        _mixup_wave,
                        _mixup_target,
                    ) = self._prepare_sample_target_from_idx(mixup_idx)
                    mixup_wave.append(_mixup_wave)
                    mixup_target.append(_mixup_target)
                multimix = True

            if self.mixup_params["alpha"] is None:
                if multimix:
                    if target_aggregation != "sum":
                        raise ValueError("target_aggregation should be `sum` for multimix")
                    wave = (sum(mixup_wave) + wave) / (n_samples + 1)
                    target = sum(mixup_target) + target
                else:
                    wave = (mixup_wave + wave) / 2
                    if target_aggregation == "sum":
                        target = mixup_target + target
                    elif target_aggregation == "max":
                        target = torch.max(mixup_target, target)
                    elif target_aggregation == "full_probability":
                        target = 1 - (1 - mixup_target) * (1 - target)
                    else:
                        raise ValueError("target_aggregation should be `sum`, `max`, or `full_probability`")
            else:
                mix_weight = np.random.beta(self.mixup_params["alpha"], self.mixup_params["alpha"])
                if self.mixup_params.get("weight_trim", False):
                    mix_weight = min(
                        max(mix_weight, self.mixup_params["weight_trim"][0]),
                        self.mixup_params["weight_trim"][1],
                    )
                wave = mix_weight * mixup_wave + (1 - mix_weight) * wave
                if self.mixup_params.get("hard_target", True):
                    target = mixup_target + target
                else:
                    target = mix_weight * mixup_target + (1 - mix_weight) * target

            target = torch.clamp(target, min=0, max=1.0)
            if self.late_normalize:
                wave = librosa.util.normalize(wave)

        if self.do_montage:
            main_sample_birds = set([self.df[self.target_col].iloc[index]] + self.df[self.sec_target_col].iloc[index])
            samples_to_sample = np.random.randint(
                self.montage_params["montage_samples"][0], self.montage_params["montage_samples"][1] + 1
            )
            if samples_to_sample > 0:
                possible_birds = self.all_birds - main_sample_birds
                chosen_birds = np.random.choice(list(possible_birds), samples_to_sample, replace=False)
                idxs = [
                    np.random.choice(np.where(self.df[self.target_col] == bird)[0])
                    for bird in chosen_birds
                    if (self.df[self.target_col] == bird).sum() > 0
                ]

                if len(idxs) > 0:

                    montage_waves, montage_targets = [], []
                    for idx in idxs:
                        (
                            _montage_wave,
                            _montage_target,
                        ) = self._prepare_sample_target_from_idx(idx)
                        montage_waves.append(_montage_wave)
                        montage_targets.append(_montage_target)

                    mix_weight = np.random.beta(
                        self.montage_params["alpha"], self.montage_params["alpha"], len(montage_waves) + 1
                    )

                    mix_weight = mix_weight / mix_weight.sum()
                    wave = wave * mix_weight[0] + sum(
                        _wave * _weight for _wave, _weight in zip(montage_waves, mix_weight[1:])
                    )
                    if self.late_normalize:
                        wave = librosa.util.normalize(wave)

                    target = target + sum(montage_targets)
                    target = torch.clamp(target, min=0, max=1.0)

        if self.late_aug is not None:
            wave = self.late_aug(wave)
            if self.late_normalize:
                wave = librosa.util.normalize(wave)

        if self.unlabeled_files is not None:
            if self.unlabeled_params["mode"] == "return":
                random_file = np.random.choice(self.unlabeled_files)
                with h5py.File(random_file, "r", swmr=True) as f:
                    unlabeled_wave, start_point = self._prepare_sample_piece(
                        f["au"],
                        end_second=None,
                        max_sampling_second=self.right_bound_unlabeled,
                    )
                if self.late_normalize:
                    unlabeled_wave = librosa.util.normalize(unlabeled_wave)

                return wave, target, unlabeled_wave
            elif self.unlabeled_params["mode"] == "mixup":
                if np.random.binomial(1, self.unlabeled_params["prob"]):
                    random_file = np.random.choice(self.unlabeled_files)
                    with h5py.File(random_file, "r", swmr=True) as f:
                        unlabeled_wave, start_point = self._prepare_sample_piece(
                            f["au"],
                            end_second=None,
                            max_sampling_second=self.right_bound_unlabeled,
                        )
                    if self.unlabeled_params.get("alpha") is not None:
                        mix_weight = np.random.beta(self.unlabeled_params["alpha"], self.unlabeled_params["alpha"])
                    else:
                        mix_weight = 0.5

                    wave = mix_weight * unlabeled_wave + (1 - mix_weight) * wave
                    if self.late_normalize:
                        wave = librosa.util.normalize(wave)
                    mask = (target > 0).float()
                else:
                    mask = torch.ones_like(target)
                return wave, target, mask

        if self.use_label_smoothing_fix:
            target[target < self.label_smoothing_treshold] = (
                self.label_smoothing * (target >= self.label_smoothing_treshold).sum() / target.shape[-1]
            )

        return wave, target


class WaveAllFileDataset(WaveDataset):
    def __init__(
        self,
        df,
        root,
        label_str2int_mapping_path,
        df_path=None,
        add_df_paths=None,
        filename_change_mapping=None,
        target_col="primary_label",
        sec_target_col="secondary_labels",
        all_target_col="birds",
        name_col="filename",
        duration_col="duration_s",
        sample_id="sample_id",
        sample_rate=32_000,
        segment_len=5.0,
        step=None,
        do_not_exceed_duration=False,
        lookback=None,
        lookahead=None,
        precompute=False,
        early_aug=None,
        late_aug=None,
        do_mixup=False,
        mixup_params={"prob": 0.5, "alpha": 1.0},
        n_cores=None,
        debug=False,
        df_filter_rule=None,
        use_audio_cache=False,
        verbose=True,
        test_mode=False,
        soundscape_mode=False,
        use_eps_in_slicing=False,
        dfidx_2_sample_id=False,
        do_noisereduce=False,
        load_normalize=False,
        late_normalize=False,
        use_h5py=False,
        replace_pathes=None,
        # In BirdClef Comp, it is claimed that all samples in 32K sr
        # we will just validate it, without doing resampling
        validate_sr=None,
        ignore_setting_dataset_value=False,
        check_all_files_exist=True,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(f"WaveAllFileDataset received extra parameters: {kwargs}")
        if df_path is not None:
            df = pd.read_csv(df_path)
        if test_mode and soundscape_mode:
            raise RuntimeError("only test_mode or soundscape_mode can be activated")
        if precompute and use_audio_cache:
            raise RuntimeError("audio_cache is useless if you use precompute")
        if use_h5py and load_normalize:
            raise RuntimeError("load_normalize is not supported with h5py")
        super().__init__(
            df=df,
            add_df_paths=add_df_paths,
            filename_change_mapping=filename_change_mapping,
            root=root,
            label_str2int_mapping_path=label_str2int_mapping_path,
            target_col=target_col,
            sec_target_col=sec_target_col,
            name_col=name_col,
            sample_rate=sample_rate,
            segment_len=segment_len,
            # In case of soundscape_mode, cache will be computed in another way
            precompute=precompute and not soundscape_mode,
            early_aug=early_aug,
            late_aug=late_aug,
            do_mixup=do_mixup,
            mixup_params=mixup_params,
            n_cores=n_cores,
            debug=debug,
            df_filter_rule=df_filter_rule,
            do_noisereduce=do_noisereduce,
            late_normalize=late_normalize,
            use_h5py=use_h5py,
            replace_pathes=replace_pathes,
            ignore_setting_dataset_value=ignore_setting_dataset_value,
            check_all_files_exist=check_all_files_exist,
            duration_col=duration_col,
        )
        self.validate_sr = validate_sr
        self.load_normalize = load_normalize
        if precompute and soundscape_mode:
            self.audio_cache = {
                # Extract only audio, without sample_rate
                im_name: load_pp_audio(
                    im_name,
                    sr=None if self.validate_sr is not None else self.sample_rate,
                    do_noisereduce=do_noisereduce,
                    normalize=(not self.late_normalize) and self.load_normalize,
                    validate_sr=self.validate_sr,
                )
                for im_name in tqdm(
                    set(self.df[self.name_col]),
                    total=len(set(self.df[self.name_col])),
                )
            }
            self.precompute = True

        self.verbose = verbose
        self.test_mode = test_mode
        self.soundscape_mode = soundscape_mode
        self.all_target_col = all_target_col
        self.sample_id = sample_id
        self.dfidx_2_sample_id = dfidx_2_sample_id
        eps = EPS if use_eps_in_slicing else 0

        if sample_id is not None:
            self.df[self.sample_id] = self.df[self.sample_id].astype("category").cat.codes

        self.sampleidx_2_dfidx = {}
        if lookahead is not None or lookback is not None or step is not None:
            print("Dataset in hard_slicing mode")
            if step is None:
                step = segment_len
            if lookahead is None:
                lookahead = 0
            if lookback is None:
                lookback = 0
            self.hard_slicing = True
            itter = 0
            if soundscape_mode:
                samples_generator = enumerate(self.df.drop_duplicates(self.name_col)[self.duration_col])
            else:
                samples_generator = enumerate(self.df[self.duration_col])
            for dfidx, dur in samples_generator:
                real_start = -lookback
                while real_start + lookback < dur + eps:
                    if do_not_exceed_duration and round(real_start + lookback + segment_len, 4) > round(dur, 4):
                        break
                    self.sampleidx_2_dfidx[itter] = {
                        "dfidx": itter if soundscape_mode else dfidx,
                        "start": int(real_start * self.sample_rate),
                        "end": int((real_start + lookback + segment_len + lookahead) * self.sample_rate),
                        "end_s": min(real_start + lookback + segment_len, dur),
                    }
                    real_start += step
                    itter += 1
        else:
            self.hard_slicing = False
            t_start = 0
            if soundscape_mode:
                samples_generator = enumerate(self.df.drop_duplicates(self.name_col)[self.duration_col])
            else:
                samples_generator = enumerate(self.df[self.duration_col])
            for dfidx, dur in samples_generator:
                n_pieces_in_file = math.ceil(dur / segment_len)
                self.sampleidx_2_dfidx.update(
                    {
                        i
                        + t_start: {
                            "dfidx": i + t_start if soundscape_mode else dfidx,
                            "start": int(segment_len * i * self.sample_rate),
                            "end_s": int(segment_len * (i + 1)),
                        }
                        for i in range(n_pieces_in_file)
                    }
                )
                t_start += n_pieces_in_file

        self.use_audio_cache = use_audio_cache
        self.test_audio_cache = {"au": None, "dfidx": None}

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    def _hadle_au_cache(self, dfidx):
        if self.test_audio_cache["dfidx"] is None or self.test_audio_cache["dfidx"] != dfidx:
            del self.test_audio_cache["au"]
            gc.collect()
            start_time = time()
            self._print_v(f"Loading {self.df[self.name_col].iloc[dfidx]} to audio cache")
            self.test_audio_cache["au"] = load_pp_audio(
                self.df[self.name_col].iloc[dfidx],
                sr=None if self.validate_sr is not None else self.sample_rate,
                do_noisereduce=self.do_noisereduce,
                normalize=(not self.late_normalize) and self.load_normalize,
                validate_sr=self.validate_sr,
            )
            self.test_audio_cache["dfidx"] = dfidx
            self._print_v(f"Loading took {time() - start_time} seconds")
        return self.test_audio_cache["au"]

    def __len__(self):
        return len(self.sampleidx_2_dfidx)

    def _prepare_sample_piece_hard(self, input, start, end):
        # Process right pad or end trim
        if end > input.shape[0]:
            input = np.pad(
                np.array(input) if self.use_h5py else input,
                ((0, end - input.shape[0])),
            )
        else:
            input = np.array(input[:end]) if self.use_h5py else input[:end]
        # Process left pad or start trim
        if start < 0:
            input = np.pad(input, ((-start, 0)))
        else:
            input = input[start:]

        return input

    def _prepare_sample_piece(self, input, start):
        input = (
            np.array(input[start : start + self.segment_len])
            if self.use_h5py
            else input[start : start + self.segment_len]
        )

        if input.shape[0] < self.segment_len:
            pad_len = self.segment_len - input.shape[0]
            input = np.pad(input, ((pad_len, 0)))
        else:
            pad_len = 0

        return input, pad_len

    def _prepare_sample_target_from_idx(self, idx: int):
        map_dict = self.sampleidx_2_dfidx[idx]
        dfidx = map_dict["dfidx"]
        start = map_dict["start"]
        end = start + self.segment_len

        if self.use_h5py:
            with h5py.File(self.df[self.name_col].iloc[dfidx], "r", swmr=True) as f:
                if self.hard_slicing:
                    wave = self._prepare_sample_piece_hard(f["au"], start=start, end=map_dict["end"])
                else:
                    wave, _ = self._prepare_sample_piece(f["au"], start=start)
        else:
            if self.precompute:
                if self.soundscape_mode:
                    wave = self.audio_cache[self.df[self.name_col].iloc[dfidx]]
                else:
                    wave = self.audio_cache[dfidx]
            else:
                # Extract only audio, without sample_rate
                if self.use_audio_cache:
                    wave = self._hadle_au_cache(dfidx)
                else:
                    wave = load_pp_audio(
                        self.df[self.name_col].iloc[dfidx],
                        sr=None if self.validate_sr is not None else self.sample_rate,
                        do_noisereduce=self.do_noisereduce,
                        normalize=not self.late_normalize,
                        validate_sr=self.validate_sr,
                    )
            if self.hard_slicing:
                wave = self._prepare_sample_piece_hard(wave, start=start, end=map_dict["end"])
            else:
                wave, _ = self._prepare_sample_piece(wave, start=start)

        if self.test_mode:
            target = -1
        else:
            if self.soundscape_mode:
                target = self._prepare_target(
                    main_tgt=None,
                    sec_tgt=None,
                    all_labels=self.df[self.all_target_col].iloc[dfidx],
                )
            else:
                main_tgt = self.df[self.target_col].iloc[dfidx]
                sec_tgt = self.df[self.sec_target_col].iloc[dfidx]
                target = self._prepare_target(main_tgt, sec_tgt)

        if self.dfidx_2_sample_id:
            dfidx = self.df[self.sample_id].iloc[dfidx]

        end = map_dict["end_s"]

        if self.early_aug is not None:
            raise RuntimeError("Not implemented")

        if self.late_normalize:
            wave = librosa.util.normalize(wave)

        return wave, target, dfidx, start, end

    def __getitem__(self, index: int):
        wave, target, dfidx, start, end = self._prepare_sample_target_from_idx(index)

        # Mixup/Cutmix/Fmix
        # .....
        if self.do_mixup and np.random.binomial(n=1, p=self.mixup_params["prob"]):
            raise RuntimeError("Not implemented")

        if self.late_aug is not None:
            raise RuntimeError("Not implemented")

        return wave, target, dfidx, start, end
