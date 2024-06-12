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
from ..utils.main_utils import sample_uniformly_np_index

DEFAULT_TARGET = 0
EPS = 1e-5
MAX_RATING = 5.0


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
        res_type="kaiser_best",
        pos_dtype=None,
        sampler_col=None,
        use_h5py=False,
        empty_soundscape_config=None,
        soundscape_pseudo_df_path=None,
        start_from_zero=False,
        ignore_setting_dataset_value=False,
        dataset_repeat=1,
        unlabeled_glob=None,
        unlabeled_params={"mode": "return", "prob": 0.5, "alpha": None},
        right_bound_main=None,
        right_bound_unlabeled=None,
    ):
        super().__init__()
        assert unlabeled_params["mode"] in ["return", "mixup"]
        assert timewise_slice_pad < segment_len
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
            df["dataset"] = "base"
        if add_df_paths is not None:
            cols_to_take = [
                target_col,
                sec_target_col,
                name_col,
                "duration_s",
                "dataset",
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
            df = pd.concat([df, add_merged_df], axis=0).reset_index(drop=True)
        if soundscape_pseudo_df_path is not None:
            soundscape_pseudo_df = pd.read_csv(soundscape_pseudo_df_path)
            if "dataset" not in soundscape_pseudo_df.columns:
                soundscape_pseudo_df["dataset"] = "soundscape"
            soundscape_pseudo_df["final_second"] = soundscape_pseudo_df["row_id"].apply(lambda x: int(x.split("_")[-1]))
            df = pd.concat([df, soundscape_pseudo_df], axis=0).reset_index(drop=True)
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
                    filename_change_mapping["base"], filename_change_mapping[row["dataset"]]
                ),
                axis=1,
            )
        if replace_pathes is not None:
            self.df[f"{name_col}_with_root"] = self.df[f"{name_col}_with_root"].apply(
                lambda x: pjoin(replace_pathes[1], relpath(x, replace_pathes[0]))
            )

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
        if self.use_h5py:
            self.df[self.name_col] = self.df[self.name_col].apply(lambda x: splitext(x)[0] + ".hdf5")

        # Validate that all files exist
        filenames_exists = self.df[self.name_col].apply(lambda x: os.path.exists(x))
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

        if use_sampler:
            self.targets = (
                self.df[sampler_col].tolist() if sampler_col is not None else self.df[self.target_col].tolist()
            )
        if mixup_params.get("weights_path", None):
            self.weights = load_json(mixup_params["weights_path"])
            self.weights = torch.FloatTensor([self.weights[el] for el in self.targets])

        if self.empty_soundscape_config is not None:
            self.empty_sampler = BackgroundNoise(**self.empty_soundscape_config["sampler_config"])

    def turn_off_all_augs(self):
        print("All augs Turned Off")
        self.do_mixup = False
        self.early_aug = None
        self.late_aug = None

    def __len__(self):
        return len(self.df) * self.dataset_repeat

    def _prepare_sample_piece(self, input, end_second=None, sample_slices=None, max_sampling_second=None):
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
        if input.shape[0] < self.segment_len:
            pad_len = self.segment_len - input.shape[0]
            return np.pad(np.array(input) if self.use_h5py else input, ((pad_len, 0)))
        elif input.shape[0] > self.segment_len:
            if self.start_from_zero:
                start = 0
            else:
                if max_sampling_second is not None:
                    right_bound = min(input.shape[0], int(max_sampling_second * self.sample_rate)) - self.segment_len
                else:
                    right_bound = input.shape[0] - self.segment_len
                start = np.random.randint(0, right_bound)
            return (
                np.array(input[start : start + self.segment_len])
                if self.use_h5py
                else input[start : start + self.segment_len]
            )
        else:
            return np.array(input) if self.use_h5py else input

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
        return all_tgt

    def _prepare_sample_target_from_idx(self, idx: int):
        if self.empty_soundscape_config is not None and np.random.binomial(1, self.empty_soundscape_config["prob"]):
            wave = self.empty_sampler.sample(sample_length=self.segment_len)
            target = torch.zeros(len(self.label_str2int)).float()
        else:
            if self.use_h5py:
                with h5py.File(self.df[self.name_col].iloc[idx], "r") as f:
                    # Very messy but let it be
                    if (
                        self.df["dataset"].iloc[idx] is not None
                        and self.df["dataset"].iloc[idx].startswith("soundscape")
                        and self.timewise_col is None
                    ):
                        end_second = self.df["final_second"].iloc[idx]
                    else:
                        end_second = None
                    if self.timewise_col is not None:
                        sample_slices = self.df[self.timewise_col].iloc[idx]
                        # print("Input sample slices", sample_slices)
                        wave, chunk_idx = self._prepare_sample_piece(
                            f["au"],
                            end_second=end_second,
                            sample_slices=sample_slices,
                            max_sampling_second=self.right_bound_main,
                        )
                    else:
                        sample_slices = None
                        wave = self._prepare_sample_piece(
                            f["au"],
                            end_second=end_second,
                            max_sampling_second=self.right_bound_main,
                        )
            else:
                if self.precompute:
                    wave = self.audio_cache[idx]
                else:
                    # Extract only audio, without sample_rate
                    wave = load_pp_audio(
                        self.df[self.name_col].iloc[idx],
                        sr=self.sample_rate,
                        do_noisereduce=self.do_noisereduce,
                        normalize=(not self.late_normalize) and self.load_normalize,
                        res_type=self.res_type,
                        pos_dtype=self.pos_dtype,
                    )

                if self.pos_dtype is not None:
                    wave = wave.astype(np.float32)

                wave = self._prepare_sample_piece(wave, max_sampling_second=self.right_bound_main)

            main_tgt = self.df[self.target_col].iloc[idx]
            if self.sec_target_col is not None:
                sec_tgt = self.df[self.sec_target_col].iloc[idx]
            else:
                sec_tgt = [""]
            if self.timewise_col is not None:
                # print("Chunk", self.df[self.timewise_col].iloc[idx][chunk_idx])
                target = self._prepare_target(
                    main_tgt, sec_tgt, all_labels=self.df[self.timewise_col].iloc[idx][chunk_idx][-1]
                )
            else:
                target = self._prepare_target(main_tgt, sec_tgt)
            if self.rating_col is not None:
                rating = self.df[self.rating_col].iloc[idx] / MAX_RATING
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
                    wave = (sum(mixup_wave) + wave) / (n_samples + 1)
                    target = sum(mixup_target) + target
                else:
                    wave = (mixup_wave + wave) / 2
                    target = mixup_target + target
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
                with h5py.File(random_file, "r") as f:
                    unlabeled_wave = self._prepare_sample_piece(
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
                    with h5py.File(random_file, "r") as f:
                        unlabeled_wave = self._prepare_sample_piece(
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
        # In BirdClef Comp, it is claimed that all samples in 32K sr
        # we will just validate it, without doing resampling
        validate_sr=None,
        ignore_setting_dataset_value=False,
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
            ignore_setting_dataset_value=ignore_setting_dataset_value,
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

        self.duration_col = duration_col
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
        if lookahead is not None and lookback is not None:
            print("Dataset in hard_slicing mode")
            if step is None:
                step = segment_len
            self.hard_slicing = True
            itter = 0
            if soundscape_mode:
                samples_generator = enumerate(self.df.drop_duplicates(self.name_col)[self.duration_col])
            else:
                samples_generator = enumerate(self.df[self.duration_col])
            for dfidx, dur in samples_generator:
                real_start = -lookback
                while real_start + lookback < dur + eps:
                    self.sampleidx_2_dfidx[itter] = {
                        "dfidx": itter if soundscape_mode else dfidx,
                        "start": int(real_start * self.sample_rate),
                        "end": int((real_start + lookback + segment_len + lookahead) * self.sample_rate),
                        "end_s": min(int(real_start + lookback + segment_len), dur),
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
            with h5py.File(self.df[self.name_col].iloc[dfidx], "r") as f:
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
