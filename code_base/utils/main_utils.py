import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from joblib import Parallel
from tqdm import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def stack_and_max_by_samples(x):
    return np.stack(x).max(axis=0)


def groupby_np_array(groupby_f, array_to_group, apply_f):
    series = (
        pd.DataFrame(
            {
                "groupby_f": groupby_f,
                "array_to_group": [el for el in array_to_group],
            }
        )
        .groupby("groupby_f")["array_to_group"]
        .apply(apply_f)
    )
    return np.stack(series.values)


def load_json(path: str) -> Mapping[str, Any]:
    """
    Read .json file and return dict
    """
    with open(path, "r") as read_file:
        loaded_dict = json.load(read_file)
    return loaded_dict


def write_json(path, data):
    """
    Saves dict into .json file
    """
    with open(path, "w", encoding="utf-8") as f:
        result = json.dump(data, f, ensure_ascii=False, indent=4)
    return result


def load_yaml(path: str) -> Mapping[str, Any]:
    """
    Read .yaml file and return dict
    """
    with open(path, "r") as read_file:
        loaded_dict = yaml.load(read_file, Loader=yaml.FullLoader)
    return loaded_dict


def sample_uniformly_np_index(merged_intervals):
    # Calculate the total span of all intervals in terms of seconds
    total_points = sum(interval[1] - interval[0] for interval in merged_intervals)

    # Generate a random position within the total span as a float
    random_position = np.random.randint(0, total_points)

    # Map the random position to an actual interval
    current_position = 0
    for index, (start, end) in enumerate(merged_intervals):
        interval_length = end - start
        if current_position + interval_length > random_position:
            # Calculate the actual second within the interval as a float
            sampled_point = start + (random_position - current_position)
            return sampled_point, index
        current_position += interval_length

    raise RuntimeError("sample_uniformly_np_index was not able to pick chunk")


def get_device():
    """
    Get device to use
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def create_oversampled_df_for_class(original_df, class_name, n_add_samples):
    class_sub_df = original_df[original_df["primary_label"] == class_name].reset_index(drop=True)
    repeat_times = math.ceil(n_add_samples / class_sub_df.shape[0])
    oversampled_df = pd.concat([class_sub_df] * repeat_times).sample(frac=1).reset_index(drop=True)
    return oversampled_df.iloc[:n_add_samples]


def standardize_audio_filters(raw_audio_filters):
    """finalize a dict of audio dilters by converting audio hits into audio sections (start, end),
    merging them when applicable (if the distance to the previous hit <= 5s).
    section for a hit is defined a (hit - BAND, hit + BAND)
    The list sections is prefixed with the type of curation:
        - 'm': the bird is vocalizing in every 5s segment of every section
        - 'a': the bird is not guaranted to vocalize in every 5s segment of every section
        - 'i': the bird is not vocalizing; ignore the audio
    """
    BAND = 4

    audio_filters = dict()
    for id_, raw_sections in raw_audio_filters.items():
        prior_hit = -100
        if raw_sections[0] in ["m", "i"]:
            curation = raw_sections[0]
            start = 1
        else:
            curation = "a"
            start = 0
        sections = [curation]
        for sec in raw_sections[start:]:
            if isinstance(sec, tuple):
                sections.append(sec)
                if sec[1] is not None:
                    prior_hit = sec[1] - 4
            else:
                if prior_hit + 5 >= sec:  # merge with previous section
                    sections[-1] = (sections[-1][0], sec + BAND)
                else:
                    start = max(sec - BAND, 0)
                    end = 5 if start == 0 else sec + BAND
                    sections.append((start, end))
                prior_hit = sec

        audio_filters[id_] = sections
    return audio_filters


def download_audio_file(url: str, save_path: str, verbose: bool = True):
    """
    Downloads an audio file from the provided URL and saves it to the specified path.

    Args:
        url (str): Direct download link to the audio file.
        save_path (str): Full path including filename where the audio should be saved.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for bad status codes
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if verbose:
            print(f"Audio file downloaded successfully to {save_path}")
    except Exception as e:
        if verbose:
            print(f"Error downloading audio file: {e}")
