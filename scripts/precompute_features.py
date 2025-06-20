import argparse
import os
from copy import deepcopy
from glob import glob
from os.path import join as pjoin
from os.path import splitext

import h5py
import librosa
import torchaudio
from joblib import delayed
from torchaudio.transforms import Resample

from code_base.utils.main_utils import ProgressParallel, load_json


def create_target_path(target_root, source_path, add_class_folder_to_path=True):
    splitted_source_path = source_path.split("/")
    filename = splitext(splitted_source_path[-1])[0]
    if add_class_folder_to_path:
        target_path = pjoin(
            target_root, splitted_source_path[-2], filename + ".hdf5"
        )
    else:
        target_path = pjoin(target_root, filename + ".hdf5")
    return target_path


def get_load_librosa_save_h5py(do_normalize, **kwargs):
    def load_librosa_save_h5py(load_path, save_path):
        if not os.path.exists(save_path):
            try:
                if kwargs.get("use_torchaudio", False):
                    au, sr = torchaudio.load(load_path)
                    if kwargs.get("sr") is not None and sr != kwargs["sr"]:
                        resampler = Resample(
                            orig_freq=sr, new_freq=kwargs["sr"]
                        )
                        au = resampler(au)
                        sr = kwargs["sr"]
                    au = au[0].numpy()
                else:
                    if "use_torchaudio" in kwargs:
                        kwargs.pop("use_torchaudio")
                    au, sr = librosa.load(load_path, **kwargs)
                if do_normalize:
                    au = librosa.util.normalize(au)
                with h5py.File(save_path, "w") as data_file:
                    data_file.create_dataset("au", data=au)
                    data_file.create_dataset("sr", data=sr)
                    data_file.create_dataset(
                        "do_normalize", data=int(do_normalize)
                    )
            except Exception as e:
                print(f"Failed to load {load_path} with {e}")

    return load_librosa_save_h5py


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("au_path", type=str, help="Path to folder with audios")
    parser.add_argument(
        "save_path",
        type=str,
        help="Path to folder to save .hdf5 files",
    )
    parser.add_argument(
        "--do_normalize",
        default=False,
        action="store_true",
        help="Normalize audio",
    )
    parser.add_argument(
        "--use_torchaudio",
        default=False,
        action="store_true",
        help="Read audio with torchaudio instead of librosa",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=32_000,
        help="Sample Rate for resampling",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=32,
        help="Number of cores for parallel processing",
    )

    args = parser.parse_args()
    print(f"Received args: {args}")

    add_class_folder_to_path = True
    if args.au_path.endswith(".json"):
        all_aus = load_json(args.au_path)
    else:
        all_aus = glob(pjoin(args.au_path, "*", "*"))
    if len(all_aus) == 0:
        all_aus = glob(pjoin(args.au_path, "*"))
        add_class_folder_to_path = False
    print(f"Found {len(all_aus)} files")

    all_targets = [create_target_path(args.save_path, el, add_class_folder_to_path=add_class_folder_to_path) for el in all_aus]

    os.makedirs(args.save_path, exist_ok=True)
    for el in set([os.path.dirname(el) for el in all_targets]):
        os.makedirs(el, exist_ok=True)

    ProgressParallel(n_jobs=args.n_cores, total=len(all_aus))(
        delayed(
            get_load_librosa_save_h5py(
                do_normalize=args.do_normalize, 
                sr=args.sr, 
                use_torchaudio=args.use_torchaudio
            )
        )(load_path, save_path)
        for load_path, save_path in zip(all_aus, all_targets)
    )

    if add_class_folder_to_path:
        saved_targets = glob(pjoin(args.save_path, "*", "*.hdf5"))
    else:
        saved_targets = glob(pjoin(args.save_path, "*.hdf5"))
    print(f"Saved {len(saved_targets)} files")

    print("Done!")
