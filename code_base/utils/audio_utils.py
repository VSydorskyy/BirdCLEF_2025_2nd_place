from copy import deepcopy

import librosa
import torchaudio

try:
    import noisereduce as nr
except:
    print("`noisereduce` was not imported")
from joblib import delayed

from .main_utils import ProgressParallel


def get_librosa_load(
    do_normalize,
    do_noisereduce=False,
    pos_dtype=None,
    return_au_len=False,
    **kwargs,
):
    def librosa_load(path):
        # assert kwargs["sr"] == 32_000
        try:
            au, sr = librosa.load(path, **kwargs)
            if do_noisereduce:
                try:
                    au = nr.reduce_noise(y=deepcopy(au), sr=sr)
                    if do_normalize:
                        au = librosa.util.normalize(au)
                    return au, sr
                except Exception as e:
                    print(f"{e} was catched while `reduce_noise`")
                    au, sr = librosa.load(path, **kwargs)
            if do_normalize:
                au = librosa.util.normalize(au)
            if pos_dtype is not None:
                au = au.astype(pos_dtype)
            if return_au_len:
                au = len(au)
            return au, sr
        except Exception as e:
            print("librosa_load failed with {e}")
            return None, None

    return librosa_load


def load_pp_audio(
    name,
    sr=None,
    normalize=True,
    do_noisereduce=False,
    pos_dtype=None,
    res_type="soxr_hq",
    validate_sr=None,
):
    # assert sr == 32_000
    au, sr = librosa.load(name, sr=sr)
    if validate_sr is not None:
        assert sr == validate_sr
    if do_noisereduce:
        try:
            au = nr.reduce_noise(y=deepcopy(au), sr=sr, res_type=res_type)
            if normalize:
                au = librosa.util.normalize(au)
            return au
        except Exception as e:
            print(f"{e} was catched while `reduce_noise`")
            au, sr = librosa.load(name, sr=sr)
    if normalize:
        au = librosa.util.normalize(au)
    if pos_dtype is not None:
        au = au.astype(pos_dtype)
    return au


def parallel_librosa_load(
    audio_pathes,
    n_cores=32,
    return_sr=True,
    return_audio=True,
    do_normalize=False,
    **kwargs,
):
    assert return_sr or return_audio
    complete_out = ProgressParallel(n_jobs=n_cores, total=len(audio_pathes))(
        delayed(get_librosa_load(do_normalize=do_normalize, **kwargs))(el_path) for el_path in audio_pathes
    )
    if return_sr and return_audio:
        return complete_out
    elif return_audio:
        return [el[0] for el in complete_out]
    elif return_sr:
        return [el[1] for el in complete_out]


def get_audio_metadata(file_path: str):
    """
    Extract metadata from an audio file using torchaudio.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: Metadata containing sample rate, duration, channels, bit depth, and encoding format.
    """
    try:
        # Get basic metadata
        metadata = torchaudio.info(file_path)

        # Extract key information
        sample_rate = metadata.sample_rate
        num_channels = metadata.num_channels
        num_frames = metadata.num_frames
        duration = num_frames / sample_rate if sample_rate else None

        # Additional metadata (if available)
        bit_depth = getattr(metadata, "bits_per_sample", None)  # Only available for certain formats
        encoding = getattr(metadata, "encoding", None)  # Available for some formats

        return {
            "sample_rate": sample_rate,
            "duration_s": duration,
            "num_channels": num_channels,
            "bit_depth": bit_depth,
            "encoding": encoding,
        }
    except:
        return {
            "sample_rate": None,
            "duration_s": None,
            "num_channels": None,
            "bit_depth": None,
            "encoding": None,
        }


# def get_weighted_probability_vector(start_second, pseudo_probs, chunk_duration=5):
#     """
#     Compute weighted probability vector over a 5-second window starting at start_second.

#     Parameters:
#     - start_second (float): The starting second within the audio.
#     - pseudo_probs (np.array): np.array where each row is a probability vector for a 5-second chunk.
#     - chunk_duration (float): Duration each row covers (default 5 seconds).

#     Returns:
#     - weighted_vector (np.ndarray): Weighted probability vector.
#     """
#     total_duration = len(pseudo_probs) * chunk_duration
#     if start_second < 0 or start_second >= total_duration:
#         raise ValueError(f"start_second must be within 0 and {total_duration} seconds. Started with {start_second}.")

#     # Figure out which rows (chunks) overlap with the 5-second window
#     window_start = start_second
#     window_end = start_second + chunk_duration

#     weighted_sum = np.zeros(pseudo_probs.shape[1])
#     remaining_window = chunk_duration

#     while round(remaining_window, 6) > 0 and window_start < total_duration:
#         current_chunk = int(window_start // chunk_duration)
#         chunk_start = current_chunk * chunk_duration
#         chunk_end = chunk_start + chunk_duration

#         # Compute overlap between window and this chunk
#         overlap_start = max(window_start, chunk_start)
#         overlap_end = min(window_end, chunk_end)
#         overlap_duration = overlap_end - overlap_start

#         assert overlap_duration >= 0, f"Overlap duration should be non-negative. Got {overlap_duration}."

#         weight = overlap_duration / chunk_duration
#         weighted_sum += pseudo_probs[current_chunk] * weight

#         window_start = chunk_end  # move to next chunk
#         remaining_window -= overlap_duration

#     return weighted_sum
