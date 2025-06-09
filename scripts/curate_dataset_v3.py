import argparse
from glob import glob

import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from code_base.datasets.wave_dataset import get_curate_id
from code_base.utils.main_utils import get_device, write_json


def get_non_speech_regions(
    audio,
    sr,
    vad_model,
    get_speech_timestamps,
    device,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=1000,
):
    """
    Returns a list of (start, end) tuples (in seconds) for non-speech regions in the audio.
    Speech segments shorter than min_speech_len are ignored (not considered as speech).
    """
    if len(audio.shape) > 1:
        audio = audio[0]
    if not hasattr(audio, "to"):
        audio = torch.tensor(audio)
    audio = audio.to(device)
    total_len = audio.shape[-1]
    audio_ms = total_len / sr * 1000
    if audio_ms <= min_silence_duration_ms:
        min_silence_duration_ms = 100
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    total_len = audio.shape[-1]
    non_speech = []
    prev_end = 0
    for ts in speech_timestamps:
        start = ts["start"]
        end = ts["end"]
        # Only add non-speech region if its duration is >= min_silence_duration_ms
        # chunk_ms = (start - prev_end) / sr * 1000
        # if start > prev_end and chunk_ms >= min_silence_duration_ms:
        if start > prev_end:
            non_speech.append((prev_end / sr, start / sr))
        prev_end = end
    # Check last region
    # chunk_ms = (total_len - prev_end) / sr * 1000
    # if prev_end < total_len and chunk_ms >= min_silence_duration_ms:
    if prev_end < total_len:
        non_speech.append((prev_end / sr, total_len / sr))
    return non_speech


def main():
    parser = argparse.ArgumentParser(description="Detect non-speech regions in audio files using Silero VAD.")
    parser.add_argument(
        "--file_globs", type=str, nargs="+", required=True, help="List of glob patterns to collect audio files."
    )
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the resulting JSON.")
    parser.add_argument("--required_sr", type=int, default=32000, help="Required sample rate for VAD.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Silero VAD threshold.")
    parser.add_argument("--min_speech_duration_ms", type=int, default=250, help="Minimum speech segment length in ms.")
    parser.add_argument(
        "--min_silence_duration_ms", type=int, default=1000, help="Minimum silence segment length in ms."
    )
    args = parser.parse_args()

    device = get_device()
    vad_model, (get_speech_timestamps, _, _, _, _) = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad"
    )
    if device == "cuda":
        vad_model = vad_model.to(device)
    vad_model.eval()

    all_filenames = []
    for glob_pattern in args.file_globs:
        all_filenames.extend(glob(glob_pattern, recursive=True))

    print(f"Found {len(all_filenames)} audio files matching the patterns")

    all_filenames_bounds = {}
    for filename in tqdm(all_filenames):
        try:
            audio, sr = torchaudio.load(filename)
            if sr != args.required_sr:
                resampler = Resample(orig_freq=sr, new_freq=args.required_sr)
                audio = resampler(audio)
                sr = args.required_sr
            non_speech_regions = get_non_speech_regions(
                audio,
                sr,
                vad_model,
                get_speech_timestamps,
                device,
                threshold=args.threshold,
                min_speech_duration_ms=args.min_speech_duration_ms,
                min_silence_duration_ms=args.min_silence_duration_ms,
            )
            all_filenames_bounds[get_curate_id(filename)] = non_speech_regions
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            all_filenames_bounds[get_curate_id(filename)] = [(-1, -1)]

    # Manual correction
    all_filenames_bounds["compau/iNat1217422.ogg"] = [(0, 1.0)]
    all_filenames_bounds["socfly1/iNat492939.ogg"] = [(0, 1.625)]
    all_filenames_bounds["amekes/iNat203272.ogg"] = [(0, 2.113031)]

    write_json(args.output_json, all_filenames_bounds)
    print(f"Saved non-speech regions to {args.output_json}")


if __name__ == "__main__":
    main()
