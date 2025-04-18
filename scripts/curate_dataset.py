import argparse
import os
from glob import glob

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from code_base.utils.main_utils import write_json

TRAIN_AUDIO_FILTERS = {  # list of tuple (start, end) and/or float (time of vocalization)
    "colcha1/XC337020": [(50, 200)],
    "colcha1/XC532406": [(0, 8)],
    "chbant1/XC315058": [(0, 19)],
    "52884/CSA15755": [(9.0, 22.57), (27, 49)],
    "gybmar/XC9608": [(0, 5), (25, 30), (35, 39)],
    "1194042/CSA18783": [(0.9, 18.5), (20, 26)],
    "1194042/CSA18802": [(0.9, 14.2), (25, 30)],
    "1346504/CSA18792": [(0, 21)],
    "134933/iNat1108984": [(1, 6), (11, 16), (21, 27)],
    "134933/iNat1160199": [(0, 20), (29, 47)],
    "21038/iNat297879.ogg": [(0, 12)],
    "21038/iNat65519": [(13, 120), (160, 300)],
    "21116/iNat65520": [(0, 6)],
    "24272/XC882885": [(5, 33), (40, 47), (49, 56)],
    "41778/XC959831": [(20, 35), (50, 75), (80, 123), (145, 170)],
    "42087/iNat155127": [(5, 12)],
    "47067/iNat68676": [(6, 43)],
    "548639/CSA34187": [(0, 8), (5, 10)],
    "555142/iNat31004": [(0, 8)],
    "64862/CSA18218": [(4.5, 22), (98, 135), (154, 161), (211, 235), (270, 290)],
    "64862/CSA18222": [(4.1, 30), (70, 95)],
    "65547/iNat1103224": [(0, 12), (11, 16.8)],
    "714022/CSA34203": [(0, 5.5), (2.5, 11), (8, 17), (17, 25)],
    "714022/CSA34204": [(0, 6), (4, 12), (12, 20), (19, 27), (25, 34), (30, 37)],
    "714022/CSA34205": [(0, 5.2), (5, 14), (15.5, 24), (25, 34), (33, 40)],
    "714022/CSA34206": [(0, 7), (6, 15), (15, 23), (22, 28)],
    "714022/CSA34207": [(0, 5.5), (7, 16), (18, 26), (29, 36), (35.8, 42)],
    "135045/iNat1122209": [(0, 10), (12, 22), (24, 32), (36, 46)],
    "135045/iNat1207345": [(9, 19), (34, 44), (64, 73), (69, 78), (78, 108), (105, 120), (120, 128)],
    "135045/iNat1207347": [
        (4, 14),
        (17, 42),
        (42, 50),
        (49.5, 58),
        (57, 66),
        (67, 76),
        (77, 87),
        (90, 100),
        (104, 111.4),
    ],
    "135045/iNat1208549": [
        (9, 19),
        (27, 36),
        (51, 60.5),
        (64, 73),
        (78, 86.5),
        (93, 100),
        (104, 112),
        (120, 129),
        (145, 154),
        (169, 176.8),
    ],
    "135045/iNat1208550": [
        (0, 7.5),
        (11, 20),
        (21, 30),
        (33, 41),
        (44.5, 53),
        (58, 66),
        (71, 81),
        (84, 94),
        (94, 104),
        (107, 116.5),
        (120, 130),
    ],
    "135045/iNat1208551": [
        (6, 16),
        (15, 52),
        (55, 65),
        (67, 97),
        (101, 111),
        (116, 126),
        (128, 149),
        (147, 161),
        (160, 169),
        (170, 183.5),
        (185, 200),
    ],
    "135045/iNat1208552": [
        (0, 13),
        (15, 74),
        (84, 95),
        (95, 116),
        (123, 138),
        (136.5, 148),
        (149, 159),
        (158, 169),
        (171, 183),
        (189, 203),
    ],
    "135045/iNat1208572": [
        (0, 30),
        (30, 41),
        (39, 51),
        (57, 71),
        (74, 84.5),
        (86.5, 96.5),
        (97, 107),
        (106, 117.5),
        (118, 136),
        (138, 152),
        (149, 198),
        (197, 212.2),
    ],
    "135045/iNat327127": [(0, 9)],
    "135045/iNat48803": [(0, 8), (20.5, 31), (44, 51.3)],
    "norscr1/XC146508": [0, 6, 13, 19, 26, 30, 35, 42, 48, 53, 61, 63, 64, 69, 80, 87, 99, 107],
    "norscr1/XC148047": [2, 6, 20, 24, 28, 41, 46, 57, 65, 69, 76, 103, 108, 112, 116, (118, 136)],
    "norscr1/XC178590": [1, 5, 9, 12, 17, 21, 28, 35, 39, 43, 48, 58, 62, 66, 70, 75, 80, 86, 91],
    "norscr1/XC178594": [2, 5, 10, 17, 25, 31, 40, 44, 50, 55, 63, 68, 74, 80, 91, 98, 103, 108, 111],
    "norscr1/XC178596": [(1, 51.15)],
    "norscr1/iNat31894": [(5, 17.71)],
    # 31-Mar
    "52884/CSA18797": [(16, 28), (26, 84), (92, 125), (125, 135), (141, 168), (560, 585)],
    "52884/CSA18801": [(0, 250)],
    "52884/CSA18804": [(0, 140), (560, 700), (740, 870)],
    # 2-Apr (large audios)
    "compau/XC837459": [(0, 100), (620, 720), (1120, 1220)],  # not heard; there are pauses
    "greegr/XC558126": [(0, 250)],  # not heard
    "grekis/XC936081": [(0, 250)],  # not heard
    "grekis/XC936811": [(0, 250)],  # not heard
    "saffin/XC879442": [(0, 250)],  # not heard; there are pauses
    "speowl1/XC525219": [(0, 100), (300, 400), (600, 700)],  # not heard
    "stbwoo2/XC709416": [(0, 100), (240, 340), (660, 760)],  # not heard
    "yercac1/XC245490": [(0, 255)],  # not heard
}


def complete_audio_filters(raw_audio_filters):
    """Finalize a dict of audio filters by converting audio hits into sections (start, end)."""
    BAND = 4
    audio_filters = {}
    for id_, raw_sections in raw_audio_filters.items():
        sections = []
        prior_hit = -100
        for sec in raw_sections:
            if isinstance(sec, tuple):
                sections.append(sec)
                prior_hit = sec[1] - BAND
            else:
                if prior_hit + 5 >= sec:  # Merge with previous section
                    sections[-1] = (sections[-1][0], sec + BAND)
                else:
                    start = max(sec - BAND, 0)
                    end = 5 if start == 0 else sec + BAND
                    sections.append((start, end))
                prior_hit = sec
        audio_filters[id_] = sections
    return audio_filters


class SpeechFilter:
    """Class to filter audio by removing human speech and identifying species singing."""

    def __init__(
        self,
        sing_min_duration=2,
        speech_notes_time=7,
        spech_merge_th=0.3,
        speech_min_duration=2,
        speech_start_th=8,
        th=0.5,
        threads=1,
        sr=32000,
        speech_db_th=-50,
    ):
        self.sing_min_duration = sing_min_duration
        self.speech_notes_time = speech_notes_time
        self.spech_merge_th = spech_merge_th
        self.speech_min_duration = speech_min_duration
        self.speech_start_th = speech_start_th
        self.th = th
        self.sr = sr
        self.speech_db_th = speech_db_th
        self.chunk_len = 0.1
        self.chunk = int(self.chunk_len * self.sr)
        torch.set_num_threads(threads)
        self.model, (self.get_speech_timestamps, _, _, _, _) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )

    def __call__(self, audio, sr, th=None):
        assert sr == self.sr
        if len(audio.shape) > 1:
            audio = audio[0]
        len_audio = len(audio)

        # Power-based detection
        chunk = self.chunk
        power = audio**2
        pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
        power = np.pad(power, (0, pad)).reshape((-1, chunk)).sum(axis=1)
        power_dB = 10 * np.log10(power)
        x = power_dB - self.speech_db_th
        start, end = 0, len_audio
        intersections = np.where(x[:-1] * x[1:] < 0)[0]
        for s, e in zip(intersections[:-1], intersections[1:]):
            if x[s] < x[s + 1] and (e - s) * self.chunk_len >= self.sing_min_duration:
                start, end = s * chunk, e * chunk
                break
            elif x[s] > x[s + 1] and s * self.chunk_len > self.speech_notes_time:
                start, end = 0, s * chunk
                break

        # Model-based detection
        threshold = th if th is not None else self.th
        speech_timestamps = self.get_speech_timestamps(
            audio[start:end], self.model, sampling_rate=self.sr, threshold=threshold
        )
        if len(speech_timestamps) > 0:
            s, e = -1e6, -1e6
            for ts in speech_timestamps:
                if ts["start"] - e < self.spech_merge_th * self.sr:  # Merge
                    e = ts["end"]
                else:
                    s, e = ts["start"], ts["end"]
                duration = (e - s) / self.sr
                start_s = (start + s) / self.sr
                if duration >= self.speech_min_duration or (duration > 0.5 and start_s >= 30):
                    if start_s <= self.speech_start_th:
                        break  # Likely a false positive
                    start, end = start, start + s
                    break
        return start, end


def run_curation_pipeline(filenames, custom_filter=None, speech_filter=None, required_sr=32000):
    """Run the curation pipeline to filter and process audio files."""
    custom_filter = custom_filter or {}
    files_edges = {}
    for filename in tqdm(filenames):
        try:
            audio, sr = torchaudio.load(filename)
            if sr != required_sr:
                resampler = Resample(orig_freq=sr, new_freq=required_sr)
                audio = resampler(audio)
                sr = required_sr
            audio = audio[0]
            id_ = "/".join(os.path.splitext(filename)[0].split("/")[-2:])
            sections = custom_filter.get(id_, None)
            if sections:
                start = int(sections[0][0] * sr)
                end = min(int(sections[-1][1] * sr), len(audio))
            elif speech_filter:
                start, end = speech_filter(audio, sr)
                end = min(end, len(audio))
            if start > 0 or end < len(audio):
                print(f"{filename} changed from ({0}, {len(audio)}) to ({start}, {end})")
            files_edges[id_] = (int(start), int(end))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            files_edges[id_] = (-1, -1)
    return files_edges


def main():
    parser = argparse.ArgumentParser(description="Curate dataset by filtering audio files.")
    parser.add_argument(
        "--file_globs", type=str, nargs="+", required=True, help="List of glob patterns to collect audio files."
    )
    parser.add_argument("--n_threads", type=int, default=1, help="Number of threads for processing.")
    parser.add_argument(
        "--output_json", type=str, required=True, help="Path to save the resulting JSON with file bounds."
    )
    args = parser.parse_args()

    train_audio_filters = complete_audio_filters(TRAIN_AUDIO_FILTERS)
    speech_filter = SpeechFilter(threads=args.n_threads)

    all_filenames = []
    for glob_pattern in args.file_globs:
        all_filenames.extend(glob(glob_pattern, recursive=True))

    print(f"Found {len(all_filenames)} audio files matching the patterns")

    all_filenames_bounds = run_curation_pipeline(
        all_filenames, custom_filter=train_audio_filters, speech_filter=speech_filter
    )

    write_json(args.output_json, all_filenames_bounds)
    print(f"Saved curated dataset bounds to {args.output_json}")


if __name__ == "__main__":
    main()
