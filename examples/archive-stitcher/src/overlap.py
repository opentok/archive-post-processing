#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Vonage, 2025

from datetime import timedelta
from pathlib import Path

from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional

import warnings

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from skimage.feature import local_binary_pattern
from skopt import gp_minimize
from skopt.space import Integer

from .data_model import Interval, FindOverlapArgs, MediaOverlap, OverlapInterval
from .utils import printerr, raise_error

categories = [UserWarning, FutureWarning, DeprecationWarning]
for cat in categories:
    warnings.filterwarnings("ignore", category=cat)

# Resize factor for the video frames
RESIZE_FACTOR: Final[np.float32] = 0.5

# Accepted error to avoid the indetermination: anyValue/0
ACCEPTED_ERROR: Final[np.float32] = 1e-8

# Minimun accepted signal length
MIN_SIGNAL_LEN: Final[int] = 4

# Signal length percentage to cap the minimum length allowed when finding overlapping periods
THRESHOLD_LENGTH_PERCENTAJE: Final[int] = 0.1  # 10%

# Percentage of allowed outliers when selecting the longest non-decreasing segments
THRESHOLD_OUTLIERS_PERCENTAJE: Final[int] = 0.1  # 10%

# The number of samples between successive frames (librosa default value)
HOP_LENGTH: Final[int] = 512

# Number of seconds covered by the similarity analysis window
WINDOW_NUM_SECS: Final[np.float32] = 1

class MediaType(Enum):
    UNDEFINED = 1
    AUDIO = 2
    VIDEO = 3

@dataclass(eq=True)
class SimilarityEntry:
    '''
    index_i: int = 0        # chroma column index
    corr: np.float32 = 0    # Max. correlation of all the 12 chromas for a fixed chroma window
    sim: np.float32 = 0     # Similarity factor between all the 12 correlation values
    '''
    index_i: int = 0
    corr: np.float32 = 0
    sim: np.float32 = 0


def plot_audio_samples(y_a: np.ndarray, y_b: np.ndarray, rate: int,
    start_time_a: float, end_time_a: float, start_time_b: float, end_time_b: float):
    _fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    librosa.display.waveshow(y_a, sr=rate, ax=ax1, marker='.', label='audio_a')
    ax1.set_xlabel('audio_a seconds')
    ax1.set_ylabel('amplitude')
    ax1.axvline(x=start_time_a, color='g', linestyle='--')
    ax1.axvline(x=end_time_a, color='g', linestyle='--')
    ax1.axvspan(start_time_a, end_time_a, color='green', alpha=0.3, label='overlapping audio samples')
    ax1.grid(True)
    ax1.legend(fontsize="x-large")
    ax1.tick_params(labelbottom=True)

    librosa.display.waveshow(y_b, sr=rate, ax=ax3, marker='.', label='audio_b')
    ax3.set_xlabel('audio_b seconds')
    ax3.set_ylabel('amplitude')
    ax3.axvline(x=start_time_b, color='g', linestyle='--')
    ax3.axvline(x=end_time_b, color='g', linestyle='--')
    ax3.axvspan(start_time_b, end_time_b, color='green', alpha=0.3, label='overlapping audio samples')
    ax3.grid(True)
    ax3.legend(fontsize="x-large")
    ax3.tick_params(labelbottom=True)

    plt.title('Audio samples display')
    plt.tight_layout()

    _fig1 = plt.gcf()
    plt.show()
    plt.draw()
    plt.close()


def plot_archives_relationship(values: list[int], overlap_ind: Interval):
    fig = plt.figure()

    num_archive_a_points = len(values)

    ax1 = fig.add_subplot(111)
    x_axis = list(range(0, num_archive_a_points))
    ax1.plot(x_axis, values, marker='*', linestyle='-', color='k')
    ax1.set_xlabel(f'archive_a frame number')
    ax1.set_ylabel(f'archive_b most similar frame to archive_a frames')
    ax1.tick_params(axis='x', labelcolor='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    x_start = overlap_ind.ini
    x_end = overlap_ind.end
    if x_start < len(values):
        slope = 0.999  # 45 degrees
        x_vals = np.linspace(x_start, x_start + len(values[x_start:x_end]), 100)
        y_vals = values[x_start] + slope * (x_vals - x_start)
        ax1.plot(x_vals, y_vals, label='45° Line', linestyle='--', color='cyan')

    ax1.axvline(x=x_start, color='g', linestyle='--')
    ax1.axvline(x=x_end, color='g', linestyle='--')
    ax1.axvspan(x_start, x_end, color='green', alpha=0.3, label='overlapping video frames')

    plt.title(f'Most similar frames')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    plt.draw()
    plt.close()


def process_frame(frame: np.ndarray, width: int, height: int, resize_prc=RESIZE_FACTOR):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (width > 320) and (height > 240):
        assert(resize_prc < 1)
        return cv2.resize(gray_image, (int(resize_prc * width), int(resize_prc * height) ))

    return gray_image  # pragma: no cover


def compute_video_score(frame_a: np.ndarray, frame_b: np.ndarray, conf: FindOverlapArgs) -> float:
    res_frame_a = process_frame(frame_a, conf.video_desc.width, conf.video_desc.height)
    res_frame_b = process_frame(frame_b, conf.video_desc.width, conf.video_desc.height)

    gradient = res_frame_a - res_frame_b

    match conf.algo_video:
        case conf.algo_video.VARLBP:
            lbp = local_binary_pattern(gradient, P=25, R=1, method="var")
            return np.sum(lbp[lbp < 15])
        case conf.algo_video.UNILBP:
            lbp = local_binary_pattern(gradient, P=8, R=1, method="uniform")
            return np.sum(lbp[lbp < 4])
        case conf.algo_video.MSE:
            return np.mean((np.abs(gradient)) ** 2)
        case conf.algo_video.WAVELET:
            _, (lh, hl, hh) = pywt.dwt2(gradient, 'haar')
            freq_details = lh + hl + hh
            res = np.asarray(np.abs(freq_details))
            max_res = np.max(res)
            return np.sum(res[res < (max_res/4)])


def open_video(archive_path: Path, frame_ini: int) -> cv2.VideoCapture:
    video = cv2.VideoCapture(str(archive_path))
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)

    if not video.isOpened():
        raise_error(f'The video {archive_path} is not opened')  # pragma: no cover

    return video


def release_video(video: cv2.VideoCapture):
    if video.isOpened():
        video.release()


def snr_db(signal_1: np.ndarray, signal_2: np.ndarray) -> float:
    '''
    Compute the Signal-to-Noise Ratio (SNR) in decibels.

    Parameters:
        signal_1: The original signal_1
        signal_2: The original signal_2
            The noise is computed as: signal_1 - signal_2

    Returns:
        float: SNR value in decibels (dB).
    '''
    len_signal_1 = signal_1.shape[1]
    len_signal_2 = signal_2.shape[1]

    signal_1 = signal_1[:, 0:min(len_signal_1, len_signal_2)]
    signal_2 = signal_2[:, 0:min(len_signal_1, len_signal_2)]

    signal_power: np.ndarray = np.sum(np.asarray(signal_1) ** 2)
    noise_power: np.ndarray = np.sum(np.asarray(signal_1 - signal_2) ** 2)

    if noise_power == 0:
        return -1  # pragma: no cover
    return 10 * np.log10(signal_power / noise_power)


def compute_audio_score(window_a: np.ndarray, window_b: np.ndarray, conf: FindOverlapArgs) -> tuple[float, float]:
    # Both windows have 12 rows with equal or approximately win_frames columns, i.e, (12 rows, win_frames columns)
    match conf.algo_audio:
        case conf.algo_audio.PEARSON:
            if (window_a.shape[1] == 0 or window_b.shape[1] == 0):
                return 0.0, 0.0  # pragma: no cover

            # axis=1: <signal>.shape[1] values for each row in <signal>.shape[0].
            a_centered = window_a - window_a.mean(axis=1, keepdims=True)
            b_centered = window_b - window_b.mean(axis=1, keepdims=True)

            # Force both signals to have the same size
            len_a_centered = a_centered.shape[1]
            len_b_centered = b_centered.shape[1]
            a_centered = a_centered[:, 0:min(len_a_centered, len_b_centered)]
            b_centered = b_centered[:, 0:min(len_a_centered, len_b_centered)]

            # Compute the numerator: sum accros columns for each row
            numerator = np.sum(a_centered * b_centered, axis=1)
            # Compute the denominator
            denominator = np.sqrt(np.sum(a_centered**2, axis=1) * np.sum(b_centered**2, axis=1))

            correlations = numerator / (denominator + ACCEPTED_ERROR)

            # Pearson chromas values are constrained to the range [-1, +1].
            # Therefore, np.nanstd(pearson_chromas) is constrained to the range [0, 1]
            # Similarity values closer to 1 have high similarity
            similarity: np.float32 = 1 - min(np.nanstd(correlations) / 0.5, 1.0)

            return np.nanmax(correlations), similarity


def get_matching_frames(archive_a: Path, archive_b: Path, conf: FindOverlapArgs) -> pd.DataFrame:
    rate: float = conf.video_desc.fps.to_float()

    frame_offset_a: int = int(max(timedelta(), conf.duration_a - conf.max_overlap).total_seconds() * rate)
    frame_count_a: int = int(conf.duration_a.total_seconds() * rate)
    frame_count_b: int = int(min(conf.duration_b, conf.max_overlap).total_seconds() * rate)

    video_a: cv2.VideoCapture = open_video(archive_a, frame_offset_a)
    video_b: cv2.VideoCapture = open_video(archive_b, 0)

    frames_relation_list = list()
    for frame_a_num in range(frame_offset_a + 1, frame_count_a + 1):
        ret_a, frame_a = video_a.read()
        if not ret_a:
            break

        score_frames_b_list = list()

        for frame_b_num in range(1, frame_count_b + 1):
            ret_b, frame_b = video_b.read()
            if not ret_b:
                break  # pragma: no cover

            score: float = compute_video_score(frame_a, frame_b, conf)
            score_frames_b_list.append([score, frame_b_num])

        score_df: DataFrame = pd.DataFrame(score_frames_b_list, columns=['score', 'frame_b'])

        frame_b_with_min_score: list = score_df.loc[score_df['score'] == score_df['score'].min(), 'frame_b'].tolist()
        last_similar_frame_b_to_frame_a: int = frame_b_with_min_score[-1]
        frames_relation_list.append([frame_a_num, last_similar_frame_b_to_frame_a])

        video_b.set(cv2.CAP_PROP_POS_FRAMES, 0)

    similar_frames_df = pd.DataFrame(frames_relation_list, columns=['frame_a', 'similar_frame_b'])

    release_video(video_a)
    release_video(video_b)

    return similar_frames_df


def trim_init_duplicates_in_segment(values: list[int], interval: Interval) -> Interval:
    # It returns end_ind - 1 if all the elements are the same
    ini_ind: int = interval.ini
    end_ind: int = interval.ini + interval.length
    new_ini: int = next((i - 1 for i in range(ini_ind + 1, end_ind) if values[i] != values[i - 1]), end_ind - 1)
    return Interval(new_ini, end_ind - new_ini)


def trim_end_duplicates_in_segment(values: list[int], interval: Interval) -> Interval:
    # For example, to remove final black frames
    ini_ind: int = interval.ini
    end_ind: int = interval.ini + interval.length
    values = values[ini_ind:end_ind][::-1]
    i: int = next((i for i, x in enumerate(values) if x != values[0]), len(values))
    return Interval(ini_ind, len(values[i:]) + 1)


def is_data_increasing_in_45_degrees_trend(values: list[int]) -> bool:
    if len(values) < MIN_SIGNAL_LEN:
        return False  # Too short

    diffs = [b - a for a, b in zip(values[:-1], values[1:])]

    # return True if at least 90% of all the values in the list diffs are close to one
    return sum(abs(x - 1) <= 1 for x in diffs) / len(diffs) >= 0.9


def get_increasing_data_intervals(values: list[int], interval_list: list[Interval]) -> list[Interval]:
    increasing_intervals_list = list()

    for i in range(0, len(interval_list)):
        trimmed_ini_interval: Interval = trim_init_duplicates_in_segment(values, interval_list[i])
        trimmed_interval: Interval = trim_end_duplicates_in_segment(values, trimmed_ini_interval)

        filtered_vals = values[trimmed_interval.ini:(trimmed_interval.ini + trimmed_interval.length)]

        if (is_data_increasing_in_45_degrees_trend(filtered_vals)):
            increasing_intervals_list.append(interval_list[i])

    return increasing_intervals_list


def remove_glitches(values: int, all_interval_list: list[Interval], media_type: MediaType) -> Interval:
    if (media_type.name == 'UNDEFINED'):
        return Interval()

    if (len(all_interval_list) == 1):
        return all_interval_list[0]

    interval_list: list = get_increasing_data_intervals(values, all_interval_list)

    if (len(interval_list) == 0):
        return Interval()  # pragma: no cover

    if (len(interval_list) == 1):
        return interval_list[0]

    # Initialization for the first loop
    count = 0
    final_ini_index = interval_list[0].ini
    final_end_index = interval_list[0].length

    filtered_intervals = list()

    for i in range(0, len(interval_list) - 1):
        end_index_prev = interval_list[i].ini + interval_list[i].length
        ini_index_current = interval_list[i+1].ini

        threshold = int(max(interval_list[i].length, interval_list[i+1].length) * THRESHOLD_OUTLIERS_PERCENTAJE)

        is_value_diff_low: bool = abs(values[end_index_prev - 1] - values[ini_index_current]) < threshold
        if (ini_index_current - end_index_prev) < threshold and is_value_diff_low:
            final_ini_index = interval_list[i-count].ini
            final_end_index += (interval_list[i+1].length + ini_index_current - end_index_prev)
            count += 1
        else:
            filtered_intervals.append([final_ini_index, final_end_index])
            final_ini_index = interval_list[i+1].ini
            final_end_index = interval_list[i+1].length
            count = 0

        if (i == (len(interval_list) - 2)):
            filtered_intervals.append([final_ini_index, final_end_index])

    longest_length = max(filtered_intervals, key=lambda x: x[1])

    return Interval(longest_length[0], longest_length[1])


def add_unique_element_to_list(max_init_index: int, max_length: int, segments_in_set: set[int, int],
    segments_in_list: list[Interval]) -> list[Interval]:
    key = (max_init_index, max_length)

    if key not in segments_in_set:
        # To avoid duplicity in segments_in_list
        segments_in_set.add(key)
        segments_in_list.append(Interval(max_init_index, max_length))

    return segments_in_list


def find_longest_non_decreasing_segment(values: list[int], media_type: MediaType) -> Interval:
    prev: Optional[int] = None

    max_init_index: int = 0
    max_length: int = 0

    curr_init_index: int = 0
    curr_length: int = 0

    segments_in_set = set()
    longest_segments_list = list()
    for idx, value in enumerate(values):
        if prev is None or value >= prev:
            curr_length += 1
            if max_length < curr_length or (curr_length > len(values) * THRESHOLD_LENGTH_PERCENTAJE):
                max_init_index = curr_init_index
                max_length = curr_length
            if (idx == len(values) - 1):
                add_unique_element_to_list(max_init_index, max_length, segments_in_set, longest_segments_list)
        else:
            add_unique_element_to_list(max_init_index, max_length, segments_in_set, longest_segments_list)
            curr_init_index = idx
            curr_length = 1

        prev = value

    if (len(longest_segments_list) == 0):
        return Interval()  # pragma: no cover

    return remove_glitches(values, longest_segments_list, media_type)


def get_overlapping_indexes(values: list[int], media_type: MediaType) -> Interval:
    interval_pre_trimming: Interval = find_longest_non_decreasing_segment(values, media_type)
    if (interval_pre_trimming.is_empty()):
        return Interval()  # pragma: no cover

    # Remove not monotically increasing values at the beginning and at the end of the overlapping period
    interval_ini_trimmed: Interval = trim_init_duplicates_in_segment(values, interval_pre_trimming)
    interval_trimmed: Interval = trim_end_duplicates_in_segment(values, interval_ini_trimmed)

    reduced_values = values[interval_trimmed.ini:interval_trimmed.ini + interval_trimmed.length]
    jump_in_frames: int = reduced_values[len(reduced_values) - 1] - reduced_values[len(reduced_values) - 2]
    length: int = interval_trimmed.length if jump_in_frames < 3 else interval_trimmed.length - 1

    return Interval(interval_trimmed.ini, length)


def update_last_index_a_precision(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, index_a: int, index_b: int) -> int:
    # 100ms of window
    small_win = max(int((win_frames*0.1)/WINDOW_NUM_SECS), 2)

    snr = list()
    for j in range(0, win_frames):
        window_a: np.ndarray = chroma_a[:, index_a + j:min(index_a + j + small_win, index_a + win_frames)]
        window_b: np.ndarray = chroma_b[:, index_b + j:min(index_b + j + small_win, index_b + win_frames)]

        if (window_a.shape[1] < MIN_SIGNAL_LEN or window_b.shape[1] < MIN_SIGNAL_LEN):
            break
        # Windows normalization
        win_a_norm: np.ndarray = (window_a - np.mean(window_a)) / max(np.std(window_a), ACCEPTED_ERROR)
        win_b_norm: np.ndarray = (window_b - np.mean(window_b)) / max(np.std(window_b), ACCEPTED_ERROR)

        snr.append(snr_db(win_a_norm, win_b_norm))

    neg_indices = [i for i, v in enumerate(snr) if v < 0]
    last_index_a: int = (index_a + neg_indices[0]) if neg_indices else (index_a + win_frames)
    return min(last_index_a, chroma_a.shape[1])


def compute_deep_audio_algorithm(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, conf: FindOverlapArgs) -> tuple[Interval, Interval]:
    chromas_relationship: dict[int, SimilarityEntry] = {}
    for i in range(0, chroma_a.shape[1] - win_frames):
        window_a: np.ndarray = chroma_a[:, i:i + win_frames]

        for j in range(0, chroma_b.shape[1] - win_frames):
            window_b: np.ndarray = chroma_b[:, j:j + win_frames]

            # Windows normalization
            win_a_norm: np.ndarray = (window_a - np.mean(window_a)) / max(np.std(window_a), ACCEPTED_ERROR)
            win_b_norm: np.ndarray = (window_b - np.mean(window_b)) / max(np.std(window_b), ACCEPTED_ERROR)

            max_corr, similarity = compute_audio_score(win_a_norm, win_b_norm, conf)

            if j not in chromas_relationship:
                chromas_relationship[j] = SimilarityEntry(index_i=i, corr=max_corr, sim=similarity)
            elif (similarity > chromas_relationship[j].sim and max_corr > chromas_relationship[j].corr):
                chromas_relationship[j] = SimilarityEntry(index_i=i, corr=max_corr, sim=similarity)

    index_values = [obj.index_i for obj in chromas_relationship.values()]
    overlap_indexes: Interval = get_overlapping_indexes(index_values, MediaType.AUDIO)
    if overlap_indexes.is_empty():
        return OverlapInterval()  # pragma: no cover

    first_index_b: int = overlap_indexes.ini
    first_index_a: int = chromas_relationship[first_index_b].index_i
    last_index: int = overlap_indexes.ini + overlap_indexes.length - 1

    last_index_a: int = min(chromas_relationship[last_index].index_i, chroma_a.shape[1] - win_frames)
    last_index_a = update_last_index_a_precision(chroma_a, chroma_b, win_frames, last_index_a, last_index)

    overlapping_length: int = last_index_a - first_index_a

    return [Interval(first_index_a, overlapping_length), Interval(first_index_b, overlapping_length)]


def select_by_difference(list_1: list, list_2: list, is_last_index: bool) -> list:
    if (list_1 == list_2):
        return list_1

    diff_first = abs(list_1[0] - list_2[0])
    diff_second = abs(list_1[1] - list_2[1])

    if diff_second > diff_first:
        return list_1 if is_last_index else list_2
    else:
        return list_2 if is_last_index else list_1


def get_best_row(data: list, is_last_index: bool) -> tuple[int, int]:
    def round1(x): return round(x, 1)

    rounded_third = max(round1(row[2]) for row in data)
    max_rows = [row for row in data if round1(row[2]) == rounded_third]

    if max_rows:
        first_a_element = max(row[0] for row in max_rows) if is_last_index else min(row[0] for row in max_rows)
        first_b_element = max(row[1] for row in max_rows) if is_last_index else min(row[1] for row in max_rows)

        best_rows_a = [row for row in max_rows if row[0] == first_a_element]
        best_rows_b = [row for row in max_rows if row[1] == first_b_element]
        result = select_by_difference(best_rows_a[0], best_rows_b[0], is_last_index)

        return result[0], result[1]

    return -1, -1  # pragma: no cover


def get_highest_similarity(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, index_a: int, conf: FindOverlapArgs) -> tuple[int, float, float]:
    chromas_relationship: dict[int, SimilarityEntry] = {}

    for i in range(index_a, index_a + win_frames):
        window_a: np.ndarray = chroma_a[:, i:min(i + win_frames, chroma_a.shape[1])]
        if (window_a.shape[1] < MIN_SIGNAL_LEN):
            break
        win_a_norm: np.ndarray  = (window_a - np.mean(window_a)) / max(np.std(window_a), ACCEPTED_ERROR)

        for j in range(0, chroma_b.shape[1] - win_frames):
            window_b: np.ndarray = chroma_b[:, j:j + win_frames]
            win_b_norm: np.ndarray = (window_b - np.mean(window_b)) / max(np.std(window_b), ACCEPTED_ERROR)

            max_corr, similarity = compute_audio_score(win_a_norm, win_b_norm, conf)

            if j not in chromas_relationship:
                chromas_relationship[j] = SimilarityEntry(index_i=i, corr=max_corr, sim=similarity)
            elif (similarity > chromas_relationship[j].sim and max_corr > chromas_relationship[j].corr):
                chromas_relationship[j] = SimilarityEntry(index_i=i, corr=max_corr, sim=similarity)

    sorted_items: list = sorted(
        chromas_relationship.items(),
        key=lambda item: (item[1].corr, item[1].sim)
    )
    key, entry = sorted_items[-1:][0]

    return key, entry.index_i, entry.corr, entry.sim


def compute_partial_audio_indices(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, is_last_index: bool, conf: FindOverlapArgs) -> tuple[int, int]:
    count = 1 if is_last_index else 0

    indices_list = list()
    while (chroma_a.shape[1] > count * win_frames):
        last_index_a: int = (chroma_a.shape[1] - count * win_frames) if is_last_index else count * win_frames
        index_b, index_a, corr, sim = get_highest_similarity(chroma_a, chroma_b, win_frames, last_index_a, conf)
        indices_list.append([index_a, index_b, corr, sim])
        count += 1

    return get_best_row(indices_list, is_last_index)


def compute_partial_audio_algorithm(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, conf: FindOverlapArgs) -> tuple[Interval, Interval]:
    first_index_a, first_index_b = compute_partial_audio_indices(chroma_a, chroma_b, win_frames, False, conf)

    if (first_index_a > -1):
        last_index_a, last_index_b = compute_partial_audio_indices(chroma_a, chroma_b, win_frames, True, conf)

        if (last_index_a > -1):
            last_index_a = update_last_index_a_precision(chroma_a, chroma_b, win_frames, last_index_a, last_index_b)

            overlapping_length: int = last_index_a - first_index_a

            return [Interval(first_index_a, overlapping_length), Interval(first_index_b, overlapping_length)]

    return [Interval(), Interval()]  # pragma: no cover


def compute_overlapping_cqt(y_a: np.ndarray, y_b: np.ndarray, rate: int,
    conf: FindOverlapArgs) -> tuple[Interval, Interval]:
    # Compute 12 chroma features (pitch classes) from Constant-Q Transform
    chroma_a: np.ndarray = librosa.feature.chroma_cqt(y=y_a, sr=rate, hop_length=HOP_LENGTH)
    chroma_b: np.ndarray = librosa.feature.chroma_cqt(y=y_b, sr=rate, hop_length=HOP_LENGTH)

    # compute mean square error between the two chromas
    if (chroma_a.shape[1] < MIN_SIGNAL_LEN or chroma_b.shape[1] < MIN_SIGNAL_LEN):
        printerr('The audio samples are too short or too silent to compute the chroma features')
        return Interval(), Interval()  # pragma: no cover

    len_a = chroma_a.shape[1]
    len_b = chroma_b.shape[1]
    def get_chroma_intervals_from_offset(offset: int) -> tuple[Interval, Interval]:
        # offset determines how much chroma_b overlaps chroma_a starting on the right of chroma_a
        # offset == 0 means chroma_b starts exactly at the start of chroma_a
        # offset > 0 means chroma_b starts inside of chroma_a
        # offset < 0 means chroma_a starts inside of chroma_b

        # so, lets imagine a coordinate axis in which we place the start of chroma_b at 0 and the end of chroma_a is at 0 + offset
        start_a_in_axis = offset - len_a
        end_a_in_axis = offset
        start_b_in_axis = 0
        end_b_in_axis = len_b

        overlap_start_in_axis = max(start_a_in_axis, start_b_in_axis)
        overlap_end_in_axis = min(end_a_in_axis, end_b_in_axis)

        if overlap_end_in_axis <= overlap_start_in_axis:
            return Interval(), Interval()  # pragma: no cover

        interval_a_start = overlap_start_in_axis - start_a_in_axis
        interval_a_end = overlap_end_in_axis - start_a_in_axis

        interval_b_start = overlap_start_in_axis - start_b_in_axis
        interval_b_end = overlap_end_in_axis - start_b_in_axis

        return Interval(interval_a_start, interval_a_end - interval_a_start), Interval(interval_b_start, interval_b_end - interval_b_start)

    def optimize_step(offset: list[int]) -> float:
        offset = int(offset[0])
        interval_a, interval_b = get_chroma_intervals_from_offset(offset)

        res = np.mean((chroma_a[:, interval_a.ini:interval_a.end + 1] - chroma_b[:, interval_b.ini:interval_b.end + 1]) ** 2)
        print(f"Offset: {offset}, MSE: {res}, IntervalA: {interval_a}, IntervalB: {interval_b}")
        return res

    print("Optimizing the chroma alignment...")
    result = gp_minimize(optimize_step, [Integer(MIN_SIGNAL_LEN,  chroma_a.shape[1] + chroma_b.shape[1] - MIN_SIGNAL_LEN)])
    print(result)
    optimal_offset = int(result.x[0])
    return get_chroma_intervals_from_offset(optimal_offset)

def find_overlap_audio(archive_a: Path, archive_b: Path, conf: FindOverlapArgs) -> OverlapInterval:
    rate: float = conf.audio_desc.sample_rate

    offset_a: timedelta = max(timedelta(), conf.duration_a - conf.max_overlap)
    y_a, sr_a = librosa.load(str(archive_a), sr=int(rate), mono=True,
                             offset=offset_a.total_seconds(),
                             duration=(conf.duration_a - offset_a).total_seconds())

    duration_b: timedelta = min(conf.max_overlap, conf.duration_b)
    y_b, sr_b = librosa.load(str(archive_b), sr=int(rate), mono=True,
                             offset=0,
                             duration=duration_b.total_seconds())

    # trim silence at the beginning and at the end of both audio samples
    y_a_trimmed, y_a_trimmed_index = librosa.effects.trim(y_a, top_db=20)
    y_b_trimmed, y_b_trimmed_index = librosa.effects.trim(y_b, top_db=20)

    assert(sr_a == sr_b and sr_a == rate)

    overlap_indeces_a, overlap_indeces_b = compute_overlapping_cqt(y_a_trimmed, y_b_trimmed, rate, conf)
    if (overlap_indeces_a.end == 0 or overlap_indeces_b.end == 0):
        return OverlapInterval()  # pragma: no cover

    # correct the indices due to the trim operation
    overlap_indeces_a.ini += y_a_trimmed_index[0]
    overlap_indeces_b.ini += y_b_trimmed_index[0]

    start_time_a = librosa.frames_to_time(overlap_indeces_a.ini, sr=rate, hop_length=HOP_LENGTH)
    start_time_b = librosa.frames_to_time(overlap_indeces_b.ini, sr=rate, hop_length=HOP_LENGTH)
    end_time_a = librosa.frames_to_time(overlap_indeces_a.end, sr=rate, hop_length=HOP_LENGTH)
    end_time_b = librosa.frames_to_time(overlap_indeces_b.end, sr=rate, hop_length=HOP_LENGTH)

    if conf.debug_plot or conf.deep_debug_plot:
        print(f"Best alignment for audio_a starts at sec: {(start_time_a + offset_a.total_seconds()):.2f}s")
        print(f"Best alignment for audio_a ends at sec: {(end_time_a + offset_a.total_seconds()):.2f}")
        print(f"Best alignment for audio_b starts at sec: {start_time_b:.2f}s")
        print(f"Best alignment for audio_b ends at sec: {end_time_b:.2f}s")  # pragma: no cover

        plot_audio_samples(y_a, y_b, rate, start_time_a, end_time_a, start_time_b, end_time_b)

    return OverlapInterval(
            offset_a=timedelta(seconds=start_time_a) + offset_a,
            offset_b=timedelta(seconds=start_time_b),
            duration=timedelta(seconds=(end_time_a - start_time_a))
            )


def find_overlap_video(archive_a: Path, archive_b: Path, conf: FindOverlapArgs) -> OverlapInterval:
    rate: float = conf.video_desc.fps.to_float()

    overlap_data_df: pd.DataFrame = get_matching_frames(archive_a, archive_b, conf)
    if overlap_data_df.empty:
        return OverlapInterval()  # pragma: no cover

    values_b_list: list[int] = overlap_data_df['similar_frame_b'].tolist()
    overlap_indexes: Interval = get_overlapping_indexes(values_b_list, MediaType.VIDEO)

    if overlap_indexes.is_empty():
        return OverlapInterval()  # pragma: no cover

    start_overlap_a: int = overlap_data_df.loc[overlap_indexes.ini, 'frame_a']
    start_overlap_b: int = overlap_data_df.loc[overlap_indexes.ini, 'similar_frame_b']
    end_overlap_a: int = overlap_data_df.loc[overlap_indexes.end, 'frame_a']

    if conf.debug_plot or conf.deep_debug_plot:
        end_time_b = (start_overlap_b + (end_overlap_a - start_overlap_a)) / rate  # pragma: no cover
        print(f"Best alignment for video_a starts at sec: {(start_overlap_a / rate):.2f}s")
        print(f"Best alignment for video_a ends at sec: {(end_overlap_a / rate):.2f}s")
        print(f"Best alignment for video_b starts at sec: {(start_overlap_b / rate):.2f}s")
        print(f"Best alignment for video_b ends at sec: {end_time_b:.2f}s")

        plot_archives_relationship(values_b_list, overlap_indexes)

    return OverlapInterval(
            offset_a=timedelta(seconds=start_overlap_a / rate),
            offset_b=timedelta(seconds=start_overlap_b / rate),
            duration=timedelta(seconds=(end_overlap_a - start_overlap_a + 1) / rate)
            )


def combine_audio_with_video_overlap(audio_overlap: OverlapInterval, video_overlap: OverlapInterval) -> MediaOverlap:
    if audio_overlap.is_empty() and not video_overlap.is_empty():
        return MediaOverlap(audio=OverlapInterval(), video=video_overlap)

    if not audio_overlap.is_empty() and video_overlap.is_empty():
        return MediaOverlap(audio=audio_overlap, video=OverlapInterval())

    audio_video_overlap_intersection: OverlapInterval = audio_overlap.intersection(video_overlap)

    # if the video overlap does not match at all the audio overlap, makes more sense to use the audio overlap
    # also, if the video/audio overlap intersection is much shorter than the video overlap, we consider it suspicious
    suspicious_video_overlap: bool = (
            audio_video_overlap_intersection.is_empty() or
            video_overlap.duration - audio_video_overlap_intersection.duration > timedelta(seconds=0.15)
            )
    if suspicious_video_overlap:
        printerr(f'Suspicious video overlap: {video_overlap} compared to audio_overlap: {audio_overlap} '
                 f'with intersection {audio_video_overlap_intersection}, using audio overlap instead')
        return MediaOverlap(audio=audio_overlap, video=audio_overlap)
    else:
        return MediaOverlap(audio=audio_overlap, video=video_overlap)


def find_overlap(archive_a: Path, archive_b: Path, conf: FindOverlapArgs) -> MediaOverlap:
    assert(conf.video_desc or conf.audio_desc)

    audio_overlap = OverlapInterval()
    video_overlap = OverlapInterval()

    if conf.audio_desc is not None:
        audio_overlap: OverlapInterval = find_overlap_audio(archive_a, archive_b, conf)

    if conf.video_desc is not None:
        video_overlap: OverlapInterval = find_overlap_video(archive_a, archive_b, conf)

    return combine_audio_with_video_overlap(audio_overlap, video_overlap)
