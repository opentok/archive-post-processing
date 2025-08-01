#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Vonage, 2025

from datetime import timedelta
from pathlib import Path

from dataclasses import dataclass
from typing import Final, Optional

import warnings

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from skimage.feature import local_binary_pattern

from .data_model import Interval, FindOverlapArgs, MediaOverlap, OverlapInterval
from .utils import printerr, raise_error

categories = [UserWarning, FutureWarning, DeprecationWarning]
for cat in categories:
    warnings.filterwarnings("ignore", category=cat)

# Resize factor for the video frames
RESIZE_FACTOR: Final[np.float32] = 0.5

# Accepted error to avoid the indetermination: anyValue/0
ACCEPTED_ERROR: Final[np.float32] = 1e-8


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


def plot_archives_relationship(values_list: list, overlap_ind: Interval):
    fig = plt.figure()

    num_archive_a_points = len(values_list)

    ax1 = fig.add_subplot(111)
    x_axis = list(range(0, num_archive_a_points))
    ax1.plot(x_axis, values_list, marker='*', linestyle='-', color='k')
    ax1.set_xlabel(f'archive_a frame number')
    ax1.set_ylabel(f'archive_b most similar frame to archive_a frames')
    ax1.tick_params(axis='x', labelcolor='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    x_start = overlap_ind.ini
    x_end = overlap_ind.end
    if x_start < len(values_list):
        slope = 0.999  # 45 degrees
        x_vals = np.linspace(x_start, x_start + len(values_list[x_start:x_end]), 100)
        y_vals = values_list[x_start] + slope * (x_vals - x_start)
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


def compute_audio_score(window_a: np.ndarray, window_b: np.ndarray, conf: FindOverlapArgs) -> tuple[float, float]:
    match conf.algo_audio:
        case conf.algo_audio.PEARSON:
            a_centered = window_a - window_a.mean(axis=1, keepdims=True)
            b_centered = window_b - window_b.mean(axis=1, keepdims=True)

            # Compute the numerator: dot product along rows
            numerator = np.sum(a_centered * b_centered, axis=1)

            # Compute the denominator: product of L2 norms along rows
            a_norm = np.linalg.norm(a_centered, axis=1)
            b_norm = np.linalg.norm(b_centered, axis=1)
            denominator = a_norm * b_norm

            correlations = numerator / (denominator + ACCEPTED_ERROR)
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


def find_longest_non_decreasing_segment(values: list) -> Interval:
    prev: Optional[int] = None

    max_init_index: int = 0
    max_length: int = 0

    curr_init_index: int = 0
    curr_length: int = 0

    for idx, value in enumerate(values):
        if prev is None or value >= prev:
            curr_length += 1
            if max_length < curr_length:
                max_init_index = curr_init_index
                max_length = curr_length
        else:
            curr_init_index = idx
            curr_length = 1

        prev = value

    return Interval(max_init_index, max_length)


def trim_init_duplicates_in_segment(values: list, interval: Interval) -> Interval:
    # It returns end_ind - 1 if all the elements are the same
    ini_ind: int = interval.ini
    end_ind: int = interval.end
    new_ini: int = next((i - 1 for i in range(ini_ind + 1, end_ind) if values[i] != values[i - 1]), end_ind - 1)
    return Interval(new_ini, end_ind - new_ini + 1)


def get_overlapping_indexes(values: list) -> Interval:
    interval: Interval = find_longest_non_decreasing_segment(values)
    return trim_init_duplicates_in_segment(values, interval)


def slide_last_chroma_a_window_over_chroma_b(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, conf: FindOverlapArgs) -> int:
    last_win_a: np.ndarray  = chroma_a[:, (chroma_a.shape[1] - win_frames):]
    last_win_a_norm: np.ndarray  = (last_win_a - np.mean(last_win_a)) / max(np.std(last_win_a), ACCEPTED_ERROR)

    most_similar_j: int = -1
    j_similarity: int = -1
    j_max_corr: int = -1
    for j in range(0, chroma_b.shape[1] - win_frames):
        window_b: np.ndarray = chroma_b[:, j:j + win_frames]
        win_b_norm: np.ndarray = (window_b - np.mean(window_b)) / max(np.std(window_b), ACCEPTED_ERROR)

        max_corr, similarity = compute_audio_score(last_win_a_norm, win_b_norm, conf)

        if (similarity > j_similarity) and (max_corr > j_max_corr):
            most_similar_j = j
            j_similarity = similarity
            j_max_corr = max_corr

    return most_similar_j


def get_overlapping_audio_indexes_from_unique_scanning(chroma_a_len: int, win_frames: int,
    most_similar_index: int) -> tuple[Interval, Interval]:
    # Hypothesis: audio overlapping starts from the first sample of the second audio signal
    win_a_index = max(0, chroma_a_len - win_frames - most_similar_index)
    overlapping_length: int = chroma_a_len - win_a_index

    return [Interval(win_a_index, overlapping_length), Interval(0, overlapping_length)]


def get_complete_chromas_similarity(chroma_a: np.ndarray, chroma_b: np.ndarray,
    win_frames: int, conf: FindOverlapArgs) -> tuple[Interval, Interval]:
    chromas_relationship: dict[int, SimilarityEntry] = {}
    for i in range(0, chroma_a.shape[1] - win_frames):
        window_a: np.ndarray = chroma_a[:, i:i + win_frames]

        for j in range(0, chroma_b.shape[1] - win_frames):
            window_b: np.ndarray = chroma_b[:, j:j + win_frames]

            # Correlation windows normalization
            win_a_norm: np.ndarray = (window_a - np.mean(window_a)) / max(np.std(window_a), ACCEPTED_ERROR)
            win_b_norm: np.ndarray = (window_b - np.mean(window_b)) / max(np.std(window_b), ACCEPTED_ERROR)

            max_corr, similarity = compute_audio_score(win_a_norm, win_b_norm, conf)

            if j not in chromas_relationship:
                chromas_relationship[j] = SimilarityEntry(index_i=i, corr=max_corr, sim=similarity)
            elif (similarity > chromas_relationship[j].sim and max_corr > chromas_relationship[j].corr):
                chromas_relationship[j] = SimilarityEntry(index_i=i, corr=max_corr, sim=similarity)

    index_values = [obj.index_i for obj in chromas_relationship.values()]
    overlap_indexes: Interval = get_overlapping_indexes(index_values)
    if overlap_indexes.is_empty():
        return OverlapInterval()  # pragma: no cover

    first_index_b: int = overlap_indexes.ini
    first_index_a: int = chromas_relationship[first_index_b].index_i
    last_index: int = overlap_indexes.ini + overlap_indexes.length - 1
    last_index_a: int = min(chromas_relationship[last_index].index_i + win_frames + 1, chroma_a.shape[1])

    overlapping_length: int = last_index_a - first_index_a

    return [Interval(first_index_a, overlapping_length), Interval(first_index_b, overlapping_length)]


def compute_overlapping_cqt(y_a: np.ndarray, y_b: np.ndarray, rate: int,
    conf: FindOverlapArgs) -> tuple[Interval, Interval]:
    # Number of sliding windows for assessing the chromas similarities
    num_audio_windows: Final[int] = 5

    # Compute 12 chroma features (pitch classes) from Constant-Q Transform
    chroma_a: np.ndarray = librosa.feature.chroma_cqt(y=y_a, sr=rate)
    chroma_b: np.ndarray = librosa.feature.chroma_cqt(y=y_b, sr=rate)

    win_frames: int = int(min(chroma_a.shape[1], chroma_b.shape[1]) / num_audio_windows)

    if not conf.deep_search:
        most_similar_index: int = slide_last_chroma_a_window_over_chroma_b(chroma_a, chroma_b, win_frames, conf)
        if (most_similar_index > -1):
            return get_overlapping_audio_indexes_from_unique_scanning(chroma_a.shape[1], win_frames, most_similar_index)

    return get_complete_chromas_similarity(chroma_a, chroma_b, win_frames, conf)


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

    assert(sr_a == sr_b and sr_a == rate)

    overlap_indeces_a, overlap_indeces_b = compute_overlapping_cqt(y_a, y_b, rate, conf)
    if (overlap_indeces_a.end == 0 or overlap_indeces_b.end == 0):
        return OverlapInterval()  # pragma: no cover

    start_time_a = librosa.frames_to_time(overlap_indeces_a.ini, sr=rate)
    start_time_b = librosa.frames_to_time(overlap_indeces_b.ini, sr=rate)
    end_time_a = librosa.frames_to_time(overlap_indeces_a.end, sr=rate)
    end_time_b = librosa.frames_to_time(overlap_indeces_b.end, sr=rate)

    if conf.debug_plot:
        print(f"Best alignment for audio_a starts at sec: {(start_time_a + offset_a.total_seconds()):.2f}s")
        print(f"Best alignment for audio_a ends at sec: {(end_time_a + offset_a.total_seconds()):.2f}")
        print(f"Best alignment for audio_b starts at sec: {start_time_b:.2f}s")
        print(f"Best alignment for audio_b ends at sec: {end_time_b:.2f}{end_time_b:.2f}s")

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
    overlap_indexes: Interval = get_overlapping_indexes(values_b_list)

    if overlap_indexes.is_empty():
        return OverlapInterval()

    start_overlap_a: int = overlap_data_df.loc[overlap_indexes.ini, 'frame_a']
    start_overlap_b: int = overlap_data_df.loc[overlap_indexes.ini, 'similar_frame_b']
    end_overlap_a: int = overlap_data_df.loc[overlap_indexes.end, 'frame_a']

    if conf.debug_plot:
        end_time_b = (start_overlap_b + (end_overlap_a - start_overlap_a + 1)) / rate
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
