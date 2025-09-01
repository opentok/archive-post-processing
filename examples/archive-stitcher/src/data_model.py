# copyright 2025 Vonage

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import Optional


class AlgoAudio(StrEnum):
    PEARSON = auto()


class AlgoVideo(StrEnum):
    VARLBP = auto()
    MSE = auto()
    WAVELET = auto()


@dataclass(eq=True)
class Conf:
    archive_a: Path
    archive_b: Path
    output: Path
    max_overlap: timedelta
    algo_video: AlgoVideo
    algo_audio: AlgoAudio
    debug_plot: bool
    deep_search: bool
    allow_output_overwrite: bool


@dataclass(eq=True, frozen=True)
class Fraction:
    num: int
    den: int

    def to_float(self) -> float:
        return self.num / self.den

    @staticmethod
    def fromstr(txt: str) -> 'Fraction':
        return Fraction(*map(int, txt.split('/')))

    def __str__(self) -> str:
        return f"{self.num}/{self.den}"


@dataclass(eq=True)
class Interval:
    ini: int = 0
    length: int = 0

    @property
    def end(self) -> int:
        return max(self.ini, self.ini + self.length - 1)

    def is_empty(self) -> bool:
        return self.length <= 0


@dataclass(eq=True, frozen=True)
class VideoDesc:
    codec: str
    profile: str
    width: int
    height: int
    pix_fmt: str
    level: int
    fps: Fraction
    timescale: Fraction
    has_b_frames: bool


@dataclass(eq=True, frozen=True)
class AudioDesc:
    codec: str
    sample_rate: int
    channels: int


@dataclass(eq=True, frozen=True)
class MediaDesc:
    video: Optional[VideoDesc]
    audio: Optional[AudioDesc]
    duration: timedelta


@dataclass
class FindOverlapArgs:
    duration_a: timedelta
    duration_b: timedelta
    video_desc: Optional[VideoDesc]
    audio_desc: Optional[AudioDesc]
    max_overlap: timedelta
    algo_video: AlgoVideo
    algo_audio: AlgoAudio
    debug_plot: bool
    deep_search: bool


@dataclass(eq=True, frozen=True)
class OverlapInterval:
    offset_a: timedelta = timedelta()
    offset_b: timedelta = timedelta()
    duration: timedelta = timedelta()

    def is_empty(self) -> bool:
        return self.duration.total_seconds() <= 0

    def intersection(self, other: 'OverlapInterval') -> 'OverlapInterval':
        if self.is_empty() or other.is_empty():
            return OverlapInterval()

        intersected_offset_a = max(self.offset_a, other.offset_a)
        intersected_offset_b = max(self.offset_b, other.offset_b)

        end_a = self.offset_a + self.duration
        other_end_a = other.offset_a + other.duration
        intersected_end_a = min(end_a, other_end_a)
        end_b = self.offset_b + self.duration
        other_end_b = other.offset_b + other.duration
        intersected_end_b = min(end_b, other_end_b)

        if intersected_end_a < intersected_offset_a or intersected_end_b < intersected_offset_b:
            return OverlapInterval()

        return OverlapInterval(
            offset_a=intersected_offset_a,
            offset_b=intersected_offset_b,
            duration=min(intersected_end_a - intersected_offset_a, intersected_end_b - intersected_offset_b),
        )

    def __str__(self) -> str:
        return f'offset_a: {self.offset_a}, offset_b: {self.offset_b}, duration: {self.duration}'


@dataclass(frozen=True)
class MediaOverlap:
    video: OverlapInterval = OverlapInterval()
    audio: OverlapInterval = OverlapInterval()


@dataclass
class MergeArgs:
    duration_a: timedelta
    duration_b: timedelta
    video_desc: Optional[VideoDesc]
    audio_desc: Optional[AudioDesc]
    overlap: MediaOverlap
