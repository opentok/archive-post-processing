# copyright 2025 Vonage

import argparse
import json
import random
import tempfile

from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from enum import StrEnum, auto
from functools import reduce
from math import ceil, floor
from pathlib import Path
from typing import Optional

from src.data_model import Fraction, MediaOverlap, OverlapInterval
from src.merge import (
        concatenate_media_files, join_audio_and_video_outputs, get_next_best_keyframe_ts_in_interval, merge_audio,
        merge_video, normalize_video_level, normalize_video_profile)
from src.utils import create_tempfile, FFMPEG, FFPROBE, printerr, run_exec
from src.validations import AudioDesc, get_media_desc, MediaDesc, validate_tools, VideoDesc


class SampleType(StrEnum):
    FULL = auto()
    AUDIO_ONLY = auto()
    VIDEO_ONLY = auto()
    VIDEO_WITH_SILENT_AUDIO = auto()
    AUDIO_WITH_STATIC_VIDEO = auto()
    NO_OVERLAP = auto()


class Effect(StrEnum):
    BLACK_FRAME_AT_END_OF_A = auto()
    BLACK_FRAME_AT_INI_OF_B = auto()
    GLITCH_AT_END_OF_A = auto()
    GLITCH_AT_INI_OF_B = auto()


@dataclass
class Conf:
    archive: Path
    min_length: timedelta
    max_length: timedelta
    min_overlap: timedelta
    max_overlap: timedelta
    av_sync_difference: timedelta
    output_prefix: Path
    sample_type: SampleType
    effects: set[Effect]
    random_seed: int


MAX_BLACK_FRAME_DURATION = timedelta(seconds=3)
MAX_GLITCH_DURATION = timedelta(milliseconds=100)
MAX_GLITCH_COUNT = 3


@dataclass
class MediaSource:
    source_file: Path
    offset: timedelta
    duration: timedelta


class MediaSegmentType(StrEnum):
    REF_FRAME = auto()
    SILENCE = auto()


class MediaSegmentAction(StrEnum):
    REPLACE = auto()
    ADD = auto()


class MediaType(StrEnum):
    AUDIO = auto()
    VIDEO = auto()


@dataclass(order=True, frozen=True, eq=True)
class MediaModification:
    insertion_point: timedelta
    media_type: MediaType
    segment_type: MediaSegmentType
    action: MediaSegmentAction
    duration: timedelta


@dataclass
class OutputMediaDesc:
    source: MediaSource

    output: Path

    has_audio: bool
    has_video: bool

    audio_ref_frame: timedelta = timedelta()
    video_ref_frame: timedelta = timedelta()

    media_modifications: list[MediaModification] = field(default_factory=list)

    audio_video_desync: timedelta = timedelta()


@dataclass
class OutputDesc:
    # segment of the source file that we'll consider for the output
    source: MediaSource
    overlap: MediaOverlap

    # description of the output that will have the reference "stitched" media
    output_ref: OutputMediaDesc
    # description of the output that will have the first part of the source media
    output_a: OutputMediaDesc
    # description of the output that will have the second part of the source media
    output_b: OutputMediaDesc

    # here is where the metadata about the output will be stored in a JSON file
    output_meta_path: Path


def video_codec_params(video_desc: VideoDesc) -> list[str]:
    return [
        '-profile:v', normalize_video_profile(video_desc.profile),
        '-level', normalize_video_level(video_desc.level),
        '-r', str(video_desc.fps),
        '-fps_mode', 'cfr',
        '-pix_fmt', 'yuv420p',
        '-video_track_timescale', str(video_desc.timescale.den),
        '-x264-params', f'fps={video_desc.fps}:timebase={video_desc.timescale}',
    ]


def floor_sample_ts(ts: timedelta, sample_per_second: Fraction) -> timedelta:
    float_value = sample_per_second.to_float()
    return timedelta(seconds=floor(ts.total_seconds() *  sample_per_second.to_float()) / sample_per_second.to_float())


def generate_video_glitch(desc: VideoDesc, source_file: Path, ref_frame: timedelta, duration: timedelta, tmpdir: Path
                          ) -> Path:
    png_path = create_tempfile(suffix='_frame.png', dir=tmpdir)
    video_path = create_tempfile(suffix='_glitch.mp4', dir=tmpdir)

    run_exec(FFMPEG,
             '-ss', ref_frame,
             '-i', source_file,
             '-frames:v', '1',
             '-q:v', '2',
             '-y',
             png_path)
    run_exec(FFMPEG,
             '-loop', '1',
             '-i', png_path,
             '-t', duration,
             *video_codec_params(desc),
             '-y',
             video_path)

    return video_path


def generate_black_video(video_desc: VideoDesc, duration: timedelta, tmpdir: Path) -> Path:
    path = create_tempfile(suffix='_black.mp4', dir=tmpdir)
    run_exec(FFMPEG,
             '-f', 'lavfi',
             '-i', 'color=c=black:'
                  f's={video_desc.width}x{video_desc.height}:'
                  f'r={video_desc.fps}:'
                  f'd={duration.total_seconds()}',
             '-t', duration,
             *video_codec_params(video_desc),
             '-y',
             path)
    return path


def copy_video_segment(video_desc: VideoDesc, source_file: Path, start: timedelta, duration: timedelta, tmpdir: Path
                       ) -> Path:
    next_keyframe_pts: timedelta = get_next_best_keyframe_ts_in_interval(
            source_file, start, duration)

    parts: list[Path] = []
    if start != next_keyframe_pts:
        pre_segment_path = create_tempfile(suffix='_presegment.mp4', dir=tmpdir)
        run_exec(FFMPEG, '-i', source_file,
                 '-ss', start,
                 '-to', min(start + duration, next_keyframe_pts - timedelta(seconds=1 / video_desc.fps.to_float())),
                 *video_codec_params(video_desc),
                 '-an',
                 '-y',
                 pre_segment_path)
        parts.append(pre_segment_path)

    if next_keyframe_pts < start + duration:
        segment_path = create_tempfile(suffix='_segment.mp4', dir=tmpdir)
        run_exec(FFMPEG, '-i', source_file,
                 '-ss', next_keyframe_pts,
                 '-to', start + duration,
                 '-c:v', 'copy',
                 '-an',
                 '-y',
                 segment_path)
        parts.append(segment_path)

    if parts:
        final_path = create_tempfile(suffix='_finalsegment.mp4', dir=tmpdir)
        concatenate_media_files(parts, final_path, MediaType.VIDEO, tmpdir)
    else:
        final_path = parts[0]

    return final_path


def generate_video(video_desc: VideoDesc, output_desc: OutputMediaDesc, tmpdir: Path) -> Path:
    video_modifications: list[MediaModification] = sorted([mod
                                                           for mod in output_desc.media_modifications
                                                           if mod.media_type == MediaType.VIDEO])
    modification_index: dict[MediaModification, Path] = {}
    for mod in video_modifications:
        if mod.segment_type == MediaSegmentType.SILENCE:
            modification_index[mod] = generate_black_video(video_desc, mod.duration, tmpdir)
        elif mod.segment_type == MediaSegmentType.REF_FRAME:
            modification_index[mod] = generate_video_glitch(video_desc, output_desc.source.source_file,
                                                            output_desc.video_ref_frame, mod.duration,
                                                            tmpdir)

    video_fragments: list[tuple[Path, timedelta]] = []

    source_offset: timedelta = timedelta()
    for mod, path in modification_index.items():
        if mod.insertion_point <= source_offset:
            video_fragments.append((path, mod.duration))
            source_offset += mod.duration if mod.action == MediaSegmentAction.REPLACE else timedelta()
        else:
            # add a segment from the source file before the modification
            segment_duration = mod.insertion_point - source_offset
            segment_path = copy_video_segment(video_desc, output_desc.source.source_file, 
                                              source_offset + output_desc.source.offset,
                                              segment_duration, tmpdir)
            video_fragments.append((segment_path, segment_duration))
            source_offset += segment_duration

            # add the modification itself
            video_fragments.append((path, mod.duration))
            source_offset += mod.duration if mod.action == MediaSegmentAction.REPLACE else timedelta()

    if source_offset < output_desc.source.duration:
        # add the remaining segment from the source file
        remaining_duration = output_desc.source.duration - source_offset
        segment_path = copy_video_segment(video_desc, output_desc.source.source_file,
                                          source_offset + output_desc.source.offset,
                                          remaining_duration, tmpdir)
        video_fragments.append((segment_path, remaining_duration))

    fps_duration = timedelta(seconds=1 / video_desc.fps.to_float())

    # concatenate all video fragments
    def merge_two_fragments(f_a: tuple[Path, timedelta], f_b: tuple[Path, timedelta]) -> tuple[Path, timedelta]:
        a_path, a_duration = f_a
        b_path, b_duration = f_b

        overlap = OverlapInterval(offset_a=a_duration, offset_b=timedelta(), duration=fps_duration)
        return merge_video(a_path, b_path, overlap, video_desc, tmpdir), a_duration + b_duration

    merged_path, _ = reduce(merge_two_fragments, video_fragments)

    final_path = create_tempfile(suffix='_final_video_output.mp4', dir=tmpdir)
    merged_path.rename(final_path)
    return final_path


def generate_audio_silence(audio_desc: AudioDesc, duration: timedelta, tmpdir: Path) -> Path:
    path = create_tempfile(suffix='_silence.m4a', dir=tmpdir)
    run_exec(FFMPEG,
             '-f', 'lavfi',
             '-i', f'anullsrc=r={audio_desc.sample_rate}:cl={"mono" if audio_desc.channels == 1 else "stereo"}',
             '-t', duration,
             '-c:a', 'aac',
             '-y',
             path)
    return path


def generate_audio_glitch(audio_desc: AudioDesc, source_file: Path, ref_frame: timedelta, duration: timedelta,
                          tmpdir: Path) -> Path:
    audio_path = create_tempfile(suffix='_glitch.m4a', dir=tmpdir)
    run_exec(FFMPEG,
             '-ss', ref_frame,
             '-i', source_file,
             '-t', duration,
             '-c:a', 'aac',
             '-vn',
             '-y',
             audio_path)
    return audio_path


def copy_audio_segment(source_file: Path, start: timedelta, duration: timedelta, tmpdir: Path) -> Path:
    segment_path = create_tempfile(suffix='_segment.m4a', dir=tmpdir)
    run_exec(FFMPEG, '-i', source_file,
             '-ss', start,
             '-to', start + duration,
             '-c:a', 'copy',
             '-vn',
             '-y',
             segment_path)
    return segment_path


def generate_audio(audio_desc: AudioDesc, output_desc: OutputMediaDesc, tmpdir: Path) -> Path:
    # apply the audio video desync to the source offset
    output_desc = replace(output_desc,
                            source=replace(output_desc.source,
                                             offset=output_desc.source.offset + output_desc.audio_video_desync))

    audio_modifications: list[MediaModification] = sorted([mod
                                                           for mod in output_desc.media_modifications
                                                           if mod.media_type == MediaType.AUDIO])
    modification_index: dict[MediaModification, Path] = {}
    for mod in audio_modifications:
        if mod.segment_type == MediaSegmentType.SILENCE:
            modification_index[mod] = generate_audio_silence(audio_desc, mod.duration, tmpdir)
        elif mod.segment_type == MediaSegmentType.REF_FRAME:
            modification_index[mod] = generate_audio_glitch(audio_desc, output_desc.source.source_file,
                                                            output_desc.audio_ref_frame, mod.duration, tmpdir)

    audio_fragments: list[tuple[Path, timedelta]] = []

    source_offset: timedelta = timedelta()
    for mod, path in modification_index.items():
        if mod.insertion_point <= source_offset:
            audio_fragments.append((path, mod.duration))
            source_offset += mod.duration if mod.action == MediaSegmentAction.REPLACE else timedelta()
        else:
            # add a segment from the source file before the modification
            segment_duration = mod.insertion_point - source_offset
            segment_path = copy_audio_segment(output_desc.source.source_file, 
                                              source_offset + output_desc.source.offset,
                                              segment_duration, tmpdir)
            audio_fragments.append((segment_path, segment_duration))
            source_offset += segment_duration

            # add the modification itself
            audio_fragments.append((path, mod.duration))
            source_offset += mod.duration if mod.action == MediaSegmentAction.REPLACE else timedelta()

    if source_offset < output_desc.source.duration:
        # add the remaining segment from the source file
        remaining_duration = output_desc.source.duration - source_offset
        segment_path = copy_audio_segment(output_desc.source.source_file,
                                          source_offset + output_desc.source.offset,
                                          remaining_duration, tmpdir)
        audio_fragments.append((segment_path, remaining_duration))

    # concatenate all audio fragments
    def merge_two_fragments(f_a: tuple[Path, timedelta], f_b: tuple[Path, timedelta]) -> tuple[Path, timedelta]:
        a_path, a_duration = f_a
        b_path, b_duration = f_b

        overlap = OverlapInterval(offset_a=a_duration, offset_b=timedelta(), duration=timedelta())
        return merge_audio(a_path, b_path, audio_desc, overlap, tmpdir), a_duration + b_duration

    merged_path, _ = reduce(merge_two_fragments, audio_fragments)

    final_path = create_tempfile(suffix='_final_audio_output.m4a', dir=tmpdir)
    merged_path.rename(final_path)
    return final_path


def generate_output_from_output_media_desc(conf: Conf, media_desc: MediaDesc, output_desc: OutputMediaDesc):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        video_path: Optional[Path] = None
        if output_desc.has_video:
            printerr('## Generate Video...')
            video_path = generate_video(media_desc.video, output_desc, tmpdir)
            printerr('')

        audio_path: Optional[Path] = None
        if output_desc.has_audio:
            printerr('## Generate audio...')
            audio_path = generate_audio(media_desc.audio, output_desc, tmpdir)
            printerr('')

        printerr('## Mix audio and video...')
        if video_path and audio_path:
            join_audio_and_video_outputs(video_path, audio_path, output_desc.output)
        else:
            (video_path or audio_path).rename(output_desc.output)
        printerr('')


# this is to avoid issues with negative timedelta string representation
def pretty_timedelta(td: timedelta) -> str:
    if td.days >= 0:
        return str(td)
    return f'-({-td!s})'



def stringify_leaves(obj):
    if isinstance(obj, dict):
        return {k: stringify_leaves(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_leaves(item) for item in obj]
    elif isinstance(obj, timedelta):
        return pretty_timedelta(obj)
    else:
        return str(obj)


def format_dataclass(obj):
    return json.dumps(stringify_leaves(asdict(obj)), indent=4)


def generate_output_from_desc(conf: Conf, media_desc: MediaDesc, output_desc: OutputDesc):
    # Generate the reference output
    printerr(f'# Generate reference output in {output_desc.output_ref.output}...')
    generate_output_from_output_media_desc(conf, media_desc, output_desc.output_ref)
    printerr('')

    # Generate output A
    printerr(f'# Generate output A in {output_desc.output_a.output}...')
    generate_output_from_output_media_desc(conf, media_desc, output_desc.output_a)
    printerr('')

    # Generate output B
    printerr(f'# Generate output B in {output_desc.output_b.output}...')
    generate_output_from_output_media_desc(conf, media_desc, output_desc.output_b)
    printerr('')

    # Save metadata to JSON file
    printerr(f'# Saving generation parameters in {output_desc.output_meta_path}...')
    metadata = {
            'conf': asdict(conf),
            'source': output_desc.source.source_file,
            'overlap': {
                'video': asdict(output_desc.overlap.video),
                'audio': asdict(output_desc.overlap.audio),
                },
            'output_ref': asdict(output_desc.output_ref),
            'output_a': asdict(output_desc.output_a),
            'output_b': asdict(output_desc.output_b),
            'effects': [effect.name for effect in conf.effects],
            'av_sync_difference': output_desc.output_b.audio_video_desync,
            }
    output_desc.output_meta_path.write_text(json.dumps(stringify_leaves(metadata), indent=4))
    printerr('')


def get_source_segment(conf: Conf, media_desc: MediaDesc) -> MediaSource:
    fps = media_desc.video.fps if media_desc.video else Fraction(media_desc.audio.sample_rate, 1)
    source_duration = floor_sample_ts(timedelta(
        seconds=random.uniform(conf.min_length.total_seconds(), conf.max_length.total_seconds())), fps)
    max_source_offset = media_desc.duration - source_duration
    source_offset = floor_sample_ts(timedelta(seconds=random.uniform(0, max_source_offset.total_seconds())), fps)

    return MediaSource(
        source_file=conf.archive,
        offset=source_offset,
        duration=source_duration
    )


def get_overlap_from_conf(conf: Conf, media_source: MediaSource, media_desc: MediaDesc) -> MediaOverlap:
    fps = media_desc.video.fps if media_desc.video else Fraction(media_desc.audio.sample_rate, 1)
    source_offset = media_source.offset
    source_duration = media_source.duration

    overlap_duration = floor_sample_ts(timedelta(
        seconds=random.uniform(
            min(source_duration.total_seconds(), conf.min_overlap.total_seconds()),
            min(source_duration.total_seconds(), conf.max_overlap.total_seconds()))
        ), fps)
    overlap_offset = floor_sample_ts(
            timedelta(seconds=random.uniform(
                source_offset.total_seconds(),
                (source_offset + source_duration - overlap_duration).total_seconds())) - source_offset,
           fps)

    video_overlap = OverlapInterval(
        offset_a=overlap_offset,
        offset_b=timedelta(),
        duration=overlap_duration,
        )

    audio_overlap_offset = max(timedelta(), video_overlap.offset_a + conf.av_sync_difference)
    audio_overlap_end_offset = min(source_duration, audio_overlap_offset + overlap_duration)
    audio_overlap = OverlapInterval(
        offset_a=video_overlap.offset_a,
        offset_b=conf.av_sync_difference,
        duration=audio_overlap_end_offset - audio_overlap_offset,
        )

    # when there is no overlap we just select the cut point with no duration
    if conf.sample_type == SampleType.NO_OVERLAP:
        video_overlap = replace(video_overlap, duration=timedelta())
        audio_overlap = replace(audio_overlap, duration=timedelta())

    return MediaOverlap(
        video=video_overlap,
        audio=audio_overlap,
        )


class OutputType(StrEnum):
    REF = auto()
    A = auto()
    B = auto()



def get_glitch_modifications(output_desc: OutputMediaDesc, overlap: MediaOverlap, output_type: OutputType,
                             media_type: MediaType, media_desc: MediaDesc) -> list[MediaModification]:
    fps: Fraction = media_desc.video.fps if media_type == MediaType.VIDEO else Fraction(media_desc.audio.sample_rate, 1)
    time_per_sample: timedelta = timedelta(seconds=1 / fps.to_float()) 

    modifications: list[MediaModification] = []
    glitch_count = random.randint(1, MAX_GLITCH_COUNT)
    overlap_duration = overlap.video.duration if media_type == MediaType.AUDIO else overlap.audio.duration

    ini_offset = (
            output_desc.source.duration - overlap_duration
            if output_type == OutputType.A
            else timedelta())
    max_offset = output_desc.source.duration if output_type == OutputType.A else overlap_duration
    max_offset -= time_per_sample
    glitch_offsets: list[timedelta] = sorted([
        floor_sample_ts(
            timedelta(seconds=random.uniform(ini_offset.total_seconds(), max_offset.total_seconds())), fps)
        for _ in range(glitch_count)])

    next_glitch_offset = max_offset
    for glitch_offset in reversed(glitch_offsets):
        glitch_end_offset = min(
                next_glitch_offset - time_per_sample,
                glitch_offset + timedelta(seconds=random.uniform(
                    time_per_sample.total_seconds(),
                    MAX_GLITCH_DURATION.total_seconds())))

        duration = floor_sample_ts(glitch_end_offset - glitch_offset, fps)
        if duration >= timedelta():
            modifications.append(MediaModification(
                insertion_point=glitch_offset,
                media_type=media_type,
                segment_type=MediaSegmentType.REF_FRAME,
                action=MediaSegmentAction.REPLACE,
                duration=floor_sample_ts(glitch_end_offset - glitch_offset, fps),
            ))
            next_glitch_offset = glitch_offset

    return modifications


def get_modifications(conf: Conf, output_desc: OutputMediaDesc, overlap: MediaOverlap, output_type: OutputType,
                      media_desc: MediaDesc) -> list[MediaModification]:
    modifications: list[MediaModification] = []
    fps: Fraction = media_desc.video.fps if media_desc.video else Fraction(1, 1)
    time_per_frame: timedelta = timedelta(
            seconds=1 / media_desc.video.fps.to_float()) if media_desc.video else timedelta()

    if conf.sample_type == SampleType.VIDEO_WITH_SILENT_AUDIO:
        modifications.append(MediaModification(
            insertion_point=timedelta(),
            media_type=MediaType.AUDIO,
            segment_type=MediaSegmentType.SILENCE,
            action=MediaSegmentAction.REPLACE,
            duration=output_desc.source.duration,
        ))

    if conf.sample_type == SampleType.AUDIO_WITH_STATIC_VIDEO:
        modifications.append(MediaModification(
            insertion_point=timedelta(),
            media_type=MediaType.VIDEO,
            segment_type=MediaSegmentType.REF_FRAME,
            action=MediaSegmentAction.REPLACE,
            duration=output_desc.source.duration,
        ))

    if Effect.BLACK_FRAME_AT_END_OF_A in conf.effects and output_type == OutputType.A:
        offset = overlap.video.offset_a + overlap.video.duration
        duration = floor_sample_ts(timedelta(seconds=random.uniform(
            time_per_frame.total_seconds(),
            MAX_BLACK_FRAME_DURATION.total_seconds())), fps)

        video_modification = MediaModification(
                insertion_point=offset,
                media_type=MediaType.VIDEO,
                segment_type=MediaSegmentType.SILENCE,
                action=MediaSegmentAction.ADD,
                duration=duration,
                )
        if output_desc.has_video:
            modifications.append(video_modification)
        if output_desc.has_audio:
            modifications.append(replace(video_modification, media_type=MediaType.AUDIO))

    if Effect.BLACK_FRAME_AT_INI_OF_B in conf.effects and output_type == OutputType.B:
        offset = overlap.video.offset_a + overlap.video.duration
        duration = floor_sample_ts(timedelta(seconds=random.uniform(
            time_per_frame.total_seconds(),
            MAX_BLACK_FRAME_DURATION.total_seconds())), fps)

        video_modification = MediaModification(
                insertion_point=timedelta(),
                media_type=MediaType.VIDEO,
                segment_type=MediaSegmentType.SILENCE,
                action=MediaSegmentAction.ADD,
                duration=duration,
                )
        if output_desc.has_video:
            modifications.append(video_modification)
        if output_desc.has_audio:
            modifications.append(replace(video_modification, media_type=MediaType.AUDIO))

    if Effect.GLITCH_AT_END_OF_A in conf.effects and output_type == OutputType.A:
        if output_desc.has_video:
            modifications += get_glitch_modifications(output_desc, overlap, output_type, MediaType.VIDEO, media_desc)
        if output_desc.has_audio:
            modifications += get_glitch_modifications(output_desc, overlap, output_type, MediaType.AUDIO, media_desc)

    if Effect.GLITCH_AT_INI_OF_B in conf.effects and output_type == OutputType.B:
        if output_desc.has_video:
            modifications += get_glitch_modifications(output_desc, overlap, output_type, MediaType.VIDEO, media_desc)
        if output_desc.has_audio:
            modifications += get_glitch_modifications(output_desc, overlap, output_type, MediaType.AUDIO, media_desc)

    return modifications


def get_output_a_media_desc(conf: Conf, ref_desc: OutputMediaDesc, overlap: MediaOverlap, media_desc: MediaDesc
                            ) -> OutputMediaDesc:
    ref_source = ref_desc.source
    result = replace(ref_desc,
                     source=replace(ref_source, duration=overlap.video.offset_a + overlap.video.duration),
                     output=conf.output_prefix.with_suffix('.a.mp4'),
                     )
    result.media_modifications = get_modifications(conf, result, overlap, OutputType.A, media_desc)

    return result


def get_output_b_media_desc(conf: Conf, ref_desc: OutputMediaDesc, overlap: MediaOverlap, media_desc
                            ) -> OutputMediaDesc:
    ref_source = ref_desc.source
    offset = overlap.video.offset_a + ref_desc.source.offset 
    result = replace(ref_desc,
                     source=replace(ref_desc.source,
                                    offset=offset,
                                    duration=ref_source.duration + ref_source.offset - offset),
                     output=conf.output_prefix.with_suffix('.b.mp4'),
                     audio_video_desync=overlap.audio.offset_b - overlap.video.offset_b,
                     )
    result.media_modifications = get_modifications(conf, result, overlap, OutputType.B, media_desc)

    return result


def output_desc_from_conf(conf: Conf, media_desc: MediaDesc) -> OutputDesc:
    # get a random segment from the source archive
    total_media_source: MediaSource = get_source_segment(conf, media_desc)

    # choose random overlap intervals for audio and video
    overlap: MediaOverlap = get_overlap_from_conf(conf, total_media_source, media_desc)

    # Create output media descriptions based on the sample type and config
    source_duration = total_media_source.duration
    output_general = OutputMediaDesc(
        source=total_media_source,
        output=conf.output_prefix.with_suffix('.ref.mp4'),
        has_audio=conf.sample_type != SampleType.VIDEO_ONLY,
        has_video=conf.sample_type != SampleType.AUDIO_ONLY,
        audio_ref_frame=timedelta(seconds=random.uniform(0, source_duration.total_seconds())),
        video_ref_frame=timedelta(seconds=random.uniform(0, source_duration.total_seconds())),
    )

    output_ref = replace(output_general,
                         # all silence or all static video modifications in the video reference
                         media_modifications=get_modifications(
                             conf, output_general, MediaOverlap(), OutputType.REF, media_desc))
    output_a = get_output_a_media_desc(conf, output_general, overlap, media_desc)
    output_b = get_output_b_media_desc(conf, output_general, overlap, media_desc)


    return OutputDesc(
        source=total_media_source,
        overlap=overlap,
        output_ref=output_ref,
        output_a=output_a,
        output_b=output_b,
        output_meta_path=conf.output_prefix.with_suffix('.json'),
        )


def parse_effect(s: str) -> Effect:
    try:
        return Effect[s.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid effect: {s}. Allowed: {[f.name.lower() for f in Effect]}")


def get_conf() -> Conf:
    timedelta_parser = lambda x: timedelta(seconds=float(x))

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--archive', dest='archive', type=Path, required=True)
    parser.add_argument('-l', '--min-length', dest='min_length',
                        help='minimum length of the generated sample in seconds, allows decimals',
                        type=timedelta_parser, required=True)
    parser.add_argument('-L', '--max-length', dest='max_length',
                        help='maximum length of the generated sample in seconds, allows decimals',
                        type=timedelta_parser, required=True)
    parser.add_argument('-o', '--min-overlap', dest='min_overlap',
                        help='minimum overlap length in seconds, allows decimals',
                        type=timedelta_parser, required=True)
    parser.add_argument('-O', '--max-overlap', dest='max_overlap',
                        help='maximum overlap length in seconds, allows decimals',
                        type=timedelta_parser, required=True)
    parser.add_argument('-d', '--output-prefix', dest='output_prefix', type=Path, required=True)
    parser.add_argument('-t', '--sample-type', dest='sample_type', type=SampleType, choices=list(SampleType),
                        help='Type of sample to generate (default: full)', default=SampleType.FULL)
    parser.add_argument('-e', '--effect', dest='effects', type=parse_effect, action='append', default=[],
                        help='Effects to apply (can be specified multiple times, '
                        f'allowed: {[f.name.lower() for f in Effect]})')
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--av-sync-difference', dest='av_sync_difference',
                        help='Difference in audio/video sync between the two outputs in seconds, allows decimals',
                        type=timedelta_parser, default=timedelta(seconds=0))

    args = parser.parse_args()
    args.effects = set(args.effects)
    args.random_seed = args.random_seed if args.random_seed is not None else random.randint(0, 2**31 - 1)

    return Conf(**vars(args))


def validate_conf(conf: Conf):
    if not conf.archive.exists():
        raise FileNotFoundError(f"Archive file {conf.archive} does not exist")
    if conf.min_length <= timedelta(0):
        raise ValueError("Minimum length must be greater than zero")
    if conf.max_length <= timedelta(0):
        raise ValueError("Maximum length must be greater than zero")
    if conf.min_length > conf.max_length:
        raise ValueError("Minimum length cannot be greater than maximum length")
    if not conf.output_prefix.parent.exists():
        raise FileNotFoundError(f"Output directory {conf.output_prefix.parent} does not exist")


def validate_media(conf: Conf) -> MediaDesc:
    media_desc = get_media_desc(conf.archive)

    if media_desc.audio is None and media_desc.video is None:
        raise ValueError(f'Archive {conf.archive} has no audio or video tracks')

    if media_desc.video and media_desc.video.codec != 'h264':
        raise ValueError('This script only supports H264 video')
    if media_desc.audio and media_desc.audio.codec != 'aac':
        raise ValueError('This script only supports AAC audio')

    match conf.sample_type:
        case SampleType.FULL:
            if not media_desc.audio or not media_desc.video:
                raise ValueError('Full sample type requires both audio and video tracks')
        case SampleType.AUDIO_ONLY:
            if not media_desc.audio:
                raise ValueError('Audio only sample type requires an audio track')
        case SampleType.VIDEO_ONLY:
            if not media_desc.video:
                raise ValueError('Video only sample type requires a video track')
        case SampleType.VIDEO_WITH_SILENT_AUDIO:
            if not media_desc.video or not media_desc.audio:
                raise ValueError('Video with silent audio sample type requires an audio and video track')
        case SampleType.AUDIO_WITH_STATIC_VIDEO:
            if not media_desc.audio or not media_desc.video:
                raise ValueError('Audio with static video sample type requires an audio and video track')
        case SampleType.NO_OVERLAP:
            pass  # No specific requirements for this type

    return media_desc


def main(conf: Conf):
    printerr(f'Using conf: {format_dataclass(conf)}')

    # set random seed for reproducibility
    random.seed(conf.random_seed)

    validate_conf(conf)

    validate_tools()

    media_desc: MediaDesc = validate_media(conf)
    printerr(f'Media description: {format_dataclass(media_desc)}')

    output_desc: OutputDesc = output_desc_from_conf(conf, media_desc)
    printerr(f'Output description: {format_dataclass(output_desc)}')

    generate_output_from_desc(conf, media_desc, output_desc)


if __name__ == '__main__':  # pragma: no cover
    main(get_conf())
