# copyright 2025 Vonage

import re
import tempfile

from dataclasses import replace
from datetime import timedelta
from enum import auto, StrEnum
from pathlib import Path
from typing import Optional

from .data_model import AudioDesc, MergeArgs, OverlapInterval, VideoDesc
from .utils import create_tempfile, FFMPEG, FFPROBE, printerr, raise_error, run_exec


MAX_KEYFRAME_INTERVAL = timedelta(seconds=20)


def merge_files(archive_a: Path, archive_b: Path, output: Path, conf: MergeArgs) -> None:
    """
    Merge two media files (audio and/or video) with overlap consideration and write merged
    output to the specified path.

    Args:
        archive_a (Path): Path to first archive file.
        archive_b (Path): Path to second archive file.
        output (Path): Path where output should be saved.
        conf (MergeArgs): Merge configuration and overlap details.
    """
    # precondition, we need either video or audio, this has to have been validated already in an outer layer
    assert(conf.video_desc or conf.audio_desc)

    # if there's no overlap interval adapt it to just concatenate the files
    duration_no_overlap = timedelta(seconds=conf.video_desc.fps.to_float()) if conf.video_desc else timedelta()
    no_overlap_interval = OverlapInterval(offset_a=conf.duration_a, offset_b=timedelta(), duration=duration_no_overlap)
    overlap = replace(conf.overlap,
                      audio=conf.overlap.audio if conf.overlap.audio != OverlapInterval() else no_overlap_interval,
                      video=conf.overlap.video if conf.overlap.video != OverlapInterval() else no_overlap_interval,
                      )

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        video_path: Optional[Path] = None
        if conf.video_desc:
            video_path = merge_video(archive_a, archive_b, overlap.video, conf.video_desc, tmpdir)

        audio_path: Optional[Path] = None
        if conf.audio_desc:
            audio_path = merge_audio(archive_a, archive_b, conf.audio_desc, overlap.audio, tmpdir)

        if video_path and audio_path:
            join_audio_and_video_outputs(video_path, audio_path, output)
        else:
            (video_path or audio_path).rename(output)


class MediaType(StrEnum):
    """Enumeration for media types."""
    VIDEO = auto()
    AUDIO = auto()


def concatenate_media_files(files: list[Path], output: Path, media_type: MediaType, tmpdir: Path) -> None:
    """
    Concatenate media files into a single output file.

    Args:
        files (list[Path]): List of file paths to concatenate.
        output (Path): Path to output file.
        media_type (MediaType): Type of media (VIDEO or AUDIO).
        tmpdir (Path): Temporary directory for intermediate files.
    """
    parts_filename: Path = tmpdir / 'parts_list.txt'
    parts_filename.write_text('\n'.join(map(lambda f: f"file '{f}'", files)))

    media_args: list[str] = ['-c:v', 'copy', '-an'] if media_type == MediaType.VIDEO else ['-c:a', 'copy', '-vn']

    run_exec(FFMPEG, '-f', 'concat', '-safe', '0', '-i', parts_filename, *media_args, '-y', output)


def merge_video(archive_a: Path, archive_b: Path, overlap: OverlapInterval, video_desc: VideoDesc,
                tmpdir: Path) -> Path:
    """
    Merge two video files based on overlap interval.

    Args:
        archive_a (Path): First video file.
        archive_b (Path): Second video file.
        overlap (OverlapInterval): Overlap details.
        video_desc (VideoDesc): Video descriptor (profile, level, fps, etc).
        tmpdir (Path): Temporary directory.

    Returns:
        Path: Path to merged video file.
    """
    output_ini: Path = create_tempfile(suffix='_vini.mp4', dir=tmpdir)
    output_mid: Path = create_tempfile(suffix='_vmid.mp4', dir=tmpdir)
    output_end: Path = create_tempfile(suffix='_vend.mp4', dir=tmpdir)

    output: Path = create_tempfile(suffix='_vout.mp4', dir=tmpdir)

    next_keyframe_pts_in_b: timedelta = get_next_best_keyframe_ts_in_interval(
            archive_b, overlap.offset_b, overlap.duration)

    # if the video overlap contains a keyframe in B, use that as the cut point
    overlap_with_keyframe: bool = next_keyframe_pts_in_b <= overlap.offset_b + overlap.duration
    cutpoint_b: timedelta = next_keyframe_pts_in_b if overlap_with_keyframe else overlap.offset_b

    # if we've shifted the cutpoint in B due to finding a keyframe, shift the same amount in A
    cutpoint_a: timedelta = overlap.offset_a + cutpoint_b - overlap.offset_b

    # cut first part of A without re-encoding
    run_exec(FFMPEG, '-i', archive_a, '-to', cutpoint_a.total_seconds(), '-c:v', 'copy', '-an', '-y', output_ini)

    needs_mid_part: bool = not overlap_with_keyframe
    if needs_mid_part:
        #re-encode overlapping part until keyframe in B
        run_exec(
                FFMPEG,
                '-i', archive_b,
                '-ss', cutpoint_b.total_seconds(),
                '-to', (next_keyframe_pts_in_b - timedelta(milliseconds=1)).total_seconds(),
                '-c:v', 'libx264',
                '-profile:v', normalize_video_profile(video_desc.profile),
                '-level', normalize_video_level(video_desc.level),
                '-r', video_desc.fps,
                '-fps_mode', 'cfr',
                '-video_track_timescale', video_desc.timescale.den,
                '-x264-params', f'fps={video_desc.fps}:timebase={video_desc.timescale}',
                '-an',
                '-y',
                output_mid)

    # cut last part of B starting from the first keyframe after overlapping
    run_exec(FFMPEG, '-i', archive_b, '-ss', next_keyframe_pts_in_b.total_seconds(), '-c:v', 'copy', '-an', '-y',
             output_end)

    # concatenate everything
    parts: list[Path] = [output_ini, output_mid, output_end] if needs_mid_part else [output_ini, output_end]
    concatenate_media_files(parts, output, MediaType.VIDEO, tmpdir)

    return output


def merge_audio(archive_a: Path, archive_b: Path, audio_desc: AudioDesc, overlap: OverlapInterval,
                tmpdir: Path) -> Path:
    """
    Merge two audio files based on overlap interval.

    Args:
        archive_a (Path): First audio file.
        archive_b (Path): Second audio file.
        audio_desc (AudioDesc): Audio descriptor.
        overlap (OverlapInterval): Overlap details.
        tmpdir (Path): Temporary directory.

    Returns:
        Path: Path to merged audio file.
    """
    output_ini: Path = create_tempfile(suffix='_ini.m4a', dir=tmpdir)
    output_end: Path = create_tempfile(suffix='_end.m4a', dir=tmpdir)

    output: Path = tmpdir / 'audio_output.m4a'

    # aac may have an initial priming delay, if we then want to concatenate the two audio parts without re-encoding
    # with the concat filter we need to account for this, otherwise the final video will be out-of-sync
    aac_priming_delay: timedelta = get_audio_aac_priming_delay(archive_a, audio_desc, tmpdir)

    # cut first part of A without re-encoding
    run_exec(FFMPEG,
             '-i', archive_a, '-ss', aac_priming_delay, '-to', overlap.offset_a, '-c:a', 'copy', '-vn', '-y',
             output_ini)
    # cut last part of B without re-encoding
    run_exec(FFMPEG, '-i', archive_b, '-ss', overlap.offset_b, '-c:a', 'copy', '-vn', '-y', output_end)

    # concatenate them
    parts: list[Path] = [output_ini, output_end]
    concatenate_media_files(parts, output, MediaType.AUDIO, tmpdir)

    return output


def join_audio_and_video_outputs(video: Path,  audio: Path, output: Path) -> None:
    """
    Join audio and video outputs into a final merged media file.

    Args:
        video (Path): Path to video file.
        audio (Path): Path to audio file.
        output (Path): Output file path.
    """
    run_exec(FFMPEG,
             '-i', video, '-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'copy', '-y', output)


AAC_PRIMING_INFO_RE = re.compile(r'^\s*Stream .*Audio: aac \(LC\).*, delay (\d+),.*')


def get_audio_aac_priming_delay(filepath: Path, audio_desc: AudioDesc, tmpdir: Path) -> timedelta:
    """
    Get AAC priming delay from an audio file.

    Args:
        filepath (Path): Path to audio file.
        audio_desc (AudioDesc): Audio descriptor.
        tmpdir (Path): Temporary directory.

    Returns:
        timedelta: Priming delay as timedelta.
    """
    result = timedelta()

    output: Path = create_tempfile(suffix='_audiodec.m4a', dir=tmpdir)
    # the only way to get this information is to decode a bit of the video (1s) and dump trace logging
    dump_txt: str = run_exec(
            FFMPEG, '-i', filepath, '-dump', '-to', timedelta(seconds=1), '-loglevel', 'trace', '-y', output,
            get_stderr=True)
    for line in dump_txt.splitlines():
        if match:=re.search(AAC_PRIMING_INFO_RE, line):
            aac_priming_delay_samples = int(match.group(1))
            result = timedelta(seconds=aac_priming_delay_samples / audio_desc.sample_rate)
            printerr(f'In audio track found AAC LC priming delay of {result}')
            break

    return result


def get_next_best_keyframe_ts_in_interval(filepath: Path, interval_start: timedelta,
    duration: timedelta) -> timedelta:
    """
    Find the timestamp of the next best keyframe in a video file within a given interval.

    Args:
        filepath (Path): Path to video file.
        interval_start (timedelta): Start of interval.
        duration (timedelta): Duration of interval.

    Returns:
        timedelta: Time of next keyframe found.
    """
    frames_csv: str = run_exec(
            FFPROBE, '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=pict_type,pts_time',
            '-read_intervals', f'{interval_start}%+{MAX_KEYFRAME_INTERVAL}', '-of', 'csv', filepath)

    def parse_line(csv_line) -> tuple[timedelta, str]:
        try:
            [_, pts_str, frame_type, *_args] = csv_line.split(',')
            return timedelta(seconds=float(pts_str)), frame_type
        except Exception as exc:
            raise_error(f'Wrong ffprobe output in {filepath} while looking for next keyframe: {csv_line}, {exc}')

    next_keyframe_pts: Optional[timedelta] = None
    for csv_line in map(str.strip, frames_csv.splitlines()):
        if csv_line:
            frame_pts, frame_type = parse_line(csv_line)

        # find the first line with PTS bigger than interval_start that is a keyframe
        if frame_pts < interval_start:
            # ffprobe when using read_intervals option for the start of the interval it matches a keyframe,
            # so usually we get some entries before our specified start time
            continue
        if frame_type != 'I':
            continue

        # while we are in the given interval check for the latest keyframe inside it, if we are out of the interval
        # just return whatever keyframe we've found
        if next_keyframe_pts is not None and frame_pts > interval_start + duration:
            break

        next_keyframe_pts = frame_pts

    if next_keyframe_pts is None:
        raise_error(f'No keyframe found in {filepath} in'
                    f' interval {interval_start}-{interval_start + MAX_KEYFRAME_INTERVAL}')

    return next_keyframe_pts


def normalize_video_profile(ffprobe_profile: str) -> str:
    """
    Normalize video profile string to lower case and remove 'constrained ' prefix.

    Args:
        ffprobe_profile (str): Profile string.

    Returns:
        str: Normalized profile.
    """
    return ffprobe_profile.lower().removeprefix('constrained ')


def normalize_video_level(ffprobe_level: int) -> str:
    """
    Normalize video level to string.

    Args:
        ffprobe_level (int): Video level.

    Returns:
        str: Normalized video level.
    """
    return str(ffprobe_level) if ffprobe_level < 10 else f'{ffprobe_level // 10}.{ffprobe_level % 10}'
