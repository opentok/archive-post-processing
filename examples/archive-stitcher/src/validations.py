# copyright 2025 Vonage

import json
import shutil

from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .data_model import AudioDesc, Conf, Fraction, MediaDesc, VideoDesc
from .utils import FFMPEG, FFPROBE, raise_error, run_exec


FFPROBE_TO_VIDEO_DESC_MAPPING: dict[str, tuple[str, Callable[str, Any]]] = {
        'codec_name': ['codec', str],
        'profile': ['profile', str],
        'width': ['width', int],
        'height': ['height', int],
        'pix_fmt': ['pix_fmt', str],
        'level': ['level', int],
        'r_frame_rate': ['fps', lambda x: Fraction.fromstr(x)],
        'time_base': ['timescale', lambda x: Fraction.fromstr(x)],
        }


FFPROBE_TO_AUDIO_DESC_MAPPING: dict[str, tuple[str, Callable[str, Any]]] = {
        'codec_name': ['codec', str],
        'sample_rate': ['sample_rate', int],
        'channels': ['channels', int],
        }


def validate_tools():
    # check we have the needed executables
    for executable in [FFMPEG, FFPROBE]:
        if not shutil.which(executable):
            raise_error(f'{executable} executable could not be found')

    # check we have the needed codecs
    codec_infos = run_exec(FFMPEG, '-codecs')
    has_x264: bool = False
    has_aac: bool = False
    for codec_info in map(str.strip, codec_infos.splitlines()):
        # check both encoding and decoding support
         has_x264 = has_x264 or codec_info.startswith('DE') and ' libx264' in codec_info
         has_aac = has_aac or codec_info.startswith('DE') and ' aac' in codec_info

         if has_aac and has_x264:
             break

    if not has_x264:
        raise_error('FFMPEG is missing libx264 encoder or decoder')
    if not has_aac:
        raise_error('FFMPEG is missing aac encoder or decoder')


def validate_conf(conf: Conf):
    # check input files
    if not conf.archive_a.exists() or not conf.archive_a.is_file():
        raise_error(f'archive-a {conf.archive_a} is not a valid file')
    if not conf.archive_b.exists() or not conf.archive_b.is_file():
        raise_error(f'archive-b {conf.archive_b} is not a valid file')

    # check the output file
    if conf.output.exists() and not conf.allow_output_overwrite:
        raise_error(f'Output {conf.output} already exists, to allow overwriting set the -y flag')
    if conf.output.exists() and not conf.output.is_file():
        raise_error(f'Output {conf.output} already exists as a directory')
    if not conf.output.parent.is_dir():
        raise_error(f'Output directory {conf.output.parent} is not valid')
    if conf.output.suffix != '.mp4':
        raise_error(f'Output file {conf.output} is not an mp4 file, please use a .mp4 extension')


def validate_media(media_desc_a: MediaDesc, media_desc_b: MediaDesc):
    # check there's at least an audio or video track
    if media_desc_a.audio is None and media_desc_a.video is None:
        raise_error(f'archives have no audio or video tracks')

    # check the media descriptions are compatible
    if media_desc_a.audio != media_desc_b.audio:
        raise_error(f'Archives have incompatible audio formats, A: {media_desc_a.audio}, B: {media_desc_b.audio}')
    if media_desc_a.video != media_desc_b.video:
        raise_error(f'Archives have incompatible video formats, A: {media_desc_a.video}, B: {media_desc_b.video}')

    # check the codecs are the expected ones
    if media_desc_a.video and media_desc_a.video.codec != 'h264':
        raise_error('This script only supports H264 video')
    if media_desc_a.audio and media_desc_a.audio.codec != 'aac':
        raise_error('This script only supports AAC audio')


def get_media_desc(filepath: Path) -> MediaDesc:
    json_desc_str: str = run_exec(
        FFPROBE, '-print_format', 'json', '-show_streams', '-show_format', filepath)

    video_desc: Optional[VideoDesc] = None
    audio_desc: Optional[AudioDesc] = None

    try:
        json_desc: Optional[dict] = res if (res:=json.loads(json_desc_str)) and isinstance(res, dict) else None
        for stream in json_desc['streams']:
            if stream['codec_type'] == 'audio':
                audio_desc = AudioDesc(**{
                    name: converter(stream[ffprobe_name])
                    for ffprobe_name, (name, converter) in FFPROBE_TO_AUDIO_DESC_MAPPING.items()
                    })
            elif stream['codec_type'] == 'video':
                video_desc = VideoDesc(**{
                    name: converter(stream[ffprobe_name])
                    for ffprobe_name, (name, converter) in FFPROBE_TO_VIDEO_DESC_MAPPING.items()
                    })

        duration = timedelta(seconds=float(json_desc['format']['duration']))

        return MediaDesc(video_desc, audio_desc, duration)
    except Exception as exc:
        raise_error(f'wrong ffprobe output for file {filepath}: {json_desc_str}, exception {exc}')
