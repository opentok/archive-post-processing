# copyright 2025 Vonage

import argparse

from datetime import timedelta
from pathlib import Path

from .data_model import AlgoAudio, AlgoVideo, Conf, FindOverlapArgs, MediaDesc, MediaOverlap, MergeArgs
from .merge import merge_files
from .overlap import find_overlap
from .utils import printerr
from .validations import get_media_desc, validate_conf, validate_media, validate_tools


def get_conf() -> Conf:
    """
    Parse command-line arguments and return a Conf object.

    Returns:
        Conf: Configuration object with user-specified parameters.
    """
    timedelta_parser = lambda x: timedelta(seconds=float(x))

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--archive-a', dest='archive_a', type=Path, required=True)
    parser.add_argument('-b', '--archive-b', dest='archive_b', type=Path, required=True)
    parser.add_argument('-o', '--output', dest='output', type=Path, required=True)
    parser.add_argument('-k', '--algo-video', dest='algo_video', type=AlgoVideo, choices=list(AlgoVideo),
                        help='Algorithm for self-similarity (default: mse)', default='mse')
    parser.add_argument('-g', '--algo-audio', dest='algo_audio', type=AlgoAudio, choices=list(AlgoAudio),
                        help='Algorithm for self-similarity (default: pearson)', default='pearson')
    parser.add_argument('-x', '--max-overlap', dest='max_overlap',
                        metavar='MAX_OVERLAP_BETWEEN_FILES_SEC',
                        help='maximum overlap in seconds between the input files in seconds, allows decimals',
                        type=timedelta_parser, required=True)
    parser.add_argument('-y', '--allow-output-overwrite', dest='allow_output_overwrite', action='store_true',
                        help='whether to allow overwriting the output file (default: false)', default=False)
    parser.add_argument('-d', '--debug-plot', dest='debug_plot', action='store_true',
                        help='whether output debugging plots (default: false)', default=False)
    parser.add_argument('-dd', '--deep-debug-plot', dest='deep_debug_plot', action='store_true',
                        help='whether output deep debugging plots (default: false)', default=False)
    parser.add_argument('-s', '--deep-search', dest='deep_search', action='store_true',
                        help='whether to conduct an extensive overlapping search (default: false)', default=False)

    return Conf(**vars(parser.parse_args()))


def main(conf: Conf):
    """
    Main entry point for archive stitcher process.

    Args:
        conf (Conf): Configuration object with parameters and options.

    Returns:
        None; performs archive stitching and merging according to configuration.
    """
    printerr(f'Using conf: {conf}')

    validate_tools()
    validate_conf(conf)

    media_desc_a: MediaDesc = get_media_desc(conf.archive_a)
    media_desc_b: MediaDesc = get_media_desc(conf.archive_b)

    validate_media(media_desc_a, media_desc_b)

    overlap: MediaOverlap = find_overlap(conf.archive_a, conf.archive_b, FindOverlapArgs(
        duration_a=media_desc_a.duration,
        duration_b=media_desc_b.duration,
        video_desc=media_desc_a.video,
        audio_desc=media_desc_a.audio,
        max_overlap=conf.max_overlap,
        algo_video=conf.algo_video,
        algo_audio=conf.algo_audio,
        debug_plot=conf.debug_plot,
        deep_debug_plot=conf.deep_debug_plot,
        deep_search=conf.deep_search,
        ))
  
    if overlap == MediaOverlap():
        printerr('No overlap found, just appending files one to another')
    else:
        printerr(f'Found overlap: {overlap}')

    merge_files(conf.archive_a, conf.archive_b, conf.output, MergeArgs(
        duration_a=media_desc_a.duration,
        duration_b=media_desc_b.duration,
        video_desc=media_desc_a.video,
        audio_desc=media_desc_a.audio,
        overlap=overlap,
        ))


if __name__ == '__main__':  # pragma: no cover
    main(get_conf())
