# copyright 2025 Vonage

import sys

from datetime import timedelta
from pathlib import Path

from src import archive_stitcher
from src.data_model import AlgoAudio, AlgoVideo, Conf

from .test_base import TestBase


class ArchiveStitcherTest(TestBase):
    def test_get_conf(self):
        sys.argv = ['archive-stitcher.py',
                    '-a', 'archive-a',
                    '-b', 'archive-b',
                    '-o', 'output',
                    '-k', 'mse',
                    '-x', '1.5',
                    '-y']

        expected_conf: Conf = Conf(
                archive_a=Path('archive-a'),
                archive_b=Path('archive-b'),
                output=Path('output'),
                max_overlap=timedelta(seconds=1, milliseconds=500),
                algo_video=AlgoVideo.MSE,
                algo_audio=AlgoAudio.PEARSON,
                allow_output_overwrite=True,
                debug_plot=False,
                deep_search=False,
                )
        self.assertEqual(expected_conf, archive_stitcher.get_conf())

    def test_main(self):
        archive_stitcher.main(self.conf)

        self.assertTrue(self.conf.output.exists())

        # TODO: check output content
