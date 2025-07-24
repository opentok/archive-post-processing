# copyright 2025 Vonage

import os
import shutil
import tempfile
import unittest

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from src.data_model import AlgoAudio, AlgoVideo, Conf
from src.utils import FFMPEG, FFPROBE


TEST_DATA_DIR: Path = Path(__file__).resolve().parent / '..' / 'test_data'


class TestBase(unittest.TestCase):
    def setUp(self):
        # temp directory for each test case
        self.dir = tempfile.TemporaryDirectory()
        self.dir_path: Path = Path(self.dir.name)

        # default config
        self.conf = Conf(
                archive_a=self.dir_path / 'archive_a.mp4',
                archive_b=self.dir_path / 'archive_b.mp4',
                output=self.dir_path / 'output.mp4',
                max_overlap=timedelta(seconds=4),
                algo_video=AlgoVideo.MSE,
                algo_audio=AlgoAudio.PEARSON,
                allow_output_overwrite=True,
                debug_plot=False,
                deep_search=False,
                )

        # setting up some example videos for the tests to work with in the test case temp dir
        self.conf.archive_a.symlink_to(TEST_DATA_DIR / 'screenshare_low_variation' / 'archive_a.mp4')
        self.conf.archive_b.symlink_to(TEST_DATA_DIR / 'screenshare_low_variation' / 'archive_b.mp4')
        (self.dir_path / 'output_ref.mp4').symlink_to(TEST_DATA_DIR / 'screenshare_low_variation' / 'output_ref.mp4')

        # linking ffmpeg and ffprobe into our temp dir and using this as our PATH
        (self.dir_path / FFMPEG).symlink_to(shutil.which(FFMPEG))
        (self.dir_path / FFPROBE).symlink_to(shutil.which(FFPROBE))

        self.env_patcher = patch.dict(os.environ, {'PATH': self.dir.name})
        self.env_patcher.start()

    def tearDown(self):
        if output_path:=os.environ.get('TEST_OUTPUT', ''):
            shutil.copytree(self.dir_path, Path(output_path) / self.dir_path.name)
        self.dir.cleanup()
        self.env_patcher.stop()
