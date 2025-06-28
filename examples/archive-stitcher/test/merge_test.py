# copyright 2025 Vonage

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from src.data_model import MediaDesc, MediaOverlap, MergeArgs, OverlapInterval
from src.merge import get_next_best_keyframe_ts_in_interval, merge_files
from src.validations import get_media_desc
from src.utils import StitcherException

from .test_base import TestBase


class MergeTest(TestBase):
    def setUp(self):
        super().setUp()

        self.archive_a_desc: MediaDesc = get_media_desc(self.conf.archive_a)
        self.archive_b_desc: MediaDesc = get_media_desc(self.conf.archive_b)

        overlap = OverlapInterval(
                offset_a=timedelta(seconds=57, milliseconds=45500),
                offset_b=timedelta(milliseconds=40000),
                duration=timedelta(seconds=2, milliseconds=40000),
                )
        self.merge_conf = MergeArgs(
                duration_a=self.archive_a_desc.duration,
                duration_b=self.archive_b_desc.duration,
                video_desc=self.archive_a_desc.video,
                audio_desc=self.archive_a_desc.audio,
                overlap=MediaOverlap(video=overlap, audio=overlap),
                )
        self.output = self.dir_path / 'output.mp4'

        self.expected_duration = timedelta(minutes=1, seconds=21)

    def validate_output(self, expected_desc: MediaDesc, output: Path, expected_delta: float):
        self.assertTrue(output.exists)

        # validate merged output
        desc: MediaDesc = get_media_desc(output)

        # validate the output media format without checking the duration
        self.assertEqual(expected_desc, replace(desc, duration=expected_desc.duration))

        # check duration is what we expect with a certain threshold
        self.assertAlmostEqual(expected_desc.duration.total_seconds(),
            desc.duration.total_seconds(), delta=expected_delta)

    def test_merge_success_given_overlap(self):
        expected_desc = MediaDesc(
                video=self.merge_conf.video_desc,
                audio=self.merge_conf.audio_desc,
                duration=self.expected_duration)
        merge_files(self.conf.archive_a, self.conf.archive_b, self.output, self.merge_conf)

        self.validate_output(expected_desc, self.output, 0.25)

    def test_merge_success_given_audio_only(self):
        self.merge_conf.video_desc = None
        expected_desc = MediaDesc(
                video=None,
                audio=self.merge_conf.audio_desc,
                duration=self.expected_duration)

        merge_files(self.conf.archive_a, self.conf.archive_b, self.output, self.merge_conf)

        self.validate_output(expected_desc, self.output, 0.25)

    def test_merge_success_given_video_only(self):
        self.merge_conf.audio_desc = None
        expected_desc = MediaDesc(
                video=self.merge_conf.video_desc,
                audio=None,
                duration=timedelta(minutes=1, seconds=18))

        merge_files(self.conf.archive_a, self.conf.archive_b, self.output, self.merge_conf)

        self.validate_output(expected_desc, self.output, 0.5)

    def test_merge_success_given_no_overlap(self):
        expected_desc = MediaDesc(
                video=self.merge_conf.video_desc,
                audio=self.merge_conf.audio_desc,
                duration=self.archive_a_desc.duration + self.archive_b_desc.duration)

        merge_files(self.conf.archive_a, self.conf.archive_b, self.output,
                    replace(self.merge_conf, overlap=MediaOverlap()))

        self.validate_output(expected_desc, self.output, 0.25)

    def test_merge_error_given_no_audio_or_video(self):
        with self.assertRaises(Exception):
            merge_files(self.conf.archive_a, self.conf.archive_b, self.output, replace(self.merge_conf, audio=None,
                                                                                       video=None))

    def test_get_next_bext_keyframe_error_given_wrong_ffprobe_output(self):
        with patch('src.merge.run_exec') as mock, self.assertRaisesRegex(StitcherException, '.*Wrong ffprobe output.*'):
            mock.return_value = 'csv missing, a field'
            get_next_best_keyframe_ts_in_interval(self.conf.archive_b, timedelta(), timedelta(seconds=10))

    def test_get_next_best_keyframe_error_given_no_key_frame(self):
        with patch('src.merge.run_exec') as mock, self.assertRaisesRegex(StitcherException, '.*No keyframe.*'):
            mock.return_value = 'something,0,P'
            get_next_best_keyframe_ts_in_interval(self.conf.archive_b, timedelta(), timedelta(seconds=10))
