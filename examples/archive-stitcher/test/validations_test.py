# copyright 2025 Vonage

from unittest.mock import patch

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from parameterized import parameterized

from src.data_model import AudioDesc, Fraction, MediaDesc, VideoDesc
from src.utils import FFMPEG, FFPROBE, StitcherException
from src.validations import get_media_desc, validate_conf, validate_media, validate_tools

from .test_base import TestBase


class ValidationsTest(TestBase):
    def test_validate_tools_success(self):
        validate_tools()

    @parameterized.expand([
        (FFPROBE, ),
        (FFMPEG, ),
        ])
    def test_validate_tools_error_given_no_binary(self, missing_binary: str):
        (self.dir_path / missing_binary).unlink()

        with self.assertRaisesRegex(StitcherException, f'{missing_binary} executable could not be found'):
            validate_tools()

    @parameterized.expand([
        ('libx264', ),
        ('aac', ),
        ])
    def test_validate_tools_error_given_no_codec(self, missing_codec: str):
        with patch('src.validations.run_exec') as mock:
            video_flags = 'D.V' if missing_codec == 'libx264' else 'DEV'
            audio_flags = 'D.A' if missing_codec == 'aac' else 'DEV'
            mock.return_value = '\n'.join([
                    f'{video_flags}.LS h264\tH.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (encoders: libx264)',
                    f'{audio_flags}.L. aac\tAAC (Advanced Audio Coding) (decoders: aac) (encoders: aac aac_at)',
                    ])

            with self.assertRaisesRegex(StitcherException, f'FFMPEG is missing {missing_codec} encoder or decoder'):
                validate_tools()

    def test_validate_conf_success(self):
        self.conf.allow_output_overwrite = False
        validate_conf(self.conf)

    @parameterized.expand([
        ('missing_a', True),
        ('missing_b', False),
        ])
    def test_validate_conf_error_given_missing_input_files(self, _name: str, a_is_missing: bool):
        fake_file = Path('/invented/path/a')
        conf = replace(self.conf, **{'archive_a' if a_is_missing else 'archive_b': fake_file})
        with self.assertRaisesRegex(StitcherException,
                                    f'archive-{"a" if a_is_missing else "b"} {fake_file} is not a valid file'):
            validate_conf(conf)

    def test_validate_conf_success_given_output_exists_and_flag(self):
        # put a file in the output path name
        self.conf.output.touch()
        self.conf.allow_output_overwrite = True

        validate_conf(self.conf)

    def test_validate_conf_error_given_output_exists_and_no_flag(self):
        # put a file in the output path name
        self.conf.output.touch()
        self.conf.allow_output_overwrite = False

        with self.assertRaisesRegex(StitcherException,
                                    f'Output {self.conf.output} already exists, to allow overwriting set the -y flag'):
            validate_conf(self.conf)

    def test_validate_conf_error_given_output_exists_as_a_directory(self):
        # put a file in the output path name
        self.conf.output.mkdir()
        self.conf.allow_output_overwrite = True

        with self.assertRaisesRegex(StitcherException, f'Output {self.conf.output} already exists as a directory'):
            validate_conf(self.conf)

    def test_validate_conf_error_given_output_not_mp4(self):
        self.conf.output = self.conf.output.with_suffix('.notmp4')

        with self.assertRaisesRegex(StitcherException,
                                    f'Output file {self.conf.output} is not an mp4 file, please use a .mp4 extension'):
            validate_conf(self.conf)

    def test_validate_conf_error_given_output_not_in_a_valid_directory(self):
        # put a file in the output path name
        self.conf.output = Path('/some/invented/path/output.mp4')
        self.conf.allow_output_overwrite = False

        with self.assertRaisesRegex(StitcherException, f'Output directory {self.conf.output.parent} is not valid'):
            validate_conf(self.conf)

    def test_get_media_desc_success(self):
        expected = MediaDesc(
                video=VideoDesc(
                    codec='h264',
                    profile='Constrained Baseline',
                    width=1280,
                    height=720,
                    pix_fmt='yuv420p',
                    level=41,
                    fps=Fraction(25, 1),
                    timescale=Fraction(1, 90000)
                    ),
                audio=AudioDesc(
                    codec='aac',
                    sample_rate=48000,
                    channels=1),
                duration=timedelta(seconds=59, microseconds=441000),
                )
        self.assertAlmostEqual(expected, get_media_desc(self.conf.archive_a), delta=0.5)

    def test_get_media_desc_error_given_wrong_ffprobe_output(self):
        with patch('src.validations.run_exec') as mock:
            mock.return_value = 'not a json{'

            with self.assertRaisesRegex(StitcherException, f'wrong ffprobe output for file {self.conf.archive_a}:.*'):
                get_media_desc(self.conf.archive_a)

    def test_validate_media_success(self):
        media_desc_a: MediaDesc = get_media_desc(self.conf.archive_a)
        media_desc_b: MediaDesc = get_media_desc(self.conf.archive_b)

        validate_media(media_desc_a, media_desc_b)

    def test_validate_media_success_given_audio_only(self):
        media_desc_a: MediaDesc = replace(get_media_desc(self.conf.archive_a), video=None)
        media_desc_b: MediaDesc = replace(get_media_desc(self.conf.archive_b), video=None)

        validate_media(media_desc_a, media_desc_b)

    def test_validate_media_success_given_video_only(self):
        media_desc_a: MediaDesc = replace(get_media_desc(self.conf.archive_a), audio=None)
        media_desc_b: MediaDesc = replace(get_media_desc(self.conf.archive_b), audio=None)

        validate_media(media_desc_a, media_desc_b)

    def test_validate_media_error_given_no_tracks(self):
        with self.assertRaisesRegex(StitcherException, 'archives have no audio or video tracks'):
            validate_media(MediaDesc(None, None, timedelta()), MediaDesc(None, None, timedelta()))

    def test_validate_media_error_given_incompatible_tracks(self):
        media_desc_a: MediaDesc = replace(get_media_desc(self.conf.archive_a), video=None)
        media_desc_b: MediaDesc = replace(get_media_desc(self.conf.archive_b), audio=None)

        with self.assertRaisesRegex(StitcherException, 'Archives have incompatible audio formats.*'):
            validate_media(media_desc_a, media_desc_b)

    def test_validate_media_error_given_incompatible_audio_properties(self):
        media_desc_a: MediaDesc = get_media_desc(self.conf.archive_a)
        media_desc_b: MediaDesc = get_media_desc(self.conf.archive_b)
        media_desc_a = replace(media_desc_a, audio=replace(media_desc_a.audio, codec='aad'))

        with self.assertRaisesRegex(StitcherException, 'Archives have incompatible audio formats.*'):
            validate_media(media_desc_a, media_desc_b)

    def test_validate_media_error_given_incompatible_video_properties(self):
        media_desc_a: MediaDesc = get_media_desc(self.conf.archive_a)
        media_desc_b: MediaDesc = get_media_desc(self.conf.archive_b)
        media_desc_a = replace(media_desc_a, video=replace(media_desc_a.video, codec='pipepiper'))

        with self.assertRaisesRegex(StitcherException, 'Archives have incompatible video formats.*'):
            validate_media(media_desc_a, media_desc_b)

    def test_validate_media_error_given_invalid_audio_codec(self):
        media_desc_a: MediaDesc = get_media_desc(self.conf.archive_a)
        media_desc_b: MediaDesc = get_media_desc(self.conf.archive_b)
        media_desc_a = replace(media_desc_a, audio=replace(media_desc_a.audio, codec='aad'))
        media_desc_b = replace(media_desc_b, audio=replace(media_desc_b.audio, codec='aad'))

        with self.assertRaisesRegex(StitcherException, 'This script only supports AAC audio'):
            validate_media(media_desc_a, media_desc_b)

    def test_validate_media_error_given_invalid_video_codec(self):
        media_desc_a: MediaDesc = get_media_desc(self.conf.archive_a)
        media_desc_b: MediaDesc = get_media_desc(self.conf.archive_b)
        media_desc_a = replace(media_desc_a, video=replace(media_desc_a.video, codec='pipepiper'))
        media_desc_b = replace(media_desc_b, video=replace(media_desc_b.video, codec='pipepiper'))

        with self.assertRaisesRegex(StitcherException, 'This script only supports H264 video'):
            validate_media(media_desc_a, media_desc_b)
