# copyright 2025 Vonage

from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from src.data_model import AlgoAudio, AlgoVideo, FindOverlapArgs, MediaDesc, MediaOverlap, OverlapInterval
from src.validations import get_media_desc
from src.overlap import *

from .test_base import TestBase


class OverlapTest(TestBase):
    def setUp(self):
        super().setUp()

        self.archive_a_desc: MediaDesc = get_media_desc(self.conf.archive_a)
        self.archive_b_desc: MediaDesc = get_media_desc(self.conf.archive_b)

        self.overlap_conf = FindOverlapArgs(
                duration_a=self.archive_a_desc.duration,
                duration_b=self.archive_b_desc.duration,
                video_desc=self.archive_a_desc.video,
                audio_desc=self.archive_a_desc.audio,
                max_overlap=timedelta(seconds=float(4)),
                algo_video=AlgoVideo.MSE,
                algo_audio=AlgoAudio.PEARSON,
                debug_plot=False,
                deep_search=False,
                )

    def validate_output(self, exp_ovr: MediaOverlap, ovr: MediaOverlap, a_delta_num: float, v_delta_num: float):
        self.assertNotEqual(exp_ovr, MediaOverlap())

        error_a_delta_dt = timedelta(seconds=a_delta_num)
        self.assertAlmostEqual(exp_ovr.audio.offset_a, ovr.audio.offset_a, delta=error_a_delta_dt)
        self.assertAlmostEqual(exp_ovr.audio.offset_b, ovr.audio.offset_b, delta=error_a_delta_dt)
        self.assertAlmostEqual(exp_ovr.audio.duration, ovr.audio.duration, delta=error_a_delta_dt)

        error_v_delta_dt = timedelta(seconds=v_delta_num)
        self.assertAlmostEqual(exp_ovr.video.offset_a, ovr.video.offset_a, delta=error_v_delta_dt)
        self.assertAlmostEqual(exp_ovr.video.offset_b, ovr.video.offset_b, delta=error_v_delta_dt)
        self.assertAlmostEqual(exp_ovr.video.duration, ovr.video.duration, delta=error_v_delta_dt)

    def test_find_overlap_success_given_video_with_mse(self):
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.MSE)

        self.assertNotEqual(self.overlap_conf.audio_desc, None)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        video_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=480000),
            timedelta(seconds=0, microseconds=40000),
            timedelta(seconds=2, microseconds=40000)
            )
        audio_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=457000),
            timedelta(milliseconds=0),
            timedelta(seconds=2, microseconds=32000)
            )
        expected_overlap = MediaOverlap(
                audio=audio_overlap_internal,
                video=video_overlap_internal,
                )
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0.4)

    def test_find_overlap_success_given_video_with_varlbp(self):
        self.overlap_conf.algo_video = AlgoVideo.VARLBP
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.VARLBP)

        self.assertNotEqual(self.overlap_conf.audio_desc, None)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        video_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=480000),
            timedelta(seconds=0, microseconds=40000),
            timedelta(seconds=2, microseconds=40000)
            )
        audio_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=457000),
            timedelta(milliseconds=0),
            timedelta(seconds=2, microseconds=32000)
            )
        expected_overlap = MediaOverlap(
                audio=audio_overlap_internal,
                video=video_overlap_internal,
                )
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0.4)

    def test_find_overlap_success_given_video_with_unilbp(self):
        self.overlap_conf.algo_video = AlgoVideo.UNILBP
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.UNILBP)

        self.assertNotEqual(self.overlap_conf.audio_desc, None)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        video_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=480000),
            timedelta(seconds=0, microseconds=40000),
            timedelta(seconds=2, microseconds=40000)
            )
        audio_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=457000),
            timedelta(milliseconds=0),
            timedelta(seconds=2, microseconds=32000)
            )
        expected_overlap = MediaOverlap(
                audio=audio_overlap_internal,
                video=video_overlap_internal,
                )
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0.8)

    def test_find_overlap_success_given_video_with_wavelet(self):
        self.overlap_conf.algo_video = AlgoVideo.WAVELET
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.WAVELET)

        self.assertNotEqual(self.overlap_conf.audio_desc, None)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        video_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=480000),
            timedelta(seconds=0, microseconds=40000),
            timedelta(seconds=2, microseconds=40000)
            )
        audio_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=457000),
            timedelta(milliseconds=0),
            timedelta(seconds=2, microseconds=32000)
            )
        expected_overlap = MediaOverlap(
                audio=audio_overlap_internal,
                video=video_overlap_internal,
                )
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0.4)

    def test_find_overlap_success_given_video_only(self):
        self.overlap_conf.audio_desc = None
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        video_overlap_internal = OverlapInterval(
            timedelta(seconds=57, microseconds=480000),
            timedelta(seconds=0, microseconds=40000),
            timedelta(seconds=2, microseconds=40000)
            )
        expected_overlap = MediaOverlap(
                audio=OverlapInterval(),
                video=video_overlap_internal,
                )
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b,
                replace(self.overlap_conf, audio_desc=None))

        self.validate_output(expected_overlap, overlap, 0, 0.4)

    def test_find_overlap_success_given_audio_only_with_deep_search(self):
        self.overlap_conf.video_desc = None
        self.overlap_conf.deep_search = True
        self.assertNotEqual(self.overlap_conf.audio_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(
                    timedelta(seconds=57, microseconds=457000),
                    timedelta(milliseconds=0),
                    timedelta(seconds=2, microseconds=32000)
                    ),
                video=OverlapInterval())
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0)

    def test_find_overlap_success_given_audio_only_without_deep_search(self):
        self.overlap_conf.video_desc = None
        self.assertFalse(self.overlap_conf.deep_search)
        self.assertNotEqual(self.overlap_conf.audio_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(
                    timedelta(seconds=57, microseconds=457000),
                    timedelta(milliseconds=0),
                    timedelta(seconds=2, microseconds=32000)
                    ),
                video=OverlapInterval())
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0)

    def test_mock_find_overlap_when_no_overlap_is_found(self):
        expected_overlap = MediaOverlap()

        with patch('src.overlap.find_overlap_video') as mock_video_overlap, \
                patch('src.overlap.find_overlap_audio') as mock_audio_overlap:
            mock_video_overlap.return_value = OverlapInterval()
            mock_audio_overlap.return_value = OverlapInterval()
            overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

            self.assertEqual(expected_overlap, overlap)

    def test_trim_init_duplicates_in_segment_with_repeated_values(self):
        in_values: list = [1, 1, 1, 2, 3, 4, 4, 4, 5]
        trimmed: Interval = trim_init_duplicates_in_segment(in_values, Interval(0, len(in_values)))

        self.assertEqual(Interval(2, 7), trimmed)

    def test_trim_init_duplicates_in_segment_without_repeated_values(self):
        in_values: list = [1, 2, 3, 4, 4, 4, 5]
        trimmed: Interval = trim_init_duplicates_in_segment(in_values, Interval(0, len(in_values)))

        self.assertEqual(Interval(0, len(in_values)), trimmed)

    def test_find_longest_non_decreasing_segment(self):
        in_values: list = [2, 3, 4, 0, 0, 6, 3, 7, 8, 11, 1, 1, 1, 2, 3, 4, 4, 4, 5, 1, 2, 2, 2]
        interval = find_longest_non_decreasing_segment(in_values)

        self.assertEqual(Interval(10, 9), interval)

    def test_get_overlapping_indexes(self):
        in_values: list = [2, 3, 4, 0, 0, 6, 3, 7, 8, 11, 1, 1, 1, 2, 3, 4, 4, 4, 5, 1, 2, 2, 2]
        overlapping_interval = get_overlapping_indexes(in_values)
        self.assertEqual(Interval(12, 7), overlapping_interval)

        interval = find_longest_non_decreasing_segment(in_values)

        self.assertTrue(interval.ini < overlapping_interval.ini)
        self.assertEqual(overlapping_interval.end, interval.end)

    def test_mock_get_matching_frames(self):
        archive_a_duration: float = self.overlap_conf.duration_a.total_seconds()
        overlap_duration: float = self.overlap_conf.max_overlap.total_seconds()
        self.assertTrue(archive_a_duration > overlap_duration)

        with patch('src.overlap.get_matching_frames') as mock_get_matches:
            find_overlap_video(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

            mock_get_matches.assert_called_once_with(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

    def test_get_matching_frames(self):
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.MSE)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        expected_values = pd.DataFrame({
            'frame_a': [1, 2, 3, 100],
            'similar_frame_b': [100, 100, 72, 70]
        })

        values: pd.DataFrame = get_matching_frames(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.assertTrue(expected_values['frame_a'].iloc[0], values['frame_a'].iloc[0])
        self.assertTrue(expected_values['frame_a'].iloc[1], values['frame_a'].iloc[1])
        self.assertTrue(expected_values['frame_a'].iloc[2], values['frame_a'].iloc[2])
        self.assertTrue(expected_values['frame_a'].iloc[-1], values['frame_a'].iloc[-1])

        self.assertTrue(expected_values['similar_frame_b'].iloc[0], values['similar_frame_b'].iloc[0])
        self.assertTrue(expected_values['similar_frame_b'].iloc[1], values['similar_frame_b'].iloc[1])
        self.assertTrue(expected_values['similar_frame_b'].iloc[2], values['similar_frame_b'].iloc[2])
        self.assertTrue(expected_values['similar_frame_b'].iloc[-1], values['similar_frame_b'].iloc[-1])

    def test_get_overlap_interval_video_when_frame_video_a_end_and_frame_video_a_ini_diff_is_lower_than_one(self):
        expected_overlap = OverlapInterval()

        overlap: OverlapInterval = find_overlap_video(
                self.conf.archive_a, self.conf.archive_b, replace(self.overlap_conf, duration_a=timedelta()))

        self.assertEqual(expected_overlap, overlap)

    def test_get_overlap_interval_video_when_frame_video_b_ini_and_frame_video_b_end_are_the_same(self):
        expected_overlap = OverlapInterval()

        overlap: OverlapInterval = find_overlap_video(
                self.conf.archive_a, self.conf.archive_b, replace(self.overlap_conf, max_overlap=timedelta()))

        self.assertEqual(expected_overlap, overlap)

    def test_get_overlap_interval_video_when_overlapping_index_are_wrongly_selected(self):
        '''
        start_overlap_index >= final_overlap_index
        '''
        expected_overlap = MediaOverlap()

        start_overlap_index = 20
        final_overlap_index = 15
        with patch('src.overlap.get_overlapping_indexes') as mock_get_indexes:
            mock_get_indexes.return_value = Interval(start_overlap_index, final_overlap_index - start_overlap_index + 1)
    
            overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b,
                                                 replace(self.overlap_conf, audio_desc=None))

            self.assertEqual(expected_overlap, overlap)

    def test_plot_archives_relationship(self):
        self.overlap_conf.debug_plot = True
        with patch('matplotlib.pyplot.show') as mock_plt:
            plot_archives_relationship([1, 2, 3, 4, 5, 6, 7], Interval(1, 5))
            mock_plt.assert_called_once()

    def test_plot_audio_samples(self):
        self.overlap_conf.debug_plot = True
        with patch('matplotlib.pyplot.show') as mock_plt:
            y_a: np.ndarray = np.array([[1.0, 2.2, 3.9], [4.9, 5.7, 6.0]])
            y_b: np.ndarray = np.array([[4.9, 5.7, 6.0], [10.2, 29.3, 33.1]])
            plot_audio_samples(y_a, y_b, 1, 1, 5, 4, 8)
            mock_plt.assert_called_once()
