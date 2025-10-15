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
                max_overlap=timedelta(seconds=float(2.1)),
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

    def test_find_overlap_success_given_only_video_with_mse(self):
        self.overlap_conf.audio_desc = None
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.MSE)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(),
                video=OverlapInterval(
                    timedelta(seconds=57, microseconds=480000),
                    timedelta(seconds=0, microseconds=40000),
                    timedelta(seconds=2, microseconds=40000)
                    ))
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0, 0.4)

    def test_find_overlap_success_given_only_video_with_varlbp(self):
        self.overlap_conf.audio_desc = None
        self.overlap_conf.algo_video = AlgoVideo.VARLBP
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.VARLBP)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(),
                video=OverlapInterval(
                    timedelta(seconds=57, microseconds=480000),
                    timedelta(seconds=0, microseconds=40000),
                    timedelta(seconds=2, microseconds=40000)
                    ))
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0, 0.4)

    def test_find_overlap_success_given_only_video_with_unilbp(self):
        self.overlap_conf.audio_desc = None
        self.overlap_conf.algo_video = AlgoVideo.UNILBP
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.UNILBP)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(),
                video=OverlapInterval(
                    timedelta(seconds=57, microseconds=480000),
                    timedelta(seconds=0, microseconds=40000),
                    timedelta(seconds=2, microseconds=40000)
                    ))
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0, 0.8)

    def test_find_overlap_success_given_only_video_with_wavelet(self):
        self.overlap_conf.audio_desc = None
        self.overlap_conf.algo_video = AlgoVideo.WAVELET
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.WAVELET)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(),
                video=OverlapInterval(
                    timedelta(seconds=57, microseconds=480000),
                    timedelta(seconds=0, microseconds=40000),
                    timedelta(seconds=2, microseconds=40000)
                    ))
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0, 0.4)

    def test_find_overlap_success_given_only_audio_with_deep_search(self):
        self.overlap_conf.video_desc = None
        self.overlap_conf.deep_search = True
        self.assertNotEqual(self.overlap_conf.audio_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(
                    timedelta(seconds=57, microseconds=457000),
                    timedelta(milliseconds=0),
                    timedelta(seconds=1, microseconds=990000)
                    ),
                video=OverlapInterval())
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.4, 0)

    def test_find_overlap_success_given_only_audio_without_deep_search(self):
        self.overlap_conf.video_desc = None
        self.assertFalse(self.overlap_conf.deep_search)
        self.assertNotEqual(self.overlap_conf.audio_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(
                    timedelta(seconds=58, microseconds=23667),
                    timedelta(microseconds=586667),
                    timedelta(seconds=1, microseconds=200000)
                    ),
                video=OverlapInterval())
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.8, 0)

    def test_find_overlap_success_given_audio_and_video(self):
        self.assertEqual(self.overlap_conf.algo_video, AlgoVideo.MSE)

        self.assertNotEqual(self.overlap_conf.audio_desc, None)
        self.assertNotEqual(self.overlap_conf.video_desc, None)

        expected_overlap = MediaOverlap(
                audio=OverlapInterval(
                    timedelta(seconds=57, microseconds=457000),
                    timedelta(milliseconds=0),
                    timedelta(seconds=1)
                    ),
                video=OverlapInterval(
                    timedelta(seconds=58, microseconds=23500),
                    timedelta(seconds=0, microseconds=586650),
                    timedelta(seconds=1, microseconds=200000)
                    ))
        overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

        self.validate_output(expected_overlap, overlap, 0.8, 0.8)

    def test_mock_find_overlap_when_no_overlap_is_found(self):
        expected_overlap = MediaOverlap()

        with patch('src.overlap.find_overlap_video') as mock_video_overlap, \
                patch('src.overlap.find_overlap_audio') as mock_audio_overlap:
            mock_video_overlap.return_value = OverlapInterval()
            mock_audio_overlap.return_value = OverlapInterval()
            overlap: MediaOverlap = find_overlap(self.conf.archive_a, self.conf.archive_b, self.overlap_conf)

            self.assertEqual(expected_overlap, overlap)

    def test_trim_init_duplicates_in_segment_with_repeated_values(self):
        in_values = [1, 1, 1, 2, 3, 4, 4, 4, 5]
        trimmed: Interval = trim_init_duplicates_in_segment(in_values, Interval(0, len(in_values)))

        self.assertEqual(Interval(2, 7), trimmed)

    def test_trim_init_duplicates_in_segment_without_repeated_values(self):
        in_values = [1, 2, 3, 4, 4, 4, 5]
        trimmed: Interval = trim_init_duplicates_in_segment(in_values, Interval(0, len(in_values)))

        self.assertEqual(Interval(0, len(in_values)), trimmed)

    def test_trim_end_duplicates_in_segment_with_repeated_values(self):
        in_values = [1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5]
        trimmed: Interval = trim_end_duplicates_in_segment(in_values, Interval(0, len(in_values)))

        self.assertEqual(Interval(0, 9), trimmed)

    def test_trim_end_duplicates_in_segment_without_repeated_values(self):
        in_values = [1, 2, 3, 4, 4, 4, 5]
        trimmed: Interval = trim_end_duplicates_in_segment(in_values, Interval(0, len(in_values)))

        self.assertEqual(Interval(0, len(in_values)), trimmed)

    def test_is_data_increasing_in_45_degrees_trend_when_data_follows_the_45_line_trend(self):
        in_values = [1, 2, 3, 4, 4, 5, 6, 7]

        self.assertTrue(is_data_increasing_in_45_degrees_trend(in_values))

    def test_is_data_increasing_in_45_degrees_trend_when_data_does_not_follow_the_45_line_trend(self):
        in_values = [1, 3, 3, 3, 10, 10, 10, 20]

        self.assertFalse(is_data_increasing_in_45_degrees_trend(in_values))

    def test_get_increasing_data_intervals_filter_with_increasing_intervals(self):
        values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 13, 14, 14, 15, 15, 16, 16, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 31, 33, 34, 35, 36, 37, 25, 26, 27, 28, 29, 30, 31, 32, 0, 158, 159, 0, 0, 4, 5, 4,
            6, 7, 9, 10, 11, 12, 12, 13, 13, 14, 14, 199, 200, 200, 201, 209, 209, 208, 209, 213, 220, 219, 86, 87,
            89, 91, 78, 78, 305, 305, 305, 306, 0, 140, 225, 75, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 87, 88, 89,
            90, 91, 92, 93, 94, 104, 105, 106, 116, 117, 118, 134, 135, 137, 138, 139, 140, 141, 0, 0, 0, 260, 262,
            264, 265, 267, 269, 270, 304, 303, 301, 300]
        interval_list = [Interval(ini=0, length=12), Interval(ini=12, length=25),
            Interval(ini=45, length=5), Interval(ini=90, length=33)]
        expected_interval_list = [Interval(ini=0, length=12), Interval(ini=12, length=25)]

        self.assertEqual(get_increasing_data_intervals(values, interval_list), expected_interval_list)

    def test_get_increasing_data_intervals_when_interval_is_not_increasing_at_45_degrees(self):
        values = [5, 5, 5, 6, 10, 11, 15, 20, 20, 20, 21, 22, 40, 50, 100]
        interval_list = [Interval(ini=0, length=len(values))]
        expected_interval_list = []

        self.assertEqual(get_increasing_data_intervals(values, interval_list), expected_interval_list)

    def test_add_unique_element_to_list_if_element_is_not_in_the_set(self):
        segments_in_set = set()
        segments_in_list = list()
        interval_list = add_unique_element_to_list(0, 150, segments_in_set, segments_in_list)

        self.assertEqual([Interval(ini=0, length=150)], interval_list)

    def test_add_unique_element_to_list_adds_another_element_to_a_non_empty_list(self):
        segments_in_set = {(0, 117)}
        segments_in_list = [Interval(ini=0, length=117)]
        interval_list = add_unique_element_to_list(117, 168, segments_in_set, segments_in_list)

        self.assertEqual([Interval(ini=0, length=117), Interval(ini=117, length=168)], interval_list)

    def test_add_unique_element_to_list_does_not_add_a_repeated_element(self):
        segments_in_set = {(0, 117), (117, 168)}
        segments_in_list = [Interval(ini=0, length=117), Interval(ini=117, length=168)]
        interval_list = add_unique_element_to_list(117, 168, segments_in_set, segments_in_list)

        self.assertEqual(segments_in_list, interval_list)

    def test_remove_glitches_discards_intervals_that_are_not_glitches(self):
        values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 13, 14, 14, 15, 15, 16, 16, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 33, 34, 35, 36, 37, 25, 26, 27, 28, 29, 30, 31, 32,
            0, 158, 159, 0, 0, 4, 5, 4, 6, 7, 9, 10, 11, 12, 12, 13, 13, 14, 14, 199, 200, 200, 201,
            209, 209, 208, 209, 213, 220, 219, 86, 87, 89, 91, 78, 78, 305, 305, 305, 306, 0, 140, 225,
            75, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 87, 88, 89, 90, 91, 92, 93, 94, 104, 105, 106, 
            116, 117, 118, 134, 135, 137, 138, 139, 140, 141, 0, 0, 0, 260, 262, 264, 265, 267, 269, 270,
            304, 303, 301, 300]
        longest_segments_list = [Interval(ini=0, length=12), Interval(ini=12, length=25), Interval(ini=37, length=8),
            Interval(ini=64, length=6), Interval(ini=90, length=33), Interval(ini=123, length=11)]

        self.assertEqual(Interval(ini=12, length=25), remove_glitches(values, longest_segments_list, MediaType.AUDIO))

    def test_remove_glitches_joins_consecutive_intervals_with_outliers(self):
        values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 13, 14, 14, 15, 15, 16, 16, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 33, 34, 35, 36, 37, 25, 26, 27, 28, 29, 30, 31,
            32, 0, 158, 159, 0, 0, 4, 5, 4, 6, 7, 9, 10, 11, 12, 12, 13, 13, 14, 14, 199, 200, 200,
            201, 209, 209, 208, 209, 213, 220, 219, 86, 87, 89, 91, 78, 78, 305, 305, 305, 306, 0, 140,
            225, 75, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 87, 88, 89, 90, 91, 92, 93, 94, 104, 105,
            106, 116, 117, 118, 134, 135, 137, 138, 139, 140, 141, 0, 0, 0, 260, 262, 264, 265, 267, 269,
            270, 304, 303, 301, 300]
        longest_segments_list = [Interval(ini=0, length=12), Interval(ini=12, length=25), Interval(ini=90, length=33)]

        self.assertEqual(Interval(ini=12, length=25), remove_glitches(values, longest_segments_list, MediaType.AUDIO))

    def test_remove_glitches_returns_empty_overlap_interval_if_media_type_is_undefined(self):
        values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 13, 14, 14, 15, 15, 16, 16, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 33, 34, 35, 36, 37, 25, 26, 27, 28, 29, 30, 31,
            32, 0, 158, 159, 0, 0, 4, 5, 4, 6, 7, 9, 10, 11, 12, 12, 13, 13, 14, 14, 199, 200, 200,
            201, 209, 209, 208, 209, 213, 220, 219, 86, 87, 89, 91, 78, 78, 305, 305, 305, 306, 0, 140,
            225, 75, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 87, 88, 89, 90, 91, 92, 93, 94, 104, 105,
            106, 116, 117, 118, 134, 135, 137, 138, 139, 140, 141, 0, 0, 0, 260, 262, 264, 265, 267, 269,
            270, 304, 303, 301, 300]
        longest_segments_list = [Interval(ini=0, length=12), Interval(ini=12, length=25), Interval(ini=90, length=33)]

        self.assertEqual(Interval(ini=0, length=0), remove_glitches(values, longest_segments_list, MediaType.UNDEFINED))

    def test_remove_glitches_joins_consecutive_intervals_with_outliers_that_entry_the_if_clause(self):
        values = [110, 82, 123, 123, 77, 129, 120, 74, 120, 118, 78, 78, 78, 79, 120, 150, 130, 119,
            123, 133, 124, 129, 124, 116, 124, 116, 117, 117, 120, 120, 119, 111, 111, 103, 120, 117,
            120, 117, 118, 121, 124, 120, 124, 131, 128, 128, 127, 117, 131, 131, 114, 133, 129, 129,
            129, 129, 25, 25, 26, 26, 27, 116, 116, 117, 116, 118, 117, 117, 118, 118, 115, 118, 118,
            114, 116, 116, 116, 116, 116, 115, 115, 115, 115, 36, 36, 38, 35, 38, 34, 39, 107, 38, 38,
            113, 114, 115, 115, 115, 116, 116, 117, 117, 115, 115, 115, 115, 115, 116, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 137, 137, 138, 319,
            320, 320, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
            338, 339, 340]
        longest_segments_list = [Interval(ini=56, length=8), Interval(ini=91, length=11), Interval(ini=102, length=48)]

        self.assertEqual(Interval(ini=102, length=48), remove_glitches(values, longest_segments_list, MediaType.AUDIO))

    def test_find_longest_non_decreasing_segment(self):
        in_values: list = [2, 3, 4, 0, 0, 6, 3, 7, 8, 11, 1, 1, 1, 2, 3, 4, 4, 4, 5, 1, 2, 2, 2]

        self.assertEqual(Interval(10, 9), find_longest_non_decreasing_segment(in_values, MediaType.AUDIO))
        self.assertEqual(Interval(10, 9), find_longest_non_decreasing_segment(in_values, MediaType.VIDEO))

    def test_get_overlapping_video_indexes(self):
        in_values: list = [2, 3, 4, 0, 0, 6, 3, 7, 8, 11, 1, 1, 1, 2, 3, 4, 4, 4, 5, 1, 2, 2, 2]
        overlapping_interval = get_overlapping_indexes(in_values, MediaType.VIDEO)
        self.assertEqual(Interval(12, 7), overlapping_interval)

        interval_a = find_longest_non_decreasing_segment(in_values, MediaType.AUDIO)
        interval_v = find_longest_non_decreasing_segment(in_values, MediaType.VIDEO)

        self.assertEqual(interval_a, interval_v)
        self.assertTrue(interval_a.ini < overlapping_interval.ini)
        self.assertEqual(overlapping_interval.end, interval_a.end)

    def test_get_overlapping_audio_indexes(self):
        in_values: dict = {}
        in_values[0] = SimilarityEntry(index_i=5, corr=0.95, sim=0.9)
        in_values[1] = SimilarityEntry(index_i=6, corr=0.95, sim=0.9)
        in_values[2] = SimilarityEntry(index_i=6, corr=0.95, sim=0.9)
        in_values[3] = SimilarityEntry(index_i=7, corr=0.95, sim=0.9)
        in_values[4] = SimilarityEntry(index_i=8, corr=0.95, sim=0.9)
        in_values[5] = SimilarityEntry(index_i=9, corr=0.95, sim=0.9)
        in_values[6] = SimilarityEntry(index_i=0, corr=0.95, sim=0.9)
        in_values[7] = SimilarityEntry(index_i=14, corr=0.95, sim=0.9)
        in_values[8] = SimilarityEntry(index_i=17, corr=0.95, sim=0.9)
        in_values[9] = SimilarityEntry(index_i=15, corr=0.95, sim=0.9)
        in_values[11] = SimilarityEntry(index_i=7, corr=0.95, sim=0.9)
        in_values[13] = SimilarityEntry(index_i=6, corr=0.95, sim=0.9)
        in_values[15] = SimilarityEntry(index_i=17, corr=0.95, sim=0.9)
        in_values[16] = SimilarityEntry(index_i=18, corr=0.95, sim=0.9)

        index_values = [obj.index_i for obj in in_values.values()]
        overlapping_interval = get_overlapping_indexes(index_values, MediaType.AUDIO)
        self.assertEqual(Interval(0, 6), overlapping_interval)

    def test_select_by_difference_if_lists_are_the_same(self):
        list_1 = [2, 1, np.float32(0.2), np.float32(0.1)]
        self.assertEqual(select_by_difference(list_1, list_1, True), list_1)
        self.assertEqual(select_by_difference(list_1, list_1, False), list_1)
        
    def test_select_by_difference_with_is_last_index_true(self):
        list_1 = [20, 10, np.float32(0.2), np.float32(0.1)]
        list_2 = [33, 13, np.float32(0.2), np.float32(0.1)]
        self.assertEqual(select_by_difference(list_1, list_2, True), list_2)
        self.assertEqual(select_by_difference(list_2, list_1, True), list_1)

    def test_select_by_difference_with_is_last_index_false(self):
        list_1 = [20, 10, np.float32(0.2), np.float32(0.1)]
        list_2 = [18, 1, np.float32(0.2), np.float32(0.1)]
        self.assertEqual(select_by_difference(list_1, list_2, False), list_2)
        self.assertEqual(select_by_difference(list_2, list_1, False), list_1)

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
