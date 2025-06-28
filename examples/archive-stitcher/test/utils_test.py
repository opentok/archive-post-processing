# copyright 2025 Vonage

import unittest

from datetime import timedelta

from src import utils
from src.data_model import OverlapInterval


class UtilsTest(unittest.TestCase):

    def test_run_exec_success(self):
        self.assertEqual('hello', utils.run_exec('printf', 'hello'))

    def test_run_exec_failure(self):
        stderr: str = 'error in stdrerror'
        stdout: str = 'some output'
        exit_code: int = 25
        with self.assertRaises(utils.StitcherException) as context:
            args = ['bash', '-c', f'printf "{stderr}" >&2 && printf "{stdout}" && exit {exit_code}']
            utils.run_exec(*args)

        self.assertIn(f'Command "{tuple(args)}" failed with status {exit_code}', str(context.exception))
        self.assertIn(f'stdout: {stdout}', str(context.exception))
        self.assertIn(f'stderr: {stderr}', str(context.exception))

    def test_overlap_interval_intersection(self):
        seconds_fn = lambda x: timedelta(seconds=x)
        interval1 = OverlapInterval(offset_a=seconds_fn(10), offset_b=seconds_fn(2), duration=seconds_fn(4))
        interval2 = OverlapInterval(offset_a=seconds_fn(12), offset_b=seconds_fn(1), duration=seconds_fn(7))
        intersection = interval1.intersection(interval2)

        self.assertEqual(OverlapInterval(offset_a=seconds_fn(12), offset_b=seconds_fn(2), duration=seconds_fn(2)),
                         intersection)

        interval1 = OverlapInterval(offset_a=seconds_fn(13), offset_b=seconds_fn(2), duration=seconds_fn(4))
        interval2 = OverlapInterval(offset_a=seconds_fn(12), offset_b=seconds_fn(3), duration=seconds_fn(2))
        intersection = interval1.intersection(interval2)

        self.assertEqual(OverlapInterval(offset_a=seconds_fn(13), offset_b=seconds_fn(3), duration=seconds_fn(1)),
                         intersection)

        # Non-overlapping intervals
        interval1 = OverlapInterval(offset_a=seconds_fn(13), offset_b=seconds_fn(2), duration=seconds_fn(4))
        interval2 = OverlapInterval(offset_a=seconds_fn(17), offset_b=seconds_fn(3), duration=seconds_fn(2))
        intersection = interval1.intersection(interval2)
        self.assertTrue(intersection.is_empty())

        interval1 = OverlapInterval(offset_a=seconds_fn(13), offset_b=seconds_fn(7), duration=seconds_fn(4))
        interval2 = OverlapInterval(offset_a=seconds_fn(10), offset_b=seconds_fn(3), duration=seconds_fn(4))
        intersection = interval1.intersection(interval2)
        self.assertTrue(intersection.is_empty())

        interval1 = OverlapInterval(offset_a=seconds_fn(13), offset_b=seconds_fn(7), duration=seconds_fn(4))
        interval2 = OverlapInterval(offset_a=seconds_fn(20), offset_b=seconds_fn(3), duration=seconds_fn(4))
        intersection = interval1.intersection(interval2)
        self.assertTrue(intersection.is_empty())
