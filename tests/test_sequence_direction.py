from datetime import datetime, timedelta
import unittest

import numpy as np

from dataset.real_dataset import RealFireDataset
from dataset.sim_dataset import SimFireDataset


def _strictly_increasing(values):
    return all(left < right for left, right in zip(values, values[1:]))


def _strictly_decreasing(values):
    return all(left > right for left, right in zip(values, values[1:]))


class SequenceDirectionTest(unittest.TestCase):
    def test_simulation_forecast_sequence_runs_earlier_to_later(self):
        dataset = object.__new__(SimFireDataset)
        dataset.sequence_rng = np.random.default_rng(7)

        sequence = dataset.extract_sequences_interval_random(1, 100, 6, reverse=False)

        self.assertIsNotNone(sequence)
        self.assertTrue(_strictly_increasing(sequence))
        self.assertLess(max(sequence[:3]), min(sequence[3:]))

    def test_simulation_backtracking_sequence_runs_later_to_earlier(self):
        dataset = object.__new__(SimFireDataset)
        dataset.sequence_rng = np.random.default_rng(7)

        sequence = dataset.extract_sequences_interval_random(1, 100, 6, reverse=True)

        self.assertIsNotNone(sequence)
        self.assertTrue(_strictly_decreasing(sequence))
        self.assertGreater(min(sequence[:3]), max(sequence[3:]))

    def test_real_sequences_and_timestamps_preserve_task_direction(self):
        dataset = object.__new__(RealFireDataset)
        origin = datetime(2024, 1, 1)
        dates = [origin + timedelta(days=offset) for offset in range(6)]

        dataset.reverse = False
        forecast = dataset.extract_sequences(end=6, sample=6)[0]
        forecast_times = dataset.normalize_timestamps([dates[index] for index in forecast])

        dataset.reverse = True
        backtrack = dataset.extract_sequences(end=6, sample=6)[0]
        backtrack_times = dataset.normalize_timestamps([dates[index] for index in backtrack])

        self.assertTrue(_strictly_increasing(forecast))
        self.assertTrue(_strictly_increasing(forecast_times))
        self.assertTrue(_strictly_decreasing(backtrack))
        self.assertTrue(_strictly_decreasing(backtrack_times))

    def test_sim_and_real_use_the_same_six_frame_time_scale(self):
        sim = object.__new__(SimFireDataset)
        real = object.__new__(RealFireDataset)
        origin = datetime(2024, 1, 1)
        for sequence, expected in (
            ([40, 41, 42, 43, 44, 45], [0.6, 0.8, 1.0]),
            ([45, 44, 43, 42, 41, 40], [0.4, 0.2, 0.0]),
        ):
            sim_times = sim.normalize_timestamps(sequence)
            real_times = real.normalize_timestamps(
                [origin + timedelta(days=index) for index in sequence]
            )
            for actual, desired, real_value in zip(sim_times, expected, real_times):
                self.assertAlmostEqual(actual, desired)
                self.assertAlmostEqual(actual, real_value)

        self.assertEqual(sim.normalize_timestamps([40] * 6), [0.0] * 3)

    def test_real_irregular_dates_keep_existing_local_normalization(self):
        real = object.__new__(RealFireDataset)
        origin = datetime(2024, 1, 1)
        # The same relative gaps across two- and twenty-day windows must
        # retain the real dataset's original six-frame values.
        for scale in (1, 10):
            dates = [
                origin + timedelta(hours=scale * offset)
                for offset in (0, 4, 8, 12, 24, 48)
            ]
            normalized = real.normalize_timestamps(dates)
            for actual, expected in zip(normalized, (0.25, 0.5, 1.0)):
                self.assertAlmostEqual(actual, expected)

            reversed_normalized = real.normalize_timestamps(dates[::-1])
            for actual, expected in zip(reversed_normalized, (1 / 6, 1 / 12, 0.0)):
                self.assertAlmostEqual(actual, expected)

    def test_simulation_sequence_is_independent_of_rank_global_seed(self):
        rank_zero = object.__new__(SimFireDataset)
        rank_one = object.__new__(SimFireDataset)
        rank_zero.sequence_rng = np.random.default_rng(12345)
        rank_one.sequence_rng = np.random.default_rng(12345)

        np.random.seed(12345)
        zero_sequences = [
            rank_zero.extract_sequences_interval_random(1, 100, 6, reverse=True)
            for _ in range(20)
        ]
        np.random.seed(12346)
        one_sequences = [
            rank_one.extract_sequences_interval_random(1, 100, 6, reverse=True)
            for _ in range(20)
        ]

        self.assertEqual(zero_sequences, one_sequences)


if __name__ == "__main__":
    unittest.main()
