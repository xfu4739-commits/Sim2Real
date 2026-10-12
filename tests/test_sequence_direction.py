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
