import unittest

import torch

from metrics.metrics import BinaryAUPRCHistogram
from train import DistributedEvalSampler


class DistributedMetricsTest(unittest.TestCase):
    def test_eval_sampler_partitions_without_duplicates_or_padding(self):
        dataset = list(range(10))
        partitions = [
            list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
            for rank in range(3)
        ]
        combined = [index for partition in partitions for index in partition]

        self.assertEqual(sorted(combined), list(range(10)))
        self.assertEqual(len(combined), len(set(combined)))

    def test_histogram_auprc_matches_known_average_precision(self):
        metric = BinaryAUPRCHistogram('cpu', bins=65536)
        metric.update(
            torch.tensor([0.9, 0.8, 0.7, 0.1]),
            torch.tensor([1, 0, 1, 0]),
        )

        self.assertAlmostEqual(metric.compute().item(), 5.0 / 6.0, places=6)

    def test_partitioned_histograms_equal_single_pass(self):
        scores = torch.tensor([0.95, 0.8, 0.6, 0.4, 0.2, 0.05])
        targets = torch.tensor([1, 0, 1, 0, 1, 0])
        whole = BinaryAUPRCHistogram('cpu', bins=1024)
        left = BinaryAUPRCHistogram('cpu', bins=1024)
        right = BinaryAUPRCHistogram('cpu', bins=1024)
        whole.update(scores, targets)
        left.update(scores[::2], targets[::2])
        right.update(scores[1::2], targets[1::2])

        merged = BinaryAUPRCHistogram('cpu', bins=1024)
        merged.positive = left.positive + right.positive
        merged.negative = left.negative + right.negative

        self.assertAlmostEqual(whole.compute().item(), merged.compute().item(), places=12)


if __name__ == '__main__':
    unittest.main()
