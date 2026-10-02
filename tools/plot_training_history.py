#!/usr/bin/env python3
"""Rebuild training curves from an existing epoch-level metrics.csv."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.training_logger import plot_training_history


def main():
    parser = argparse.ArgumentParser(
        description='Generate loss_curves.png and metrics_curves.png from metrics.csv.'
    )
    parser.add_argument('metrics_csv', help='path to an epoch-level metrics.csv')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    loss_path, metrics_path = plot_training_history(args.metrics_csv, args.output_dir)
    print(f'Generated: {loss_path}')
    print(f'Generated: {metrics_path}')


if __name__ == '__main__':
    main()
