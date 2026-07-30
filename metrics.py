"""
Root metrics launcher script.
Re-exports metrics evaluation functions from src/metrics.py
"""
import sys
import os

from src.metrics import (
    normalize_persian_for_eval,
    WhisperMetricsEvaluator,
    get_compute_metrics_fn
)

if __name__ == '__main__':
    print("Testing root metrics.py launcher...")
