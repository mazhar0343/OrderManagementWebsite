"""FastF1-based Formula 1 race predictor.

This package exposes the :class:`FastF1RacePredictor` class which combines
timing data from practice and qualifying sessions to produce an estimated race
result ordering.
"""

from .predictor import FastF1RacePredictor, PredictionResult

__all__ = [
    "FastF1RacePredictor",
    "PredictionResult",
]
