"""Command-line interface for the FastF1 race predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .predictor import FastF1RacePredictor, PredictionResult


def _parse_event(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastf1-race-predictor",
        description="Predict Formula 1 race finishing order using FastF1 session data.",
    )
    parser.add_argument("--season", type=int, required=True, help="Target season, e.g. 2024")
    parser.add_argument(
        "--event",
        required=True,
        help="Event name or round number (e.g. 'Bahrain' or '1').",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="Optional list of session labels (default: FP1 FP2 FP3 Q).",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory used for FastF1 caching (default: ./cache).",
    )
    parser.add_argument(
        "--min-laps",
        type=int,
        default=5,
        help="Minimum laps before a driver's pace is considered reliable (default: 5).",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Limit the number of entries displayed (e.g. --top 10).",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Write the full prediction result to the specified JSON file.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Write the tabular predictions to the specified CSV file.",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Display the extended metric columns alongside the ranking.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit additional warnings when sessions cannot be processed.",
    )
    return parser


def _format_table(result: PredictionResult, *, top: Optional[int], show_metrics: bool) -> pd.DataFrame:
    table = result.table.copy()
    if top is not None:
        table = table.head(top)

    base_columns = [
        "predicted_position",
        "code",
        "driver",
        "team",
        "score",
        "confidence",
        "predicted_gap_seconds",
        "missing_sessions",
    ]
    display_columns = base_columns if show_metrics else base_columns[:-1]
    display_columns = [col for col in display_columns if col in table.columns]

    numeric_format = {
        "score": lambda v: f"{v:.4f}",
        "confidence": lambda v: f"{v:.2f}",
        "predicted_gap_seconds": lambda v: f"{v:.2f}",
    }

    for column, formatter in numeric_format.items():
        if column in table.columns:
            table[column] = table[column].map(formatter)

    if "missing_sessions" in table.columns and not show_metrics:
        table.drop(columns=["missing_sessions"], inplace=True)

    return table.loc[:, display_columns]


def _print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No predictions could be generated.")
        return
    print(df.to_string(index=False))


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    event_identifier = _parse_event(args.event)

    predictor = FastF1RacePredictor(
        cache_dir=args.cache_dir,
        min_laps=args.min_laps,
        verbose=args.verbose,
    )

    result = predictor.predict(
        args.season,
        event_identifier,
        sessions=args.sessions,
    )

    table = _format_table(result, top=args.top, show_metrics=args.show_metrics)
    _print_table(table)

    if args.export_json:
        result.save_json(args.export_json)

    if args.export_csv:
        result.table.to_csv(args.export_csv, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
