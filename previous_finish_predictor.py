#!/usr/bin/env python3
"""Train a simple Formula 1 race predictor using previous finishing positions.

This script downloads race classification results through the FastF1 API for the
requested seasons (default: 2022 through the current year), builds a training
dataset that maps each driver's finishing position in race *n-1* to their
finishing position in race *n*, trains a small tree-based regressor, and uses it
to forecast the next Grand Prix.

Example usages::

    python previous_finish_predictor.py                      # train & predict next race
    python previous_finish_predictor.py --season 2024 --event 18  # predict 2024 round 18
    python previous_finish_predictor.py --export-dataset data.csv

The model is intentionally simple: the only feature is the finishing position
from the previous race a driver completed. This baseline can be extended later
by adding richer features.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


log = logging.getLogger(__name__)


def _jsonable_timestamp(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:  # pragma: no cover - defensive only
        return None


def _parse_event(value: str) -> str | int:
    value = value.strip()
    return int(value) if value.isdigit() else value


def _normalise_round_number(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive only
        return None


def find_upcoming_event(season: int) -> Tuple[int, pd.Series]:
    schedule = fastf1.get_event_schedule(season)
    schedule = schedule.loc[schedule["EventFormat"].str.lower() != "testing"].copy()

    utc_now = pd.Timestamp.utcnow().tz_convert("UTC")
    today = utc_now.normalize().tz_localize(None)
    schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])
    schedule["EventDate"] = schedule["EventDate"].dt.tz_localize(None)

    upcoming = schedule.loc[schedule["EventDate"] >= today]
    if upcoming.empty:
        upcoming = schedule.tail(1)

    event_row = upcoming.iloc[0]
    round_number = _normalise_round_number(event_row.get("RoundNumber"))
    if round_number is None:
        raise RuntimeError("Could not determine round number for upcoming event")
    return round_number, event_row


def resolve_event(season: int, event: Optional[str | int]) -> Tuple[int, pd.Series, fastf1.events.Event]:
    schedule = fastf1.get_event_schedule(season)
    schedule = schedule.loc[schedule["EventFormat"].str.lower() != "testing"].copy()

    if event is not None:
        event_identifier = _parse_event(str(event)) if isinstance(event, str) else event
        event_info = fastf1.get_event(season, event_identifier)
        if event_info is None:
            raise RuntimeError(f"Could not resolve event '{event}' for season {season}")
        round_number = _normalise_round_number(getattr(event_info, "RoundNumber", None))
        if round_number is None:
            raise RuntimeError("Resolved event is missing a round number")
        event_rows = schedule.loc[schedule["RoundNumber"] == round_number]
        if event_rows.empty:
            event_row = pd.Series({
                "EventName": getattr(event_info, "EventName", str(event_identifier)),
                "RoundNumber": round_number,
                "EventDate": getattr(event_info, "EventDate", pd.NaT),
            })
        else:
            event_row = event_rows.iloc[0]
    else:
        round_number, event_row = find_upcoming_event(season)
        event_info = fastf1.get_event(season, round_number)
        if event_info is None:
            raise RuntimeError(f"Could not fetch event metadata for round {round_number}")

    round_number = _normalise_round_number(event_row.get("RoundNumber"))
    if round_number is None:
        raise RuntimeError("Event schedule record did not contain a round number")

    return round_number, event_row, event_info


def find_previous_race(season: int, round_number: int) -> Tuple[int, int]:
    if round_number > 1:
        return season, round_number - 1

    prev_season = season - 1
    if prev_season < 1950:  # sanity; F1 started 1950
        raise RuntimeError("No previous race available to reference")

    schedule = fastf1.get_event_schedule(prev_season)
    schedule = schedule.loc[schedule["EventFormat"].str.lower() != "testing"].copy()
    if schedule.empty:
        raise RuntimeError(f"No race schedule found for season {prev_season}")

    prev_round_candidates = schedule["RoundNumber"].dropna().astype(int)
    if prev_round_candidates.empty:
        raise RuntimeError(f"Could not determine previous round for season {prev_season}")

    prev_round = int(prev_round_candidates.max())
    return prev_season, prev_round


@dataclass
class TrainingMetrics:
    mae: float
    r2: float
    baseline_mae: float
    samples: int
    test_samples: int


class PreviousFinishPredictor:
    """Predict race finishing order based solely on previous finishing position."""

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        *,
        model: Optional[RandomForestRegressor] = None,
        verbose: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        self.verbose = verbose
        self.model = model or RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=1,
            random_state=42,
        )
        self._trained = False
        self._dataset: Optional[pd.DataFrame] = None
        self._last_finish_lookup: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Data collection

    def build_dataset(
        self,
        *,
        start_season: int = 2022,
        end_season: Optional[int] = None,
    ) -> pd.DataFrame:
        """Collect race results and build a modelling dataset."""

        if end_season is None:
            end_season = datetime.now(timezone.utc).year

        records: List[Dict[str, object]] = []
        last_finish: Dict[str, Dict[str, int]] = {}

        seasons = list(range(start_season, end_season + 1))
        for season in seasons:
            try:
                schedule = fastf1.get_event_schedule(season)
            except Exception as exc:  # pragma: no cover - network dependent
                raise RuntimeError(f"Failed to download schedule for season {season}: {exc}") from exc

            schedule = schedule.loc[schedule["EventFormat"].str.lower() != "testing"].copy()
            schedule = schedule.sort_values("RoundNumber")

            for _, event_row in schedule.iterrows():
                round_number = _normalise_round_number(event_row.get("RoundNumber"))
                if round_number is None:
                    continue

                try:
                    session = fastf1.get_session(season, round_number, "R")
                    session.load(laps=False, telemetry=False, weather=False, messages=False)
                except Exception as exc:  # pragma: no cover - network dependent
                    if self.verbose:
                        log.warning("Skipping %s %s: %s", season, event_row.get("EventName"), exc)
                    continue

                results = session.results.copy()
                if results is None or results.empty:
                    continue

                results = results.loc[results["Position"].notna()].copy()
                if results.empty:
                    continue

                results["Position"] = pd.to_numeric(results["Position"], errors="coerce")
                results = results.loc[results["Position"].notna()].copy()
                results["Position"] = results["Position"].astype(int)

                for _, driver_row in results.iterrows():
                    driver_number = str(driver_row.get("DriverNumber"))
                    prev_entry = last_finish.get(driver_number)

                    record = {
                        "season": season,
                        "round": round_number,
                        "event": session.event.EventName,
                        "event_date": _jsonable_timestamp(event_row.get("EventDate")),
                        "driver_number": driver_number,
                        "driver": driver_row.get("FullName") or driver_row.get("BroadcastName") or driver_number,
                        "code": driver_row.get("Abbreviation") or driver_number,
                        "team": driver_row.get("TeamName", ""),
                        "finish_position": int(driver_row.get("Position")),
                        "status": driver_row.get("Status"),
                        "prev_finish_position": prev_entry.get("position") if prev_entry else None,
                        "prev_finish_season": prev_entry.get("season") if prev_entry else None,
                        "prev_finish_round": prev_entry.get("round") if prev_entry else None,
                    }
                    records.append(record)

                    last_finish[driver_number] = {
                        "season": season,
                        "round": round_number,
                        "position": int(driver_row.get("Position")),
                    }

        dataset = pd.DataFrame.from_records(records)
        dataset.sort_values(["season", "round", "driver_number"], inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        self._dataset = dataset
        self._last_finish_lookup = last_finish
        return dataset

    # ------------------------------------------------------------------
    # Model training & evaluation

    def train(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainingMetrics:
        if dataframe is None:
            if self._dataset is None:
                raise RuntimeError("No dataset available; call build_dataset first")
            dataframe = self._dataset

        train_df = dataframe.dropna(subset=["prev_finish_position"]).copy()
        if train_df.empty:
            raise RuntimeError("Dataset does not contain rows with previous finishes")

        X = train_df[["prev_finish_position"]]
        y = train_df["finish_position"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)

        test_pred = self.model.predict(X_test)
        mae = float(mean_absolute_error(y_test, test_pred))
        r2 = float(r2_score(y_test, test_pred))
        baseline_mae = float(mean_absolute_error(y_test, X_test["prev_finish_position"]))

        # Refit on the full dataset for downstream predictions.
        self.model.fit(X, y)
        self._trained = True

        return TrainingMetrics(
            mae=mae,
            r2=r2,
            baseline_mae=baseline_mae,
            samples=int(len(train_df)),
            test_samples=int(len(X_test)),
        )

    # ------------------------------------------------------------------
    # Prediction helpers

    def predict_event(
        self,
        *,
        season: Optional[int] = None,
        event: Optional[str | int] = None,
    ) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Model has not been trained; call train() first")

        target_season = season or datetime.now(timezone.utc).year
        round_number, event_row, event_info = resolve_event(target_season, event)

        prev_season, prev_round = find_previous_race(target_season, round_number)
        prev_session = fastf1.get_session(prev_season, prev_round, "R")
        prev_session.load(laps=False, telemetry=False, weather=False, messages=False)
        prev_results = prev_session.results.copy()

        prev_results = prev_results.loc[prev_results["Position"].notna()].copy()
        prev_results["Position"] = pd.to_numeric(prev_results["Position"], errors="coerce")
        prev_results = prev_results.loc[prev_results["Position"].notna()].copy()
        prev_results["Position"] = prev_results["Position"].astype(int)

        if prev_results.empty:
            raise RuntimeError(
                "Previous race results are empty; cannot generate prediction dataset"
            )

        data = pd.DataFrame({
            "driver_number": prev_results["DriverNumber"].astype(str),
            "driver": prev_results["FullName"].fillna(prev_results["BroadcastName"]),
            "code": prev_results["Abbreviation"],
            "team": prev_results["TeamName"],
            "prev_finish_position": prev_results["Position"],
            "prev_status": prev_results["Status"],
        })

        predictions = self.model.predict(data[["prev_finish_position"]])
        data["predicted_finish"] = predictions
        data.sort_values("predicted_finish", inplace=True)
        data.reset_index(drop=True, inplace=True)
        data["predicted_position"] = np.arange(1, len(data) + 1)
        data["baseline_position"] = data["prev_finish_position"].astype(int)
        data["season"] = target_season
        data["event"] = getattr(event_info, "EventName", event_row.get("EventName", "Unknown"))
        data["round"] = round_number
        data["prev_race_season"] = prev_season
        data["prev_race_round"] = prev_round
        data["prev_race_name"] = getattr(
            fastf1.get_event(prev_season, prev_round), "EventName", f"Round {prev_round}"
        )

        return data[
            [
                "predicted_position",
                "driver_number",
                "code",
                "driver",
                "team",
                "predicted_finish",
                "baseline_position",
                "prev_finish_position",
                "prev_status",
                "season",
                "event",
                "round",
                "prev_race_season",
                "prev_race_round",
                "prev_race_name",
            ]
        ]


def format_prediction_table(df: pd.DataFrame, *, top: Optional[int] = None) -> pd.DataFrame:
    table = df.copy()
    if top is not None:
        table = table.head(top)

    numeric_cols = ["predicted_finish"]
    for column in numeric_cols:
        if column in table:
            table[column] = table[column].map(lambda v: f"{v:.2f}")

    return table[
        [
            "predicted_position",
            "code",
            "driver",
            "team",
            "predicted_finish",
            "baseline_position",
            "prev_finish_position",
            "prev_status",
        ]
    ]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Formula 1 race outcome predictor that relies solely on the "
            "previous finishing position and forecast the next race."
        )
    )
    parser.add_argument("--start-season", type=int, default=2022, help="First season to include")
    parser.add_argument(
        "--end-season",
        type=int,
        help="Last season to include (default: current season)",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Season to predict (default: current season)",
    )
    parser.add_argument(
        "--event",
        help="Event name or round number to predict (default: upcoming event)",
    )
    parser.add_argument("--cache-dir", default="cache", help="FastF1 cache directory")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of training data reserved for validation",
    )
    parser.add_argument(
        "--export-dataset",
        type=Path,
        help="Optional path to export the assembled training dataset (CSV)",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Only display the top-N predicted finishers",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    predictor = PreviousFinishPredictor(cache_dir=args.cache_dir, verbose=args.verbose)

    log.info(
        "Building dataset from seasons %s-%s",
        args.start_season,
        args.end_season or datetime.now(timezone.utc).year,
    )
    dataset = predictor.build_dataset(start_season=args.start_season, end_season=args.end_season)
    log.info("Collected %s driver-race samples", len(dataset))

    if args.export_dataset:
        dataset.to_csv(args.export_dataset, index=False)
        log.info("Exported dataset to %s", args.export_dataset)

    metrics = predictor.train(dataset, test_size=args.test_size)
    log.info(
        "Validation MAE: %.3f | Baseline MAE (prev position): %.3f | R^2: %.3f | Samples: %d",
        metrics.mae,
        metrics.baseline_mae,
        metrics.r2,
        metrics.samples,
    )

    prediction_table = predictor.predict_event(season=args.season, event=args.event)
    printable = format_prediction_table(prediction_table, top=args.top)

    target_event = prediction_table["event"].iloc[0]
    log.info(
        "Predicted finishing order for %s (Round %s)",
        target_event,
        prediction_table["round"].iloc[0],
    )
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
