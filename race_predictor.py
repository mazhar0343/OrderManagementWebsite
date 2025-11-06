#!/usr/bin/env python3
"""Predict the upcoming Formula 1 race using FastF1 session data.

This single-file script identifies the next race on the current (or specified)
season calendar, gathers any sessions that have already taken place (practice,
qualifying, sprint), and applies a simple pace-based heuristic to forecast the
finishing order for the race.

Example:

    python race_predictor.py            # predicts next race for the current year
    python race_predictor.py --season 2024  # overrides the season year
    python race_predictor.py --event "São Paulo Grand Prix"  # manual event selection

The predictor requires an internet connection the first time it downloads data
for a given session. Subsequent runs reuse the cached data stored in ``./cache``
by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

import fastf1
import numpy as np
import pandas as pd


SessionLabel = str


# Canonical session weights keyed by normalised session names (upper, no spaces).
DEFAULT_SESSION_WEIGHTS: Mapping[str, float] = {
    "PRACTICE1": 0.12,
    "PRACTICE2": 0.18,
    "PRACTICE3": 0.12,
    "QUALIFYING": 0.34,
    "SPRINTSHOOTOUT": 0.12,
    "SPRINT": 0.12,
}


# Aliases to translate user-provided or schedule-derived session labels into the
# strings understood by FastF1.
SESSION_ALIASES: Mapping[str, SessionLabel] = {
    "P1": "Practice 1",
    "PRACTICE1": "Practice 1",
    "FP1": "Practice 1",
    "P2": "Practice 2",
    "PRACTICE2": "Practice 2",
    "FP2": "Practice 2",
    "P3": "Practice 3",
    "PRACTICE3": "Practice 3",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "QUALI": "Qualifying",
    "QUALIFYING": "Qualifying",
    "SPRINTSHOOTOUT": "Sprint Shootout",
    "SPRINT SHOOTOUT": "Sprint Shootout",
    "SQ": "Sprint Shootout",
    "SPRINTQUALIFYING": "Sprint Shootout",
    "SPRINT QUALIFYING": "Sprint Shootout",
    "SPRINT": "Sprint",
}


DEFAULT_SESSION_ORDER: Tuple[SessionLabel, ...] = (
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Qualifying",
    "Sprint Shootout",
    "Sprint",
)


@dataclass
class PredictionResult:
    """Container for prediction output."""

    season: int
    event: str
    event_round: int
    sessions_used: Tuple[SessionLabel, ...]
    generated_at: datetime
    table: pd.DataFrame

    def as_dataframe(self) -> pd.DataFrame:
        return self.table.copy(deep=True)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "season": self.season,
            "event": self.event,
            "round": self.event_round,
            "sessions_used": list(self.sessions_used),
            "generated_at": self.generated_at.isoformat(),
            "predictions": self.table.to_dict(orient="records"),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def save_json(self, output: Path, *, indent: Optional[int] = 2) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(indent=indent), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


class FastF1RacePredictor:
    """Predict race finishing order using completed session lap data."""

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        *,
        session_weights: Optional[Mapping[str, float]] = None,
        min_laps: int = 5,
        verbose: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

        self.session_weights: Dict[str, float] = self._normalise_weights(
            session_weights or DEFAULT_SESSION_WEIGHTS
        )
        self.min_laps = min_laps
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API

    def predict(
        self,
        season: int,
        event: str | int,
        *,
        sessions: Optional[Sequence[SessionLabel]] = None,
    ) -> PredictionResult:
        target_sessions = self._resolve_sessions(sessions)

        aggregated: List[Tuple[SessionLabel, pd.DataFrame]] = []
        driver_metadata: Dict[str, Dict[str, object]] = {}
        sessions_used: List[SessionLabel] = []

        for label in target_sessions:
            try:
                session = fastf1.get_session(season, event, label)
                session.load(
                    laps=True,
                    telemetry=False,
                    weather=False,
                    messages=False,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                if self.verbose:
                    warnings.warn(
                        f"Skipping session '{label}' due to load error: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue

            summary = self._summarise_session(session)
            if summary.empty:
                if self.verbose:
                    warnings.warn(
                        f"No usable lap data found in session '{label}'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue

            aggregated.append((label, summary))
            sessions_used.append(label)
            driver_metadata.update(self._extract_driver_metadata(session))

        if not aggregated:
            raise RuntimeError(
                "Unable to build prediction: no session data was loaded successfully."
            )

        feature_table = self._build_feature_table(aggregated)
        enriched = self._attach_driver_metadata(feature_table, driver_metadata)
        scored = self._score_predictions(enriched, sessions_used)

        event_info = fastf1.get_event(season, event)
        event_name = getattr(event_info, "EventName", str(event))
        round_number = int(getattr(event_info, "RoundNumber", -1))

        return PredictionResult(
            season=season,
            event=event_name,
            event_round=round_number,
            sessions_used=tuple(sessions_used),
            generated_at=datetime.now(timezone.utc),
            table=scored,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _canonical(self, label: str) -> str:
        return label.upper().replace(" ", "").replace("-", "")

    def _resolve_sessions(
        self, sessions: Optional[Sequence[SessionLabel]]
    ) -> Tuple[SessionLabel, ...]:
        if sessions:
            resolved = [self._alias(label) for label in sessions]
        else:
            resolved = [
                label
                for label in DEFAULT_SESSION_ORDER
                if self.session_weights.get(self._canonical(label), 0) > 0
            ]
        # Preserve order while removing duplicates
        seen: set[str] = set()
        ordered: List[SessionLabel] = []
        for label in resolved:
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        return tuple(ordered)

    def _alias(self, label: SessionLabel) -> SessionLabel:
        canonical = self._canonical(label)
        return SESSION_ALIASES.get(canonical, label)

    def _normalise_weights(
        self, weights: Mapping[str, float]
    ) -> Dict[str, float]:
        positive = {
            self._canonical(label): float(value)
            for label, value in weights.items()
            if float(value) > 0
        }
        total = sum(positive.values())
        if total <= 0:
            raise ValueError("Session weights must contain at least one positive entry")
        return {label: value / total for label, value in positive.items()}

    def _summarise_session(self, session: fastf1.core.Session) -> pd.DataFrame:
        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame()

        laps = laps.copy()
        laps = laps.loc[laps["LapTime"].notna()].copy()
        if laps.empty:
            return pd.DataFrame()

        laps["lap_seconds"] = laps["LapTime"].dt.total_seconds()

        accurate = laps.loc[laps["IsAccurate"].fillna(True)].copy()
        if accurate.empty:
            accurate = laps

        grouped = accurate.groupby("DriverNumber", observed=True)
        summary = grouped.agg(
            best_lap_seconds=("lap_seconds", "min"),
            median_lap_seconds=("lap_seconds", "median"),
            mean_lap_seconds=("lap_seconds", "mean"),
            lap_count=("lap_seconds", "count"),
        )

        long_run = accurate.loc[accurate["TyreLife"].fillna(0) >= 5]
        if long_run.empty:
            long_run = accurate
        long_summary = long_run.groupby("DriverNumber", observed=True)["lap_seconds"].median()
        summary = summary.merge(
            long_summary.rename("long_run_seconds"),
            left_index=True,
            right_index=True,
            how="left",
        )

        summary.reset_index(inplace=True)
        summary.rename(columns={"DriverNumber": "driver_number"}, inplace=True)
        summary["driver_number"] = summary["driver_number"].astype(str)
        summary["low_lap_warning"] = summary["lap_count"] < self.min_laps

        return summary

    def _extract_driver_metadata(
        self, session: fastf1.core.Session
    ) -> Dict[str, Dict[str, object]]:
        metadata: Dict[str, Dict[str, object]] = {}

        for driver_number in session.drivers or []:
            try:
                info = session.get_driver(driver_number)
            except Exception:  # pragma: no cover - defensive
                continue

            if info is None:
                continue

            number = str(info.get("DriverNumber", driver_number))
            full_name = (
                info.get("FullName")
                or " ".join(filter(None, [info.get("FirstName"), info.get("LastName")])).strip()
                or info.get("BroadcastName", "").strip()
            )
            code = (info.get("Abbreviation") or info.get("BroadcastName", " ")[:3]).strip()
            team = info.get("TeamName") or info.get("Team") or ""

            metadata[number] = {
                "driver_number": number,
                "driver": full_name or f"Driver {number}",
                "code": code or number,
                "team": team,
            }

        return metadata

    def _build_feature_table(
        self, aggregated: Sequence[Tuple[SessionLabel, pd.DataFrame]]
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for label, frame in aggregated:
            renamed = frame.rename(
                columns={
                    column: column if column == "driver_number" else f"{label}_{column}"
                    for column in frame.columns
                }
            )
            frames.append(renamed)

        base = frames[0]
        for frame in frames[1:]:
            base = base.merge(frame, on="driver_number", how="outer")

        base["driver_number"] = base["driver_number"].astype(str)
        return base

    def _attach_driver_metadata(
        self,
        features: pd.DataFrame,
        metadata: Mapping[str, Mapping[str, object]],
    ) -> pd.DataFrame:
        if not metadata:
            features = features.copy()
            features["driver"] = "Unknown"
            features["code"] = features["driver_number"]
            features["team"] = ""
            return features

        meta_frame = pd.DataFrame.from_dict(metadata, orient="index")
        meta_frame.reset_index(drop=True, inplace=True)
        meta_frame["driver_number"] = meta_frame["driver_number"].astype(str)

        merged = features.merge(meta_frame, on="driver_number", how="left")
        merged["driver"] = merged["driver"].fillna("Unknown")
        merged["code"] = merged["code"].fillna(merged["driver_number"])
        merged["team"] = merged["team"].fillna("")
        return merged

    def _score_predictions(
        self,
        df: pd.DataFrame,
        sessions_used: Sequence[SessionLabel],
    ) -> pd.DataFrame:
        scored = df.copy()
        scored["score"] = 0.0

        for label in sessions_used:
            weight = self.session_weights.get(self._canonical(label), 0.0)
            if weight <= 0:
                continue

            best_col = f"{label}_best_lap_seconds"
            median_col = f"{label}_median_lap_seconds"
            long_col = f"{label}_long_run_seconds"
            laps_col = f"{label}_lap_count"
            low_lap_col = f"{label}_low_lap_warning"

            best_component = np.zeros(len(scored))
            if best_col in scored:
                best_series = scored[best_col]
                available = best_series.dropna()
                if not available.empty:
                    fastest = available.min()
                    best_component = np.where(
                        best_series.notna(),
                        fastest / best_series,
                        0.0,
                    )

            median_component = np.zeros(len(scored))
            if median_col in scored:
                median_series = scored[median_col]
                available = median_series.dropna()
                if not available.empty:
                    baseline = available.min()
                    median_component = np.where(
                        median_series.notna(),
                        baseline / median_series,
                        0.0,
                    )

            combined = 0.6 * best_component + 0.4 * median_component

            if long_col in scored:
                long_series = scored[long_col]
                available = long_series.dropna()
                if not available.empty:
                    base_long = available.min()
                    long_component = np.where(
                        long_series.notna(),
                        base_long / long_series,
                        0.0,
                    )
                    combined = 0.7 * combined + 0.3 * long_component

            contribution = weight * combined
            scored[f"{label}_pace_score"] = contribution
            scored["score"] += contribution

            if laps_col in scored:
                laps_series = scored[laps_col]
                available = laps_series.dropna()
                if not available.empty:
                    lap_bonus = np.where(
                        laps_series.notna(),
                        laps_series / available.max(),
                        0.0,
                    )
                    bonus = weight * 0.05 * lap_bonus
                    scored[f"{label}_lap_bonus"] = bonus
                    scored["score"] += bonus

            if low_lap_col in scored:
                low_lap_penalty = np.where(
                    scored[low_lap_col].fillna(False),
                    weight * 0.03,
                    0.0,
                )
                scored[f"{label}_lap_penalty"] = low_lap_penalty
                scored["score"] -= low_lap_penalty

        max_score = scored["score"].max()
        min_score = scored["score"].min()
        if np.isclose(max_score, min_score):
            scored["confidence"] = 1.0
            scored["predicted_gap_seconds"] = 0.0
        else:
            scored["confidence"] = (scored["score"] - min_score) / (max_score - min_score)
            scored["predicted_gap_seconds"] = (
                (max_score - scored["score"]) / max_score * 20.0
            )

        scored["confidence"] = scored["confidence"].clip(0, 1)

        def missing_sessions(row: pd.Series) -> List[str]:
            missing: List[str] = []
            for label in sessions_used:
                column = f"{label}_best_lap_seconds"
                if column not in row or pd.isna(row[column]):
                    missing.append(label)
            return missing

        scored["missing_sessions"] = scored.apply(missing_sessions, axis=1)

        scored.sort_values(by=["score", "driver_number"], ascending=[False, True], inplace=True)
        scored.reset_index(drop=True, inplace=True)
        scored["predicted_position"] = np.arange(1, len(scored) + 1)

        cols = [
            "predicted_position",
            "driver_number",
            "code",
            "driver",
            "team",
            "score",
            "confidence",
            "predicted_gap_seconds",
            "missing_sessions",
        ]
        metric_columns = [column for column in scored.columns if column not in cols]
        ordered_columns = cols + metric_columns
        return scored.loc[:, ordered_columns]


# ----------------------------------------------------------------------
# Event selection utilities


def find_upcoming_event(season: int) -> Tuple[int, pd.Series]:
    schedule = fastf1.get_event_schedule(season)
    schedule = schedule[schedule["EventFormat"] != "testing"]
    utc_now = pd.Timestamp.utcnow()
    today = utc_now.tz_convert("UTC").normalize().tz_localize(None)

    upcoming = schedule.loc[schedule["EventDate"] >= today]
    if upcoming.empty:
        upcoming = schedule.tail(1)

    event_row = upcoming.iloc[0]
    return int(event_row["RoundNumber"]), event_row


def completed_sessions_for_event(event_row: pd.Series, *, reference_time: pd.Timestamp) -> List[str]:
    sessions: List[str] = []

    for idx in range(1, 6):
        name = event_row.get(f"Session{idx}")
        if not isinstance(name, str):
            continue
        lower = name.lower()
        if "race" in lower or "testing" in lower:
            continue

        session_time = event_row.get(f"Session{idx}DateUtc")
        if pd.isna(session_time):
            session_time = event_row.get(f"Session{idx}Date")

        if pd.isna(session_time):
            sessions.append(name)
            continue

        timestamp = pd.Timestamp(session_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        if timestamp <= reference_time:
            sessions.append(name)

    return sessions


# ----------------------------------------------------------------------
# CLI helpers


def _parse_event(value: str) -> str | int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


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
    parser = argparse.ArgumentParser(
        description=(
            "Predict the upcoming Formula 1 race using timing data from completed sessions."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Target season year (default: current year).",
    )
    parser.add_argument(
        "--event",
        help="Override the event (name or round number).",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="Explicit session labels to use (defaults to completed sessions for the event).",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory for FastF1 cache (default: ./cache).",
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
        help="Limit the number of entries displayed.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Write the full prediction output to the specified JSON file.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Write the tabular predictions to the specified CSV file.",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Include diagnostic metric columns in the printed table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings about sessions that could not be processed.",
    )

    args = parser.parse_args(argv)

    now_utc = pd.Timestamp.utcnow()
    season = args.season or datetime.now(timezone.utc).year

    if args.event:
        event_identifier = _parse_event(args.event)
        event_info = fastf1.get_event(season, event_identifier)
        if event_info is None:
            raise SystemExit("Could not resolve the specified event.")
        event_row = fastf1.get_event_schedule(season).loc[
            lambda df: df["RoundNumber"] == getattr(event_info, "RoundNumber", -1)
        ]
        if event_row.empty:
            event_row = pd.Series({
                "EventName": getattr(event_info, "EventName", str(event_identifier)),
                "RoundNumber": getattr(event_info, "RoundNumber", -1),
            })
        else:
            event_row = event_row.iloc[0]
    else:
        round_number, event_row = find_upcoming_event(season)
        event_identifier = round_number
        event_info = fastf1.get_event(season, round_number)

    default_sessions: Optional[List[str]] = None
    if isinstance(event_row, pd.Series):
        default_sessions = completed_sessions_for_event(event_row, reference_time=now_utc)

    sessions = args.sessions or default_sessions
    if not sessions:
        raise SystemExit(
            "No completed sessions found for this event yet; try again after practice or qualifying."
        )

    predictor = FastF1RacePredictor(
        cache_dir=args.cache_dir,
        min_laps=args.min_laps,
        verbose=args.verbose,
    )

    result = predictor.predict(
        season,
        event_identifier,
        sessions=sessions,
    )

    print(
        f"Predicted finishing order for {result.event} (Round {result.event_round}) "
        f"— sessions used: {', '.join(result.sessions_used)}"
    )

    table = _format_table(result, top=args.top, show_metrics=args.show_metrics)
    _print_table(table)

    if args.export_json:
        result.save_json(args.export_json)
        print(f"Saved JSON output to {args.export_json}")

    if args.export_csv:
        result.table.to_csv(args.export_csv, index=False)
        print(f"Saved CSV output to {args.export_csv}")


if __name__ == "__main__":
    main()
