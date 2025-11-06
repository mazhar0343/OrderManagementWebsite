# FastF1 Race Predictor

A lightweight Formula 1 race outcome predictor that leverages the
[FastF1](https://theoehrly.github.io/Fast-F1/) timing API to analyse practice and
qualifying pace. The predictor aggregates lap data from selected sessions,
applies a configurable weighting scheme, and produces an estimated finishing
order for the upcoming race.

## Features

- Pulls detailed session telemetry via FastF1 with local caching for repeated use
- Computes per-driver pace metrics (best/median/long-run laps, lap volume)
- Generates a ranked prediction table with confidence scores and gap estimates
- Offers a command-line interface with JSON/CSV export options

## Getting Started

### Prerequisites

- Python 3.10 or later
- An active internet connection (FastF1 downloads timing data on demand)

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install exposes the `fastf1-race-predictor` console entrypoint and
installs required dependencies such as `fastf1`, `pandas`, and `numpy`.

### Cache Configuration

FastF1 stores downloaded data in a cache directory so subsequent runs are much
faster. By default, this project uses `./cache`; you can change this via the CLI
or environment variables.

## Usage

Predict the 2024 Bahrain Grand Prix after qualifying data is available:

```bash
fastf1-race-predictor --season 2024 --event Bahrain --sessions FP1 FP2 Q --top 10
```

The same command can be executed without installation from the repository root:

```bash
python -m fastf1_race_predictor.cli --season 2024 --event 1 --sessions FP1 FP2 FP3 Q
```

Useful flags:

- `--top`: limit the number of rows printed
- `--show-metrics`: include diagnostic columns such as missing session data
- `--export-json predictions.json`: write the full result to disk
- `--export-csv predictions.csv`: export the table for further analysis

## How the Predictor Works

1. FastF1 loads the requested sessions (typically FP1–FP3 and Qualifying).
2. For each driver, the predictor computes:
   - Best, median, and long-run lap times (based on accurate laps)
   - Lap counts to gauge reliability
   - Flags for low running that trigger penalties
3. Weighted pace scores from each session are combined into an overall driver
   score, producing a ranked finishing order with confidence values.

The current implementation focuses on pace-based heuristics. It does not yet
incorporate weather forecasts, tyre choices during the race, or safety car
probabilities. Those can be added by extending `FastF1RacePredictor`.

## Troubleshooting

- **Slow or failing requests**: the first run against a session downloads
  several megabytes of timing data. Ensure your connection is stable; rerunning
  the command will reuse cached data.
- **Session not yet available**: FastF1 can only load sessions that the FIA has
  published. If you run the predictor before an event has finished qualifying,
  skip unavailable sessions via `--sessions`.
- **API rate limits**: heavy automated usage may trigger remote rate limits.
  Consider caching at the season level and spacing out requests.

## Development

Install optional tooling:

```bash
pip install -e .[dev]
```

Run formatting and quality checks:

```bash
ruff check src
pytest
```

## License

This project is released under the MIT License.
