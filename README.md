# FastF1 Race Predictor (Single Script)

This repository now ships a single Python file, `race_predictor.py`, that uses
[FastF1](https://theoehrly.github.io/Fast-F1/) timing data to forecast the
finishing order of the next Formula 1 race on the calendar. The script locates
the upcoming event (e.g. Brazil/São Paulo) for the current season, collects all
completed sessions (practice, qualifying, sprint), and scores each driver based
on their pace and consistency.

## Prerequisites

- Python 3.10 or later
- Packages: `fastf1`, `pandas`, `numpy`

Install the dependencies in your environment of choice:

```bash
python3 -m pip install fastf1 pandas numpy
```

FastF1 keeps a cache of the downloaded telemetry. By default the script uses
`./cache`; you can change this with `--cache-dir`.

## Usage

Run the predictor with no arguments to target the next race automatically:

```bash
python race_predictor.py
```

Typical options:

- `--season 2024` – override the season year
- `--event "São Paulo Grand Prix"` – force a specific event (name or round number)
- `--top 10` – limit the number of rows printed
- `--show-metrics` – include diagnostic columns (session scores, lap bonuses, etc.)
- `--export-json predictions.json` / `--export-csv predictions.csv` – persist outputs

Example (predict the upcoming Brazil round, assuming practice and qualifying have
already run):

```bash
python race_predictor.py --event "São Paulo Grand Prix" --top 10
```

## How It Works

1. Determine the next (or specified) race via `fastf1.get_event_schedule`.
2. For each completed session, fetch lap data and compute pace metrics:
   - Best, median, and long-run lap times
   - Lap counts and low-running penalties
3. Weight and combine the metrics to produce a ranked finishing order with
   confidence estimates and predicted time gaps.

If no sessions have been completed yet, the script will notify you to rerun it
once practice or qualifying results are available.

## Troubleshooting

- **Slow first run** – the script downloads several datasets per session. The
  cache makes subsequent executions much faster.
- **Session missing** – if a practice or sprint has not yet been staged, it is
  skipped automatically. You can also pass `--sessions` to specify the data you
  want to include.
- **Event cannot be found** – ensure the season/year is correct or pass the
  round number with `--event 20`.

## License

This project is released under the MIT License.
