# FastF1 Race Predictor (Single Script)

This repository ships two complementary Python scripts powered by
[FastF1](https://theoehrly.github.io/Fast-F1/):

- `race_predictor.py` – forecasts finishing order using session pace metrics
  (practice, qualifying, sprint).
- `previous_finish_predictor.py` – trains a lightweight ML model that relies
  solely on each driver's finishing position in the previous race (2022 → now).

## Prerequisites

- Python 3.10 or later
- Packages: `fastf1`, `pandas`, `numpy`, `scikit-learn`

Install the dependencies in your environment of choice:

```bash
python3 -m pip install fastf1 pandas numpy scikit-learn
```

FastF1 keeps a cache of the downloaded telemetry. By default the script uses
`./cache`; you can change this with `--cache-dir`.

## Usage – Session Pace Predictor

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

## Usage – Previous Finish Predictor

Run the model builder + predictor without arguments to gather race results from
2022 onwards, train a random forest regressor, and forecast the next race on the
calendar:

```bash
python previous_finish_predictor.py
```

Key options:

- `--start-season 2022` / `--end-season 2024` – control the historical window
- `--season 2025 --event 3` – target a specific season and round (name or number)
- `--export-dataset data.csv` – persist the assembled training dataset
- `--top 10` – limit the printed leaderboard

The script prints dataset statistics, validation metrics (MAE, R² vs a baseline
that simply repeats the previous finishing position), and the predicted finishing
order for the requested event.

## How It Works – Session Pace Predictor

1. Determine the next (or specified) race via `fastf1.get_event_schedule`.
2. For each completed session, fetch lap data and compute pace metrics:
   - Best, median, and long-run lap times
   - Lap counts and low-running penalties
3. Weight and combine the metrics to produce a ranked finishing order with
   confidence estimates and predicted time gaps.

If no sessions have been completed yet, the script will notify you to rerun it
once practice or qualifying results are available.

## How It Works – Previous Finish Predictor

1. Download race classification results for the requested seasons via FastF1.
2. Assemble a per-driver dataset linking the previous race's finishing position
   to the current race's finishing position (dropping entries without history).
3. Train a `RandomForestRegressor` and benchmark it against the trivial baseline
   that simply repeats the previous result.
4. Gather results from the most recent completed race to generate features for
   the target event and rank drivers by the model's predicted finishing position.

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
