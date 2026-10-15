# Air-raid alert forecasting — Kyiv City

KSE Agentic AI School, Stage 2. At the top of every hour, forecast the probability that
**a new air-raid alert starts in Kyiv City within the next 3 hours**, using only information
published before that moment.

Why this target and source, and what was rejected, is in [DECISIONS.md](DECISIONS.md).

## Data
`data/raw/kyiv_digital/history_2026-10-15.json` — snapshot of the official
[Kyiv Digital air-alert history API](https://kyiv.digital/open-api/air-alert/history), with start (`state=1`) and end
(`state=0`) events from Feb 2022 to 15 Oct 2026. Timestamps are Kyiv local time without an
offset; they are converted to UTC with DST handling. The loader checks the file's sha256.

## Setup and run
On macOS LightGBM needs the OpenMP runtime first: `brew install libomp`.
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest              # data, labeling and leakage tests
.venv/bin/python scripts/run_eval.py   # baselines + models on the validation period
```

## Layout
```
src/air_alerts/
  config.py     snapshot path + hash, time zone, horizon, split dates
  data.py       load events, Kyiv -> UTC, build alert episodes
  labels.py     hourly forecast grid, 3h target, train/val/test split with purging
  features.py   features computed only from events at or before t
  baselines.py  reference forecasters
  models.py     logistic regression, LightGBM, sklearn gradient boosting
  evaluate.py   Brier / log loss / ROC-AUC / PR-AUC, all hours and calm hours
scripts/run_eval.py
tests/          data, label and leakage tests
```

## Status
- [x] Data loading, labeling, leakage tests, baselines
- [x] First models (logistic regression, gradient boosting) — both lose to the 24h baseline, see DECISIONS.md
- [ ] Models that adapt to the escalation
- [ ] One-time evaluation on the test period
