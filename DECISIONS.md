# Decision log

Short record of what was decided, why, and what evidence it rests on. Dates are 2026-10-15 unless noted.

## Scope and target
- **Kyiv City only.** One clean, official, up-to-date series beats 24 noisy ones under a 1.5-day deadline.
- **Target: does a new alert episode start in (t, t+3h]**, forecast issued at every hour t.
  - 6h was rejected: in the last 28 days its base rate is 0.90, nothing left to predict.
  - "Is an alert active at t+1" was rejected: persistence solves it trivially.
  - A start exactly at t counts as past (known at issue time), not as the label.

## Data source
- **Kyiv Digital `/air-alert/history`** (raw JSON, sha256-pinned snapshot from 2026-10-15).
- Considered and rejected as model inputs:
  - *Vadimkin official CSV* — the same feed (99.7% of Kyiv starts match within 5 min, 2 s apart), but it stops on 2026-09-07 (repo issues #6, #7). It also has 38% exact duplicate rows and mixes hromada-level records.
  - *Vadimkin volunteer CSV* — current, but for Kyiv only 71–74% of starts match the official feed since 2023, and 25–38% after 2026-09-07. 220 Kyiv end times are imputed (+30 min).
- **Time zone:** `created_at` has no offset. Matching starts against UTC sources: Europe/Kyiv with DST matches 66–67% in both summer and winter; fixed UTC+2/+3 fail in one season each, UTC fails always. The repeated autumn hour is resolved by keeping the log ordered (one event affected).

## Cleaning
- Data from **2023-01-01**: all 53 sequence anomalies (47 repeated starts, 6 repeated ends) are in 2022. Since 2023 starts and ends alternate perfectly, so no end-time repair is needed. 2022 events are still used as feature history for early 2023.
- Restarts within 5 min of an all-clear: only 18 since 2023. Merging them changes base rates by < 0.001, so we don't merge.

## The September 2026 change
- From 2026-09-06 06:00 Kyiv uses yellow (drone) / red (missile, massive drone) levels (kyivcity.gov.ua, 6 and 8 Sep). An episode still runs from the first threat until a single all-clear, and there is no all-clear between threat types. **The episode definition is unchanged**, so the target means the same before and after.
- The alert rate jumped on **2026-08-24, two weeks before the new format** (1.7 → 7.6 starts/day). That is real escalation, not an artifact of the new levels.
- **`causes` is never a model input.** It exists only from 2026-09-07, and start records carry several threat types at once (e.g. drone+missile+massive-drone). Most likely they were filled in after the start. Allowed only for slicing evaluation after the fact.

## Evaluation protocol
- Train: 2023-01-01 → 2026-08-23. Validation: 2026-08-24 → 2026-09-16 (includes the escalation). Test: 2026-09-17 → 2026-10-14, looked at once.
- Forecast rows whose 3h label window crosses a split boundary are purged (4 rows).
- Probabilistic metrics: Brier, log loss, ROC-AUC, PR-AUC, and a Brier skill score against the period's own base rate (a hindsight constant). Reported for all hours and for calm hours (no alert active at t).

## Leakage controls (tests/test_leakage.py)
- Features recomputed from the log truncated at t must equal the features from the full log (300+ sampled times).
- Adding future events must not change features at t.
- Baselines fitted on train must not change when val/test labels are scrambled.
- Mutation check: a planted leak (current alert duration from its real end) makes 3 of these tests fail.

## Baseline findings (validation)
- Base rate: train 0.148 → val 0.568 → test 0.701. Anything fitted to the training *level* fails: train base rate BSS −0.72, hour-of-day climatology −0.73.
- Best baseline: **Poisson rate from starts in the last 24h** — BSS +0.15 (all hours) / +0.19 (calm), ROC-AUC 0.75 / 0.78. This is the bar a model must clear.
