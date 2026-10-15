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

## First models (validation, fixed hyperparameters, trained on 2023-01-01 → 2026-08-23)
| model | BSS all / calm | ROC-AUC all / calm | mean p (base rate 0.568) |
|---|---|---|---|
| recent_rate_24h (baseline) | **+0.15 / +0.19** | **0.75 / 0.78** | 0.47 |
| logistic (log-scaled features) | −0.22 / −0.16 | 0.63 / 0.68 | 0.31 |
| LightGBM | −0.90 / −0.86 | 0.36 / 0.28 | 0.14 |
| sklearn HistGradientBoosting (cross-check) | −0.87 / −0.82 | 0.37 / 0.32 | 0.14 |

- LightGBM needs `brew install libomp` on macOS. sklearn's histogram GBDT, with the same settings, is kept as a cross-check, and it gives the same result, so this is not a LightGBM quirk.
- **Why both lose:** what changed is not only the alert *level* but the *relationship*. In training (calm era), recent activity barely predicts the next 3h: P rises from 0.12 to 0.19 across `starts_24h` bins and is flat above 2 starts. Single-feature ROC-AUC of `starts_24h` is 0.56 in train vs 0.75 in val. 240 of 574 validation hours have more than 6 starts in the last 24h; only 69 of 31,941 training hours do.
- **Boosted trees invert the ranking** (AUC < 0.5). In the sparse high-activity tail they learned "very active → lower P". LightGBM's P falls from 0.08 to 0.044 as `starts_24h` goes from 3 to 6 and then stays flat: trees cannot extrapolate beyond the training range. They predict 0.08–0.10 exactly where validation sees 0.63–0.84. Shuffling `starts_24h` *improves* LightGBM's validation AUC by 0.064.
- **Logistic is monotone and extrapolates in the right direction** (P rises 0.23 → 0.37 across the same bins), but the slope and intercept are learned from the calm era, so it is far too low. The correlated count features also split credit (7d positive, 24h negative), while 24h is the signal that matters in the escalation.
- **The 24h Poisson baseline wins because it learns nothing from the calm era.** It maps current intensity straight to a probability.

## Walk-forward retraining and the baseline feature (validation)
Protocol: before each Kyiv midnight D, refit on every row whose label is observed (t + 3h <= D), then forecast all hours of D. Tests check that flipping every label not yet observed by D leaves D's forecasts unchanged. A planted leak (`t <= D` instead of `t + 3h <= D`) fails 3 tests.

| model | BSS all / calm | ROC-AUC all / calm | mean p (base rate 0.568) |
|---|---|---|---|
| recent_rate_24h (baseline) | **+0.15 / +0.19** | **0.75 / 0.78** | 0.47 |
| logistic, static → walk-forward | −0.22 → −0.12 / −0.16 → −0.06 | 0.63 → 0.62 | 0.31 → 0.37 |
| logistic+base, static → walk-forward | −0.21 → −0.10 / −0.15 → −0.04 | 0.63 → 0.63 | 0.32 → 0.37 |
| LightGBM, static → walk-forward | −0.90 → −0.07 / −0.86 → +0.01 | 0.36 → 0.61 | 0.14 → 0.48 |
| LightGBM+base | identical to LightGBM | | |

- **Walk-forward fixes most of LightGBM's failure** (BSS −0.90 → −0.07, AUC 0.36 → 0.61). Retraining daily lets it see escalation hours, so it stops predicting 0.08 where the actual rate is 0.6–0.8. Logistic gains less, because its static version already extrapolated in the right direction.
- **Neither beats the baseline.** At the start of each 8-day validation block, escalation rows are 0%, 0.6% and 1.2% of the training set. With equal weights, three years of calm data outvote three weeks of escalation.
- **The baseline feature changes nothing for LightGBM (exactly identical predictions).** `baseline_logit` is a monotone function of `starts_24h`, and trees only use the order of a feature's values. For logistic it adds little (+0.01–0.02 BSS) because `log_starts_24h` already carries almost the same information.
- **The 24h baseline's own skill drifts down across the period:** 8-day BSS +0.36, +0.13, −0.05. Validation is short (3 × ~190 hours), so all these numbers are noisy.

## Time-decay sample weights (walk-forward, validation)
Weight = 0.5^(age / half-life), with age counted from the newest training row of each fit. Half-lives fixed in advance: **14 days**, and one alternative, **60 days**. No other values were tried.

| walk-forward model | BSS all / calm | log loss | ROC-AUC all / calm | mean p (0.568) |
|---|---|---|---|---|
| recent_rate_24h (baseline) | **+0.15 / +0.19** | 0.685 (0.611 with p floored at 0.05) | **0.75 / 0.78** | 0.47 |
| logistic, no decay | −0.12 / −0.06 | 0.744 | 0.62 / 0.68 | 0.37 |
| **logistic, decay 14d** | **+0.06 / +0.10** | 0.656 | 0.67 / 0.70 | 0.55 |
| logistic, decay 60d | +0.02 / +0.07 | 0.673 | 0.63 / 0.68 | 0.50 |
| LightGBM, no decay | −0.07 / +0.01 | 0.740 | 0.61 / 0.65 | 0.48 |
| LightGBM, decay 14d | −0.14 / −0.08 | 0.851 | 0.64 / 0.66 | 0.54 |
| LightGBM, decay 60d | −0.07 / −0.01 | 0.772 | 0.64 / 0.68 | 0.54 |

- **Decay helps logistic** (BSS −0.12 → +0.06 at 14d, +0.02 at 60d). It now predicts close to the actual level (mean p 0.55 vs 0.57). Both half-lives help, so the direction does not depend on the exact value.
- **Decay does not help LightGBM; at 14d it is worse.** A 14-day half-life leaves about 14 / ln 2 × 24 ≈ 480 hours of effective data. That is enough for logistic's 13 coefficients, but too little for 300 trees × 15 leaves, which then fit noise (log loss 0.85).
- **The apparent log-loss win over the baseline is an artifact.** The baseline says P = 0 in the 16 hours with no start in the last 24h, and 4 of them were followed by an alert. Those hours alone add 0.096 to its log loss. On the other 558 hours the baseline scores 0.606 vs logistic 0.659. Floored at 0.05, the baseline scores 0.611 overall.
- **The baseline still wins in every 8-day block:** +0.36 / +0.13 / −0.05 vs logistic-decay14 +0.18 / +0.07 / −0.08.

## Frozen before the test run (committed before `run_eval.py --test` was executed)
Test period: forecasts issued 2026-09-17 00:00 → 2026-10-14 21:00 Kyiv. It is scored once, with no changes afterwards.
- **Primary forecaster: `recent_rate_24h_floor`**, the 24h Poisson rate with P kept in [0.05, 0.95]. It is the best on validation (BSS +0.15, log loss 0.611). The 0.05 floor was set after seeing validation, as a round number, to remove exact-zero probabilities.
- **Best learned model: logistic, walk-forward, 14-day decay** (validation BSS +0.06).
- References, to show the path from the first attempt: `recent_rate_24h` without the floor; logistic and LightGBM trained once (on train+val for the test); LightGBM walk-forward without decay.
- Protocol: models trained once are fit on 2023-01-01 → 2026-09-16. Walk-forward models refit every Kyiv midnight on all labels observed by then, including earlier test days.
- Expectation written down beforehand: the test base rate is 0.70, higher than validation (0.57), and the escalation continues. The recent-rate approach should keep working. Models trained once should do badly again, though less than on validation, because train+val now includes 3 weeks of escalation.

## Test results (scored once, 2026-09-17 → 2026-10-14, 670 hours, base rate 0.701)
| model | BSS all / calm | log loss | ROC-AUC all / calm | mean p |
|---|---|---|---|---|
| recent_rate_24h_floor (**pre-registered primary**) | −0.105 / −0.250 | 0.658 | 0.54 / 0.56 | 0.59 |
| recent_rate_24h (no floor) | identical: no test hour had 0 starts in the last 24h | | | |
| logistic, trained once | −0.220 / −0.446 | 0.705 | 0.61 / 0.63 | 0.47 |
| LightGBM, trained once | −0.058 / −0.104 | 0.639 | 0.61 / 0.67 | 0.62 |
| LightGBM, walk-forward | −0.006 / −0.012 | 0.612 | 0.62 / 0.64 | 0.66 |
| **logistic, walk-forward, 14-day decay** | **+0.044 / +0.024** | **0.587** | **0.64** / 0.65 | 0.67 |

- **The pre-registered primary lost.** Validation picked the 24h baseline. On test it is worse than a constant at the period's base rate, and it barely ranks hours (AUC 0.54).
- **The best learned model won on test**, and robustly: its Brier improvement over the baseline is +0.031, with a 95% day-block bootstrap CI of [+0.016, +0.047] over 28 days. It beats the baseline in all 4 weeks. Its improvement over walk-forward LightGBM is +0.010, CI [+0.000, +0.019].
- **Why the baseline broke:** its Poisson mapping from counts to probability is fixed. In the test period alerts are almost continuous, so the count no longer separates hours: with ≤ 4 starts in 24h, 57–83% of hours were still followed by a start. It also underpredicts the level (0.59 vs 0.70), especially in calm hours (0.58 vs 0.77). In calm hours a new start is *more* likely than during an alert (0.77 vs 0.59), because a new start can only follow an all-clear. The baseline cannot represent that.
- **Why the decay model wins:** it learns from recent *outcomes*, not only counts. Every day it refits the level and the role of `active_now` / time since the last all-clear on roughly the last 2 weeks. On validation this was hidden because the escalation had just begun (0–1.2% of the training rows); by the test period it had 3+ weeks of escalation to learn from.
- **Absolute skill is small** (BSS +0.04, AUC 0.64). In this regime, forecasting 3h onsets is mostly about getting the level right; ranking individual hours remains hard.
- Post-hoc analysis above is descriptive only; no model was changed after scoring the test.
