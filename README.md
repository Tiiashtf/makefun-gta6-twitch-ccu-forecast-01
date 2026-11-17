[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Tiiashtf/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Tiiashtf/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — hour-by-hour forecast

A forecaster for `gta6-twitch-ccu-forecast-01`: given a case file with hour 1's
Twitch concurrent-viewer print for the Grand Theft Auto VI category, it
forecasts hours 2–24 of release day (2026-11-19 UTC) as five quantiles each
(5%, 25%, 50%, 75%, 95%).

## Model

An empirical launch-shape model learned from the comparables shipped in each
case file (past big launches' hourly Twitch counts, 14 days before to 3 days
after their release day):

- For every comparable launch, take the log-ratio of each hour's viewers to
  that launch's hour-1 viewers. The shape of a launch day depends strongly on
  how big hour 1 already is: a launch whose audience is mostly in by midnight
  UTC peaks at hour 1 and decays, while one whose unlocks land later in the
  day grows through the day (correlation between log hour-1 viewers and the
  hour-19 log-ratio is about −0.93 across the comparables).
- For each forecast hour, fit a local kernel-weighted linear regression of the
  log-ratio on log hour-1 viewers (Gaussian kernel, bandwidth 2.0 in log
  units). Each comparable is weighted by how good an analogue it is for a
  rolling-midnight, single-player console launch with no early access
  (rolling-midnight 1.0, midnight-like 0.6, other comparables 0.25,
  early-access-contaminated 0.25, weak 0.1).
- The spread comes from the kernel-weighted interquartile range of the
  residuals, so the quantiles are honest about how much launch days vary.
- Forecast: hour-1 print × exp(mu_h + z_tau × sigma_h). The fitted median is
  clamped to the range of ratios the comparables actually show for that hour,
  so a launch bigger than any before it cannot extrapolate the fitted slope
  past all observed behaviour.

The model's own pre-release category history (the GTA VI premiere spike and
quiet weeks) is not a launch and is left out of the fit.

## Validation

Practice cases (answers = SullyGnome's public hourly averages, checked against
each case's hour-1 print):

| case | this model | house baseline |
| --- | --- | --- |
| Wolverine (2026-09-15) | 0.164 | 0.744 |
| Red Dead Redemption 2 (2018-10-26) | 0.155 | 0.287 |
| God of War Ragnarök (2022-11-09) | 0.124 | 0.215 |
| Marvel's Spider-Man 2 (2023-10-20) | 0.202 | 0.303 |
| **mean** | **0.158** | **0.387** |

Mean pinball loss on the log scale (lower is better). 90% coverage 0.92,
typical median miss ~25%. Leave-one-out over the comparables gives 0.25 on
rolling-midnight launches against the baseline's 0.38. Nothing is tuned to a
single launch: bandwidth, weights and spread are flat across a wide range of
values, and adding hour-1 channels as a second predictor made it worse, so it
stays out.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Reads every `*.in` in the input directory
and writes one `<id>.out` per case. Deterministic (no clock, no randomness,
sorted iteration), runs in well under a second per case. It never crashes a
case: any failure to build the model falls back to a persistence forecast, and
any failure to read a case still writes a valid output file.

## Layout

- `main.py` — the forecaster.
- `LICENSE`, `README.md`, `.makefun/owner.json`.
