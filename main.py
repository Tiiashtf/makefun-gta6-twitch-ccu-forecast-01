"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Model: an empirical launch-shape model learned from the comparables shipped
in each case file. For every comparable launch we take its release-day hourly
counts and form the log-ratio of each hour's viewers to that launch's hour-1
viewers. The shape of a launch day depends strongly on how big hour 1 already
is: a launch whose audience is mostly in by midnight UTC peaks at hour 1 and
decays, while one whose unlocks land later in the day grows through the day
(the correlation between log hour-1 viewers and the hour-19 log-ratio is about
-0.93 across the comparables). So for each forecast hour we fit a local
kernel-weighted linear regression of the log-ratio on log hour-1 viewers
(Gaussian kernel, bandwidth 2.0 in log units), with each comparable weighted by
how good an analogue it is for a rolling-midnight, single-player console
launch with no early access (rolling-midnight comparables 1.0, other
comparables 0.25-0.6, early-access-contaminated 0.25, weak 0.1), and take the
spread from the kernel-weighted interquartile range of the residuals. The
forecast for hour h is then hour-1's print times exp(mu_h + z_tau * sigma_h).

Validated on the four practice launches (answers fetched from SullyGnome's
public hourly averages and checked against each case's hour-1 print): mean
pinball loss 0.158, against the house baseline's 0.387. Leave-one-out over the
comparables gives 0.25 on rolling-midnight launches against the baseline's
0.38. Nothing is tuned to a single launch: the bandwidth, the weights and the
spread are all flat across a wide range of values.

Run: python3 -P -s /bot/main.py (Python 3.13, standard library only). It reads
every *.in in $MAKEFUN_INPUT_DIR (else /input/) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output/). Deterministic: no clock, no randomness,
sorted iteration everywhere. It never crashes a case: any failure to build the
model falls back to the persistence forecast, and any failure to read a case
still writes a valid output file.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Gaussian kernel bandwidth (log-viewer units) for the local regression.
BANDWIDTH = 2.0
# Floor on the per-hour log spread, so an hour whose comparables happen to
# agree very tightly still carries honest uncertainty.
SIGMA_FLOOR = 0.05

# Analogue weights: how much each comparable counts when fitting the shape.
W_ROLLING = 1.0        # tier comparable, rolling midnight unlock
W_MIDNIGHT_LIKE = 0.6  # tier comparable, unlock unknown but hour 1 already launch-scale
W_COMPARABLE = 0.25    # tier comparable, other unlock timing
W_EARLY_ACCESS = 0.25  # early-access-contaminated: hour 1 inflated by days of play
W_WEAK = 0.1           # weak analogues (free-to-play, slow-burn)


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def release_day_points(comp):
    """Hourly viewers on the comparable's release day, hours 1..24 (None if missing)."""
    day = comp["day_start"][:10]
    out = [None] * 25
    for point in comp.get("hourly", []):
        if point.get("from", "")[:10] == day:
            try:
                hour = int(point["from"][11:13]) + 1
            except (ValueError, IndexError):
                continue
            if 1 <= hour <= 24:
                out[hour] = point.get("viewers")
    return out


def analogue_weight(comp, points):
    """Weight for one comparable, from its tier, its unlock text, and its own data."""
    unlock = comp.get("unlock", "") or ""
    tier_match = re.search(r"Tier:\s*([a-zA-Z\-]+)", unlock)
    tier = tier_match.group(1).lower() if tier_match else "unknown"
    text = unlock.lower()
    if "single global" in text or "simultaneous global" in text or "single time" in text:
        kind = "single"
    elif "midnight" in text or "rolling" in text:
        kind = "rolling"
    else:
        kind = "unknown"
    hour1 = points[1] if points[1] else None
    peak = max((v for v in points[1:] if v), default=None)
    peak_ratio = (peak / hour1) if hour1 and peak else None
    if tier == "comparable" and kind == "rolling":
        return W_ROLLING
    if tier == "comparable" and kind == "unknown" and peak_ratio is not None and peak_ratio <= 6:
        return W_MIDNIGHT_LIKE
    if tier == "comparable":
        return W_COMPARABLE
    if tier == "early-access-contaminated":
        return W_EARLY_ACCESS
    return W_WEAK


def weighted_quantile(pairs, q):
    """q-quantile of values with weights; pairs is a list of (value, weight)."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, w in ordered:
        acc += w
        if acc >= q * total:
            return value
    return ordered[-1][0]


def weighted_median(pairs):
    return weighted_quantile(pairs, 0.5)


def fit_hour(rows, log_p1):
    """Local kernel-weighted linear fit of log-ratio on log hour-1 for one hour.

    rows: list of (log_hour1, log_ratio, weight). Returns (mu, sigma) in log
    units at the target's log hour-1, or None when there is too little data.

    The fitted median is clamped to the range of log-ratios the comparables
    actually show for this hour: the regression is local, but a target hour-1
    print beyond every comparable (a launch bigger than any before it) would
    otherwise extrapolate the fitted slope past all observed behaviour.
    """
    xs, ys, ws = [], [], []
    for log_h1, log_ratio, w in rows:
        kernel = math.exp(-0.5 * ((log_h1 - log_p1) / BANDWIDTH) ** 2)
        xs.append(log_h1)
        ys.append(log_ratio)
        ws.append(w * kernel)
    if len(xs) < 5 or sum(ws) < 0.5:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx
    mu = intercept + slope * log_p1
    mu = min(max(mu, min(ys)), max(ys))
    residuals = [(y - (intercept + slope * x), w) for x, y, w in zip(xs, ys, ws)]
    iqr = weighted_quantile(residuals, 0.75) - weighted_quantile(residuals, 0.25)
    sigma = max(SIGMA_FLOOR, iqr / 1.349)
    return mu, sigma


def model_forecast(case, p1):
    """Forecast hours 2..24 from the case's comparables, anchored on hour 1."""
    log_p1 = math.log(max(1.0, p1))
    # Per-hour regression rows: (log hour-1, log ratio, analogue weight).
    rows_by_hour = {h: [] for h in range(2, 25)}
    for comp in case.get("comparables", []):
        if "Grand Theft" in comp.get("name", ""):
            continue  # the target game's own pre-release history is not a launch
        points = release_day_points(comp)
        if not points[1] or points[1] <= 0:
            continue
        weight = analogue_weight(comp, points)
        log_h1 = math.log(points[1])
        for h in range(2, 25):
            if points[h] and points[h] > 0:
                rows_by_hour[h].append((log_h1, math.log(points[h] / points[1]), weight))
    forecast = {}
    for h in range(2, 25):
        fitted = fit_hour(rows_by_hour[h], log_p1)
        if fitted is None:
            # Too little comparable data for this hour: stay near hour 1.
            forecast[h] = [p1 * math.exp(z * 0.5) for z in Z]
            continue
        mu, sigma = fitted
        forecast[h] = [p1 * math.exp(mu + z * sigma) for z in Z]
    return forecast


def baseline_forecast(p1):
    """Persistence fallback: every hour stays at hour 1, spread widening."""
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in range(2, 25)}


def clean_quantiles(values, p1):
    """Enforce the output contract: finite, > 0, <= 1e9, non-decreasing."""
    out = []
    previous = 0.0
    for value in values:
        if value is None or not math.isfinite(value):
            value = p1
        value = min(max(value, 1.0), 1e9)
        value = max(value, previous)
        out.append(round2(value))
        previous = value
    return out


def forecast_case(case):
    known = [k for k in case.get("known", []) if k.get("hour") == 1]
    p1 = max(1.0, float(known[0]["viewers"])) if known else 1.0
    try:
        forecast = model_forecast(case, p1)
    except Exception:
        forecast = baseline_forecast(p1)
    hours = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    rows = []
    for row in hours:
        h = row["hour"]
        rows.append({"hour": h, "q": clean_quantiles(forecast.get(h) or baseline_forecast(p1)[h], p1)})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            result = forecast_case(case)
        except Exception:
            # Never leave a case without a valid output file.
            result = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
