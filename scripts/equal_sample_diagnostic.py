"""Equal-sample diagnostic: does cumulative diversity steady itself only through sample size?

On the stationary evaluation histories (seeds 5-9), nothing changes after index 19,
so any change in a measure's spread across users over time comes from how it is
computed. Each measure is divided by its own value at index 19 (the alarm
reference), except the recent-versus-early ratio, which already compares 10 recent
items with 10 items drawn from earlier history (metrics.recent_vs_early, 20 draws).

If cumulative's spread shrinks over time while the equal-sample ratio's spread
stays flat, cumulative's apparent stability is a sampling effect: it averages ever
more items, so a fixed relative drop becomes harder to reach.
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from metrics import diversity_series, recent_vs_early  # noqa: E402
from generate_synthetic_batch import STEPS, generate_batch  # noqa: E402

OUTPUT = REPO_ROOT / "results" / "synthetic" / "equal_sample_diagnostic.csv"
EVALUATION_SEEDS = range(5, 10)
REFERENCE_INDEX = 19
QUANTILES = (0.1, 0.5, 0.9)


def main():
    measures = {"cumulative": [], "rolling10": [], "rolling20": [], "recent_vs_early": []}
    for seed in EVALUATION_SEEDS:
        # Step 1: stationary histories only (scenario index 0).
        items, histories, _, _ = generate_batch(seed, 500, 2, 40, 80)
        stationary = histories[:, 0]
        series = diversity_series(items, stationary)
        # Step 2: relative change from the alarm reference, the quantity thresholds act on.
        for name in ("cumulative", "rolling10", "rolling20"):
            measures[name].append(series[name] / series[name][:, REFERENCE_INDEX:REFERENCE_INDEX + 1])
        # Its own stream per seed; data/synthetic.json draws from [seed, 2] over all three
        # scenarios, so individual ratios there differ slightly from these.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        measures["recent_vs_early"].append(recent_vs_early(items, stationary, rng))
        print(f"seed {seed}: 500 stationary histories", flush=True)

    # Step 3: quantiles across all 2,500 users at every index from the reference on.
    rows = []
    for name, chunks in measures.items():
        values = np.concatenate(chunks)
        assert np.isfinite(values[:, REFERENCE_INDEX:]).all()
        low, median, high = np.quantile(values[:, REFERENCE_INDEX:], QUANTILES, axis=0)
        for offset, index in enumerate(range(REFERENCE_INDEX, STEPS)):
            rows.append({"measure": name, "index": index, "users": len(values),
                         "p10": low[offset], "median": median[offset], "p90": high[offset],
                         "spread_p90_minus_p10": high[offset] - low[offset]})
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")

    # Step 4: print the spread at a few indices for a quick read.
    spread = {(r["measure"], r["index"]): r["spread_p90_minus_p10"] for r in rows}
    for name in measures:
        print(f"{name:16s}" + "  ".join(f"idx {i}: {spread[name, i]:.3f}" for i in (30, 60, 90, 119)))


if __name__ == "__main__":
    main()
