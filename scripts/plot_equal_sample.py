"""Figure: spread across stationary users of each measure, relative to its reference, over time."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results" / "synthetic"
MEASURES = {"cumulative": ("Cumulative / value at 19", "#eb6834"),
            "rolling10": ("Rolling-10 / value at 19", "#2a78d6"),
            "rolling20": ("Rolling-20 / value at 19", "#1d9e75"),
            "recent_vs_early": ("Recent 10 / earlier 10 (equal sample)", "#7f77dd")}

# Step 1: per-index quantiles over 2,500 stationary users.
with (RESULTS / "equal_sample_diagnostic.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))
series = {m: [r for r in rows if r["measure"] == m] for m in MEASURES}

# Step 2: left, the 10-90% spread over time; right, median and band for two measures.
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
for measure, (label, color) in MEASURES.items():
    index = [int(r["index"]) for r in series[measure]]
    left.plot(index, [float(r["spread_p90_minus_p10"]) for r in series[measure]], color=color, lw=2, label=label)
for measure in ("cumulative", "recent_vs_early"):
    label, color = MEASURES[measure]
    index = [int(r["index"]) for r in series[measure]]
    right.fill_between(index, [float(r["p10"]) for r in series[measure]],
                       [float(r["p90"]) for r in series[measure]], color=color, alpha=0.2, lw=0)
    right.plot(index, [float(r["median"]) for r in series[measure]], color=color, lw=2, label=label)
left.set(title="Spread across users (90th minus 10th percentile)", xlabel="Index (0-based)", ylabel="Spread")
right.set(title="Median with 10–90% band", xlabel="Index (0-based)", ylabel="Relative value")
for ax in (left, right):
    ax.grid(color="#e6e5e1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("Equal-sample diagnostic on stationary histories (2,500 users, evaluation seeds 5–9)",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "equal_sample_diagnostic.png", dpi=170)
print(f"Saved {RESULTS / 'equal_sample_diagnostic.png'}")
