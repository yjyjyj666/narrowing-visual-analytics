"""Figure: when first alarms occur in the MF case, under both references (10% target)."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RESULTS = Path(__file__).resolve().parents[1] / "results" / "mf"
DETECTORS = {"cumulative": ("Cumulative", "#eb6834"), "rolling10": ("Rolling-10", "#2a78d6"),
             "rolling20": ("Rolling-20", "#1d9e75")}
TARGET = 0.10
REFERENCES = {"startup": "Reference: index 19 (last startup step)",
              "early_mf": "Reference: index 29 (tenth MF step)"}

# Step 1: first-alarm counts per 10-index bucket, pooled over the five seeds.
with (RESULTS / "mf_alarm_timing.csv").open(newline="") as file:
    rows = [r for r in csv.DictReader(file) if float(r["target_false_alarm_rate"]) == TARGET]
with (RESULTS / "mf_alarm_summary.csv").open(newline="") as file:
    users = int(next(csv.DictReader(file))["users"])

# Step 2: one panel per reference, grouped bars per detector, same bucket axis.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
buckets = sorted({(int(r["first_index"]), int(r["last_index"])) for r in rows})
for ax, (reference, title) in zip(axes, REFERENCES.items()):
    for offset, (detector, (label, color)) in zip((-0.27, 0, 0.27), DETECTORS.items()):
        counts = {(int(r["first_index"]), int(r["last_index"])): int(r["alarmed"]) for r in rows
                  if r["reference"] == reference and r["detector"] == detector}
        x = [i + offset for i, b in enumerate(buckets) if b in counts]
        ax.bar(x, [counts[b] / users for b in buckets if b in counts], width=0.27, color=color, label=label, zorder=3)
    if reference == "startup":
        # Timing label only: alarms here are not attributed to the switch to MF.
        ax.axvspan(-0.5, 0.5, color="#e6e5e1", zorder=0)
        ax.text(0, 0.335, "first ten\nMF steps", ha="center", va="top", fontsize=9, color="#52514e")
    ax.set_xticks(range(len(buckets)), [f"{a}–{b}" for a, b in buckets], rotation=45, ha="right")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("Index of first alarm (0-based)")
    ax.grid(axis="y", color="#e6e5e1", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Share of MF users")
axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axes[0].set_ylim(0, 0.35)
axes[1].legend(frameon=False, title="Detector")
fig.suptitle(f"MF case: timing of first alarms at the {TARGET:.0%} false-alarm target "
             f"({users:,} users, seeds 123–127)", x=0.01, ha="left", fontsize=13)
fig.text(0.01, 0.01, "Thresholds come from the synthetic calibration (seeds 0–4). "
         "Alarm timing only; MF has no known change point.", fontsize=9, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(RESULTS / "mf_alarm_timing.png", dpi=170)
print(f"Saved {RESULTS / 'mf_alarm_timing.png'}")
