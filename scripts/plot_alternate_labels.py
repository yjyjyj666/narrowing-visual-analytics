"""Figure: detection rate under the known change point versus the pilot label (H1)."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RESULTS = Path(__file__).resolve().parents[1] / "results" / "synthetic"
DETECTORS = {"cumulative": "Cumulative", "rolling10": "Rolling-10", "rolling20": "Rolling-20"}
LABELS = {"known_change": ("Known change point", "#35638b"), "pilot_label": ("Pilot label", "#b35835")}
TARGETS = (0.05, 0.10, 0.20)

# Step 1: pooled rows for abrupt and gradual histories together.
with (RESULTS / "alternate_label_results.csv").open(newline="") as file:
    rows = {(r["label"], r["detector"], float(r["target_false_alarm_rate"])): r
            for r in csv.DictReader(file) if r["true_scenario"] == "both"}

# Step 2: one panel per target; each bar is annotated with its realized false-alarm rate.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
for ax, target in zip(axes, TARGETS):
    for offset, (label, (name, color)) in zip((-0.2, 0.2), LABELS.items()):
        for i, detector in enumerate(DETECTORS):
            row = rows[label, detector, target]
            rate = float(row["detection_rate"])
            ax.bar(i + offset, rate, width=0.38, color=color, label=name if i == 0 else None, zorder=3)
            ax.text(i + offset, rate + 0.015, f"FA {float(row['false_alarm_rate']):.0%}",
                    ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(range(3), DETECTORS.values())
    ax.set_title(f"{target:.0%} false-alarm target", loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e5e1", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Detector")
axes[0].set_ylabel("Detection rate (hits / labelled positives)")
axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axes[0].set_ylim(0, 1.05)
axes[2].legend(frameon=False, title="Ground truth", loc="upper left")
fig.suptitle("H1: Rolling-10's lead over cumulative shrinks under the pilot label "
             "(evaluation seeds 5–9)", x=0.01, ha="left", fontsize=13)
fig.text(0.01, 0.01, "Same histories, thresholds and scoring rules. Known change: 5,000 abrupt and gradual "
         "histories; pilot label: the 2,080 of them it labels narrowed. FA = realized false-alarm rate.",
         fontsize=9, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(RESULTS / "alternate_label_comparison.png", dpi=170)
print(f"Saved {RESULTS / 'alternate_label_comparison.png'}")
