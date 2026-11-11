import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

results_dir = Path(__file__).resolve().parents[1] / "results" / "synthetic"
with (results_dir / "synthetic_alarm_results.csv").open(newline="") as file:
    rows = {
        (float(row["target_false_alarm_rate"]), row["detector"], row["scenario"]): row
        for row in csv.DictReader(file)
    }

detectors = ("cumulative", "rolling10", "rolling20")
labels = ("Cumulative", "Rolling-10", "Rolling-20")
targets = (0.05, 0.10, 0.20)
fig, axes = plt.subplots(2, 3, figsize=(11.7, 6.15), sharey="row")
fig.patch.set_facecolor("#f8fafc")
fig.subplots_adjust(left=.09, right=.98, top=.79, bottom=.13, hspace=.33, wspace=.18)
fig.text(.09, .951, "Gradual narrowing is harder to detect", fontsize=22, weight="bold", color="#17263a")
fig.text(.09, .895, "2,500 evaluation histories per scenario  •  Three false-alarm calibration targets",
         fontsize=11, color="#3f5268")

for column, target in enumerate(targets):
    axes[0, column].set_title(f"{target:.0%} false-alarm target", loc="left", pad=10, weight="bold")
    for row_index, field in enumerate(("hits", "mean_delay_steps_among_hits")):
        ax = axes[row_index, column]
        ax.set_facecolor("#f8fafc")
        for offset, scenario, color, label in [
            (-.19, "abrupt", "#35638b", "Abrupt"),
            (.19, "gradual", "#b35835", "Gradual"),
        ]:
            selected = [rows[target, detector, scenario] for detector in detectors]
            values = [
                int(row[field]) / int(row["users"]) if field == "hits" else float(row[field])
                for row in selected
            ]
            ax.bar([i + offset for i in range(3)], values, width=.36, color=color,
                   label=label, zorder=3)
        ax.set_ylim((0, 1) if row_index == 0 else (0, 30))
        ax.set_yticks([0, .25, .5, .75, 1] if row_index == 0 else [0, 10, 20, 30])
        if row_index == 0:
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xticks(range(3), labels)
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#3f5268")
        ax.grid(axis="y", color="#dce5ee", zorder=0)
        ax.spines[:].set_visible(False)

axes[0, 0].set_ylabel("Detection rate")
axes[1, 0].set_ylabel("Mean delay (steps)")
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(.98, .91),
           ncols=2, frameon=False)
fig.text(.09, .046, "Delay averages include only detections within 40 steps of the known change.",
         fontsize=9, color="#3f5268")
fig.savefig(results_dir / "rq2_abrupt_vs_gradual.png", dpi=190, facecolor=fig.get_facecolor())

for target in targets:
    for detector in detectors:
        abrupt, gradual = (rows[target, detector, scenario] for scenario in ("abrupt", "gradual"))
        assert int(gradual["hits"]) < int(abrupt["hits"])
        assert float(gradual["mean_delay_steps_among_hits"]) > float(abrupt["mean_delay_steps_among_hits"])
print("Verified: all 9 detector × target pairs have fewer hits and longer mean delay under gradual narrowing.")
