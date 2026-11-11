"""Plot a controlled example using the team's metrics.py."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", nargs="?", type=Path,
                        default=REPO_ROOT / "src" / "metrics.py",
                        help="Path to the shared metrics module")
    args = parser.parse_args()
    metrics_file = args.metrics_file.resolve(strict=True)
    sys.path.insert(0, str(metrics_file.parent))
    from metrics import diversity_series

    # Each row is a separate item. Items in the same topic share a vector
    # in this example, making the expected cosine distances easy to check.
    change_index = 50
    topics = np.r_[np.tile(np.arange(5), 10), np.zeros(70, dtype=int)]
    items = np.eye(5)[topics]
    history = np.arange(len(items))
    series = diversity_series(items, history)

    check_index = 59
    for name, start in [("cumulative", 0), ("rolling20", 40), ("rolling10", 50)]:
        expected = pdist(items[start : check_index + 1], "cosine").mean()
        observed = series[name][check_index]
        np.testing.assert_allclose(observed, expected, atol=1e-12)
        print(f"Step 60 | {name:10s} | {observed:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    steps = history + 1
    for name, label, color, style in [
        ("cumulative", "Cumulative", "#365977", "-"),
        ("rolling20", "Rolling-20", "#a65712", "--"),
        ("rolling10", "Rolling-10", "#006f67", "-."),
    ]:
        ax.plot(steps, series[name], label=label, color=color, linestyle=style, linewidth=2.4)

    ax.axvline(change_index + 1, color="#575757", linestyle=":", linewidth=1.5)
    ax.annotate("Switch to one topic (step 51)", (change_index + 1, 0.94),
                xytext=(7, 0), textcoords="offset points", color="#3e3e3e")
    ax.set(title="One consumption history, three diversity windows",
           xlabel="Consumption step", ylabel="Mean pairwise cosine distance",
           xlim=(1, len(steps)), ylim=(-0.045, 1.02))
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    ax.grid(axis="y", alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.41), frameon=False)
    output = REPO_ROOT / "results" / "synthetic" / "one_user_cosine_diversity.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
