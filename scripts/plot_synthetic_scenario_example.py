"""Generate three paired example histories and plot the team's diversity scores."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 101
CHANGE_INDEX = 50
NUM_STEPS = 120
NUM_CANDIDATES = 10
TOPIC_COUNT = 20


def generate_history(item_topics, preference_scores, favorite_topic, scenario):
    """Use the same random stream for each scenario and never repeat an item."""
    rng = np.random.default_rng(SEED + 1)
    available = np.ones(len(item_topics), dtype=bool)
    favorite_items = item_topics == favorite_topic
    history = []
    for step in range(NUM_STEPS):
        restricted_probability = 0.0
        if step >= CHANGE_INDEX:
            if scenario == "Abrupt narrowing":
                restricted_probability = 1.0
            elif scenario == "Gradual narrowing":
                restricted_probability = min(1.0, (step - CHANGE_INDEX) / 30)

        restricted_count = np.count_nonzero(
            rng.random(NUM_CANDIDATES) < restricted_probability
        )
        candidates = rng.choice(
            np.flatnonzero(available & favorite_items),
            size=restricted_count,
            replace=False,
        )
        unrestricted_pool = available.copy()
        unrestricted_pool[candidates] = False
        other_candidates = rng.choice(
            np.flatnonzero(unrestricted_pool),
            size=NUM_CANDIDATES - restricted_count,
            replace=False,
        )
        candidates = np.concatenate([candidates, other_candidates])
        chosen = candidates[np.argmax(preference_scores[candidates])]
        available[chosen] = False
        history.append(chosen)

    return np.array(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", nargs="?", type=Path,
                        default=REPO_ROOT / "src" / "metrics.py",
                        help="Path to the shared metrics module")
    args = parser.parse_args()
    metrics_file = args.metrics_file.resolve(strict=True)
    sys.path.insert(0, str(metrics_file.parent))
    from metrics import diversity_series

    rng = np.random.default_rng(SEED)
    concentration = np.full(TOPIC_COUNT, 0.5)
    item_vectors = rng.dirichlet(concentration, size=2000)
    preference = rng.dirichlet(concentration)
    item_topics = item_vectors.argmax(axis=1)
    favorite_topic = preference.argmax()
    preference_scores = item_vectors @ preference
    scenarios = ["Stationary", "Abrupt narrowing", "Gradual narrowing"]
    histories = [
        generate_history(item_topics, preference_scores, favorite_topic, scenario)
        for scenario in scenarios
    ]

    for history in histories:
        assert len(np.unique(history)) == NUM_STEPS
        np.testing.assert_array_equal(history[:CHANGE_INDEX], histories[0][:CHANGE_INDEX])
    assert np.all(item_topics[histories[1][CHANGE_INDEX:]] == favorite_topic)
    assert np.all(item_topics[histories[2][CHANGE_INDEX + 30:]] == favorite_topic)

    scores = diversity_series(item_vectors, histories)
    steps = np.arange(1, NUM_STEPS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True, layout="constrained")
    fig.suptitle("Three consumption histories with the same first 50 steps", fontsize=15)
    for index, (ax, scenario, history) in enumerate(zip(axes, scenarios, histories)):
        for name, label, color, style in [
            ("cumulative", "Cumulative", "#365977", "-"),
            ("rolling20", "Rolling-20", "#a65712", "--"),
            ("rolling10", "Rolling-10", "#006f67", "-."),
        ]:
            ax.plot(steps, scores[name][index], label=label, color=color,
                    linestyle=style, linewidth=2)
        if index:
            ax.axvline(CHANGE_INDEX + 1, color="#555555", linestyle=":", linewidth=1.5)
            ax.text(CHANGE_INDEX + 3, 0.96, "Change starts", va="top",
                    color="#555555", transform=ax.get_xaxis_transform(), fontsize=9)
        ax.set(title=scenario, xlabel="Consumption step", xlim=(1, NUM_STEPS),
               ylim=(0, 1), xticks=[1, 20, 40, 60, 80, 100, 120])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        favorite_share = np.mean(item_topics[history[CHANGE_INDEX:]] == favorite_topic)
        print(f"{scenario}: {favorite_share:.1%} favorite-topic consumption after step 50")

    axes[0].set_ylabel("Mean pairwise cosine distance")
    axes[0].legend(loc="lower left", frameon=False)
    output = REPO_ROOT / "results" / "synthetic" / "synthetic_scenario_example.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
