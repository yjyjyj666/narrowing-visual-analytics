"""Generate paired synthetic histories and compute diversity using metrics.py."""

import argparse
import sys
from pathlib import Path

import numpy as np

SCENARIOS = ("stationary", "abrupt", "gradual")
STEPS = 120
CANDIDATES = 10
TOPICS = 20


def generate_history(rng, scores, restricted, change_index, scenario):
    available = np.ones(len(scores), dtype=bool)
    history = np.empty(STEPS, dtype=int)
    for step in range(STEPS):
        probability = 0.0
        if step >= change_index:
            if scenario == "abrupt":
                probability = 1.0
            elif scenario == "gradual":
                probability = min(1.0, (step - change_index) / 30)
        restricted_count = np.count_nonzero(rng.random(CANDIDATES) < probability)
        restricted_pool = np.flatnonzero(available & restricted)
        if len(restricted_pool) < restricted_count:
            raise ValueError(
                f"{scenario}, step {step + 1}: need {restricted_count} restricted "
                f"candidates, but only {len(restricted_pool)} remain"
            )
        targeted = rng.choice(restricted_pool, size=restricted_count, replace=False)
        open_pool = available.copy()
        open_pool[targeted] = False
        other = rng.choice(
            np.flatnonzero(open_pool), size=CANDIDATES - restricted_count, replace=False
        )
        candidates = np.concatenate([targeted, other])
        chosen = candidates[np.argmax(scores[candidates])]
        history[step] = chosen
        available[chosen] = False
    return history


def generate_batch(seed, users, top_k, change_min, change_max):
    """Create an item pool and three histories per user with a shared prefix."""
    rng = np.random.default_rng(seed)
    concentration = np.full(TOPICS, 0.5)
    items = rng.dirichlet(concentration, size=2000)
    topics = items.argmax(axis=1)
    preferences = rng.dirichlet(concentration, size=users)
    favorite_topics = np.argsort(preferences, axis=1)[:, -top_k:]
    changes = rng.integers(change_min, change_max + 1, size=users)
    histories = np.empty((users, len(SCENARIOS), STEPS), dtype=int)

    for user, preference in enumerate(preferences):
        restricted = np.isin(topics, favorite_topics[user])
        scores = items @ preference
        for index, scenario in enumerate(SCENARIOS):
            # Reuse the random stream so the three histories share their prefix.
            history_rng = np.random.default_rng(np.random.SeedSequence([seed, user, 1]))
            try:
                histories[user, index] = generate_history(
                    history_rng, scores, restricted, changes[user], scenario
                )
            except ValueError as error:
                raise ValueError(f"seed {seed}, user {user}, k={top_k}: {error}") from error
    return items, histories, changes, favorite_topics


def check_batch(items, histories, changes, favorite_topics):
    topics = items.argmax(axis=1)
    for user, change in enumerate(changes):
        stable, abrupt, gradual = histories[user]
        np.testing.assert_array_equal(stable[:change], abrupt[:change])
        np.testing.assert_array_equal(stable[:change], gradual[:change])
        assert all(len(np.unique(history)) == STEPS for history in histories[user])
        assert np.isin(topics[abrupt[change:]], favorite_topics[user]).all()
        assert np.isin(topics[gradual[change + 30:]], favorite_topics[user]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", type=Path, help="Path to Elaine's metrics.py")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--top-k", type=int, choices=(1, 2), default=2)
    parser.add_argument("--change-min", type=int, default=40, help="First possible change index")
    parser.add_argument("--change-max", type=int, default=80, help="Last possible change index")
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz file")
    args = parser.parse_args()
    if args.seed < 0 or args.users < 1:
        parser.error("seed must be nonnegative and users must be positive")
    if not 20 <= args.change_min <= args.change_max < STEPS:
        parser.error("change indices must be ordered and between 20 and 119")
    if args.output.suffix != ".npz":
        parser.error("output must end in .npz")

    metrics_file = args.metrics_file.resolve(strict=True)
    if metrics_file.name != "metrics.py":
        parser.error("metrics_file must point to metrics.py")
    sys.path.insert(0, str(metrics_file.parent))
    from metrics import diversity_series

    items, histories, changes, favorite_topics = generate_batch(
        args.seed, args.users, args.top_k, args.change_min, args.change_max
    )
    check_batch(items, histories, changes, favorite_topics)
    flat_scores = diversity_series(items, histories.reshape(-1, STEPS))
    scores = {name: values.reshape(histories.shape) for name, values in flat_scores.items()}

    # The stationary scenario has no known change point.
    change_indices = np.column_stack((np.full(args.users, -1), changes, changes))
    np.savez_compressed(
        args.output, item_vectors=items, histories=histories, t_star=change_indices,
        favorite_topics=favorite_topics, scenarios=SCENARIOS, seed=args.seed,
        top_k=args.top_k, **scores,
    )
    print(f"Saved {args.users} users × {len(SCENARIOS)} scenarios × {STEPS} steps to {args.output}")


if __name__ == "__main__":
    main()
