"""Reproduce the baseline, candidate and Random T-RECS development runs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mf_checked import InterleavedImplicitMF


USERS, ITEMS, TOPICS = 500, 2000, 20
SLATE_SIZE, STARTUP_STEPS, EVALUATION_STEPS = 10, 20, 100
ALS_ITERATIONS, ALS_WEIGHT, ALS_REGULARIZATION = 20, 40, 0.1
COHORT_SEED, COHORT_SIZE, NARROWING_WINDOW = 130, 10, 20
POLICIES = {'baseline': (10, 0), 'candidate': (20, 2), 'random': (0, SLATE_SIZE)}


def random_maximum_weights(pool_size):
    """Probability that each ascending rank is the best of a uniform slate."""
    weights = np.zeros(pool_size)
    weights[-1] = SLATE_SIZE / pool_size
    # Count subsets by their maximum rank without evaluating large binomials.
    for rank in range(pool_size - 1, SLATE_SIZE - 1, -1):
        weights[rank - 1] = weights[rank] * (rank - SLATE_SIZE + 1) / rank
    return weights


def narrowing_counts(topic_history, favorite_topics):
    """Count users who had a confirmed twenty-step run during evaluation."""
    windows = np.lib.stride_tricks.sliding_window_view(topic_history, NARROWING_WINDOW, axis=1)
    sorted_windows = np.sort(windows, axis=-1)
    changes = np.count_nonzero(np.diff(sorted_windows, axis=-1), axis=-1)
    # Check the prescribed favorite-topic pair separately from any fixed group.
    in_favorites = (topic_history[:, :, None] == favorite_topics[:, None, :]).any(axis=-1)
    favorite_windows = np.lib.stride_tricks.sliding_window_view(in_favorites, NARROWING_WINDOW, axis=1)
    return {
        'any_one_or_two_topics': int((changes <= 1).any(axis=1).sum()),
        'own_favorite_topics': int(favorite_windows.all(axis=-1).any(axis=1).sum()),
        'ending_one_or_two_topics': int((changes[:, -1] <= 1).sum()),
    }


def simulate(seed, policy):
    factors, random_slots = POLICIES[policy]
    has_mf = factors > 0

    # Separate the true profiles from the latent factors MF learns from interactions.
    rng = np.random.default_rng(seed)
    users = rng.dirichlet(np.full(TOPICS, 0.5), size=USERS)
    items = rng.dirichlet(np.full(TOPICS, 0.5), size=ITEMS).T
    truth = users @ items
    model = InterleavedImplicitMF(
        num_users=USERS, num_items=ITEMS, num_latent_factors=factors or POLICIES['baseline'][0],
        actual_user_representation=users.copy(), actual_item_representation=items.copy(),
        num_items_per_iter=SLATE_SIZE, seed=seed,
        model_params={'iterations': ALS_ITERATIONS, 'weight': ALS_WEIGHT,
                      'reg': ALS_REGULARIZATION, 'rng_spec': seed},
    )

    # Observe the same fixed cohort under every policy while training on all users.
    cohort = np.sort(np.random.default_rng(COHORT_SEED).choice(USERS, COHORT_SIZE, replace=False))
    rows = np.arange(USERS)
    step_count = STARTUP_STEPS + EVALUATION_STEPS
    histories = np.empty((USERS, step_count), dtype=np.int32)
    slates = np.empty((USERS, step_count, SLATE_SIZE), dtype=np.int32)
    consumed = np.zeros_like(truth, dtype=bool)
    quality = {name: np.empty((EVALUATION_STEPS, len(cohort))) for name in
               ('chosen_item_match', 'random_slate_match', 'random_best_item_match')}
    if has_mf:
        quality['pure_mf_slate_match'] = np.empty((EVALUATION_STEPS, len(cohort)))

    for step in range(step_count):
        startup = step < STARTUP_STEPS
        if not startup and has_mf:
            # Measure the learned ordering without changing the actual draws.
            rng_state = deepcopy(model.random_state.bit_generator.state)
            pure_slate = model.recommend(repeated_items=False)[cohort]
            model.random_state.bit_generator.state = rng_state

        model.run(1, startup=startup or not has_mf, reset_interactions=False, repeated_items=False,
                  random_items_per_iter=random_slots if not startup and has_mf else 0, no_new_items=True)
        chosen, shown = model.interactions, model.items_shown
        histories[:, step] = chosen
        slates[:, step] = shown

        # Check eligibility and independently verify the actual user choice.
        assert np.all((0 <= shown) & (shown < ITEMS))
        assert not consumed[rows[:, None], shown].any()
        assert np.all(np.diff(np.sort(shown, axis=1), axis=1) > 0)
        assert np.all((shown == chosen[:, None]).any(axis=1))
        np.testing.assert_allclose(truth[rows, chosen], truth[rows[:, None], shown].max(axis=1))
        assert len(model.all_interactions) == (step + 1) * USERS

        if not startup:
            # Compare with uniform slates from each user's eligible pool before consumption.
            index = step - STARTUP_STEPS
            eligible = truth[cohort][~consumed[cohort]].reshape(len(cohort), -1)
            quality['chosen_item_match'][index] = truth[cohort, chosen[cohort]]
            quality['random_slate_match'][index] = eligible.mean(axis=1)
            quality['random_best_item_match'][index] = (
                np.sort(eligible, axis=1) @ random_maximum_weights(eligible.shape[1])
            )
            if has_mf:
                quality['pure_mf_slate_match'][index] = truth[cohort[:, None], pure_slate].mean(axis=1)

        consumed[rows, chosen] = True
        if has_mf and STARTUP_STEPS - 1 <= step < step_count - 1:
            model.train()

    # Record raw histories with the existing exporter keys and enough provenance to compare runs.
    configuration = dict(seed=seed, policy=policy, latent_factors=factors, exploration_slots=random_slots,
                         iterations=ALS_ITERATIONS, weight=ALS_WEIGHT, regularization=ALS_REGULARIZATION,
                         startup_steps=STARTUP_STEPS, evaluation_steps=EVALUATION_STEPS,
                         cohort_seed=COHORT_SEED, narrowing_window=NARROWING_WINDOW)
    archive = dict(**configuration, histories=histories, slates=slates, items=items, users=users,
                   cohort=cohort, **quality)
    actual_topics = items.argmax(axis=0)[histories[:, STARTUP_STEPS:]]
    summary = dict(**configuration, users=USERS, items=ITEMS, cohort=cohort.tolist(),
                   narrowing=narrowing_counts(actual_topics, np.argsort(users, axis=1)[:, -2:]))
    for name, window in (('first_ten', slice(0, 10)), ('last_ten', slice(-10, None))):
        summary[name] = {key: float(values[window].mean()) for key, values in quality.items()}
    if has_mf:
        gap = quality['pure_mf_slate_match'] - quality['random_slate_match']
        summary['users_beating_random_in_last_ten'] = int((gap[-10:].mean(axis=0) > 0).sum())
        summary['steps_beating_random'] = int((gap.mean(axis=1) > 0).sum())
    return archive, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--policies', choices=POLICIES, nargs='+', default=list(POLICIES))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if any(not 100 <= seed <= np.iinfo(np.uint32).max for seed in args.seeds):
        parser.error('MF seeds must be between 100 and 4294967295')

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='No measurements are currently defined')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Publish each new archive only after the requested runs finish successfully.
    with TemporaryDirectory(dir=args.output_dir) as staging:
        staging = Path(staging)
        files, summaries = [], []
        for seed in dict.fromkeys(args.seeds):
            startup = None
            for policy in dict.fromkeys(args.policies):
                archive, summary = simulate(seed, policy)
                warmup = archive['histories'][:, :STARTUP_STEPS]
                if startup is None:
                    startup = warmup.copy()
                else:
                    np.testing.assert_array_equal(warmup, startup)
                relative_path = Path(policy) / f'mf_seed{seed}.npz'
                (staging / policy).mkdir(exist_ok=True)
                np.savez_compressed(staging / relative_path, **archive)
                files.append(relative_path)
                summaries.append(summary)
                print(f'Completed {policy}, seed {seed}', flush=True)

        (staging / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n', encoding='utf-8')
        for relative_path in files + [Path('summary.json')]:
            destination = args.output_dir / relative_path
            destination.parent.mkdir(exist_ok=True)
            (staging / relative_path).replace(destination)


if __name__ == '__main__':
    main()
