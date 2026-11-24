"""Check distinct slates, the Random comparison and the narrowing counts."""

from itertools import combinations
import warnings

import numpy as np

from run_mf_experiment import InterleavedImplicitMF, narrowing_counts, random_maximum_weights


def check_slate():
    # Leave exactly ten eligible items. Eight ranked plus two Random must cover them all.
    num_items = 11
    users = np.eye(2)
    values = np.linspace(0.0, 1.0, num_items)
    items = np.stack((values, 1.0 - values))
    model = InterleavedImplicitMF(
        num_users=len(users), num_items=num_items, num_latent_factors=2,
        actual_user_representation=users, actual_item_representation=items,
        num_items_per_iter=10, seed=137,
    )
    model.run(1, startup=True, repeated_items=False, no_new_items=True)
    first_choice = model.interactions.copy()
    slate = model.recommend(random_items_per_iter=2, repeated_items=False)
    eligible = [np.delete(np.arange(num_items), previous) for previous in first_choice]
    np.testing.assert_array_equal(np.sort(slate, axis=1), eligible)


def check_random_expectation():
    # Compare the closed-form weights with direct enumeration on a small catalog.
    utilities = np.array([.1, .1, .2, .3, .5, .55, .6, .7, .8, .9, .9, 1.0])
    enumerated = np.mean([max(slate) for slate in combinations(utilities, 10)])
    np.testing.assert_allclose(utilities @ random_maximum_weights(len(utilities)), enumerated)


def check_narrowing():
    # Include a later qualifying window, non-favorite topics and a 19-step near miss.
    histories = np.array([[7] + [0, 1] * 10, [6] + [2, 3] * 10, [9] + [0, 1] * 9 + [0, 8]])
    favorites = np.tile([0, 1], (len(histories), 1))
    assert narrowing_counts(histories, favorites) == {
        'any_one_or_two_topics': 2, 'own_favorite_topics': 1, 'ending_one_or_two_topics': 2,
    }


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='No measurements are currently defined')
    check_slate()
    check_random_expectation()
    check_narrowing()
    print('PASS: distinct eligible slates, exact Random expectation and twenty-step narrowing')
