"""Check diversity measures against brute-force pairwise distances, and the alarm rule on fixed series."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics import diversity_series, first_alarms

rng = np.random.default_rng(7)
items = rng.dirichlet(np.full(20, .5), size=2000)
histories = np.stack([rng.choice(2000, size=120, replace=False) for _ in range(40)])
series = diversity_series(items, histories)

# Every defined value equals the mean pairwise cosine distance of the items it covers.
def brute(history, a, b):
    return pdist(items[history[a:b]], 'cosine').mean()
for u in range(5):
    for t in range(120):
        expected = {'cumulative': brute(histories[u], 0, t+1) if t >= 1 else np.nan,
                    'rolling10': brute(histories[u], t-9, t+1) if t >= 9 else np.nan,
                    'rolling20': brute(histories[u], t-19, t+1) if t >= 19 else np.nan}
        for name, value in expected.items():
            assert np.isclose(series[name][u, t], value, equal_nan=True), (name, u, t)

# Chunking and the single-history form give the same values as one batch.
small_chunks = diversity_series(items, histories, chunk=7)
single = diversity_series(items, histories[3])
for name in series:
    assert np.allclose(series[name], small_chunks[name], equal_nan=True)
    assert np.allclose(series[name][3], single[name], equal_nan=True)

# Known extremes: repeated identical vectors give 0, orthogonal vectors give 1.
same = diversity_series(np.ones((30, 4)), np.zeros(30, dtype=int), windows=(10,))
assert np.allclose(same['rolling10'][9:], 0) and np.isnan(same['rolling10'][:9]).all()
ortho = diversity_series(np.eye(30), np.arange(30), windows=(10,))
assert np.allclose(ortho['cumulative'][1:], 1) and np.isnan(ortho['cumulative'][0])

# Alarm fires on the third consecutive step below (1 - theta) * reference, not before start.
values = np.array([[1, 1, .5, .5, .5, 1, 1, 1, 1, 1],   # run at 2-4 -> fires at 4
                   [1, 1, .5, .5, 1, .5, .5, 1, 1, 1],  # no run of three
                   [.5, .5, .5, 1, .5, .5, .5, 1, 1, 1],  # run before start ignored -> fires at 6
                   [1, 1, .5, .5, np.nan, .5, 1, 1, 1, 1]])  # NaN breaks the run
assert first_alarms(values, 1.0, theta=.3, start=2).tolist() == [4, -1, 6, -1]
assert first_alarms(values, np.array([1., 1., 1., 1.]), theta=.6, start=2).tolist() == [-1] * 4
assert first_alarms(values[0], 1.0, theta=.3, start=2) == 4
try:
    first_alarms(values, np.nan, theta=.3, start=2)
    raise AssertionError('NaN reference accepted')
except ValueError:
    pass
print('PASS: diversity matches brute force; chunking, extremes and alarm rule checked')
