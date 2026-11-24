"""Consumption-diversity measures and the threshold alarm shared by all detectors.

Diversity of a set of items is the mean pairwise cosine distance of their actual
attribute vectors. Steps are 0-based, and the value at step t uses only the items
consumed at steps 0..t, so every measure is online and never looks ahead.
The formulas are migrated unchanged from the pilot analysis (POC).
"""
import numpy as np


def _unit_rows(item_vectors):
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError('every item needs a non-zero attribute vector')
    return item_vectors / norms


def _window_diversity(prefix, a, b):
    """Diversity of the items at steps a..b-1, for every history in `prefix`.

    prefix[:, i, j] is the summed similarity of all pairs (steps < i, steps < j), so
    a block of the similarity matrix is four lookups. The block sum S includes the
    n diagonal entries, each 1, so the summed distance over ordered pairs i != j is
    n(n-1) - (S - n) = n*n - S.
    """
    n = b - a
    block = prefix[:, b, b] - prefix[:, a, b] - prefix[:, b, a] + prefix[:, a, a]
    return (n * n - block) / (n * (n - 1))


def diversity_series(item_vectors, histories, windows=(10, 20), chunk=500):
    """Cumulative and rolling diversity at every step of each history.

    item_vectors: (num_items, dim) actual item attributes. T-RECS stores them as
        (dim, num_items), so transpose those before calling.
    histories: (num_users, num_steps) consumed item indices, or one 1-D history.
    Returns {'cumulative': ..., 'rolling10': ..., 'rolling20': ...}, each shaped like
    `histories`. A step without enough items (fewer than two items for cumulative,
    fewer than W for rolling-W) is NaN.
    """
    histories = np.asarray(histories, dtype=int)
    single = histories.ndim == 1
    histories = np.atleast_2d(histories)
    num_users, num_steps = histories.shape
    if any(w < 2 or w > num_steps for w in windows):
        raise ValueError('each window must hold between 2 and num_steps items')
    unit = _unit_rows(np.asarray(item_vectors, dtype=float))
    steps = np.arange(num_steps)
    series = {name: np.full((num_users, num_steps), np.nan)
              for name in ['cumulative'] + [f'rolling{w}' for w in windows]}
    # Chunk users so the (users, steps, steps) prefix array stays small.
    for lo in range(0, num_users, chunk):
        rows = slice(lo, lo + chunk)
        vectors = unit[histories[rows]]
        prefix = np.zeros((len(vectors), num_steps + 1, num_steps + 1))
        prefix[:, 1:, 1:] = (vectors @ vectors.transpose(0, 2, 1)).cumsum(1).cumsum(2)
        # Cumulative: all items at steps 0..t; a pair first exists at t = 1.
        t = steps[1:]
        series['cumulative'][rows, 1:] = _window_diversity(prefix, np.zeros_like(t), t + 1)
        # Rolling-W: the last W items, steps t-W+1..t; first full window at t = W-1.
        for w in windows:
            t = steps[w - 1:]
            series[f'rolling{w}'][rows, w - 1:] = _window_diversity(prefix, t - w + 1, t + 1)
    if single:
        return {name: values[0] for name, values in series.items()}
    return series


def first_alarms(values, reference, theta, start, consecutive=3):
    """Step at which each series first raises an alarm, or -1 if it never does.

    An alarm fires once a series has stayed below (1 - theta) * reference for
    `consecutive` steps, counting only steps >= start. The returned step is the
    last step of that run, because an online detector cannot confirm it earlier.
    values: (num_users, num_steps) or one 1-D series; reference: scalar or one
    value per series. NaN values never count toward a run.
    """
    values = np.asarray(values, dtype=float)
    single = values.ndim == 1
    values = np.atleast_2d(values)
    reference = np.broadcast_to(np.asarray(reference, dtype=float), values.shape[:1])
    if not np.isfinite(reference).all():
        raise ValueError('reference must be finite; check the reference step has enough items')
    below = values[:, start:] < (1 - theta) * reference[:, None]
    run = np.zeros(len(values), dtype=int)
    alarm = np.full(len(values), -1)
    for offset in range(below.shape[1]):
        run = np.where(below[:, offset], run + 1, 0)
        fired = (run == consecutive) & (alarm < 0)
        alarm[fired] = start + offset
    return alarm[0] if single else alarm
