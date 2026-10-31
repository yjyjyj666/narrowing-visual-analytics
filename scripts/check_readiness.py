"""Readiness audit only, not a test of the project's narrowing hypothesis."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from trecs.models import ImplicitMF, RandomRecommender

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='No measurements are currently defined')
OUT = Path(__file__).resolve().parents[1] / 'results' / 'readiness'
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = dict(users=500, items=2000, attributes=20, factors=10,
              startup_steps=20, evaluation_steps=100, slate_size=10,
              als_iterations=5, drift=0, repeated_consumption=False)

def check_fit(model):
    a = model.als_model
    ui = np.asarray(a.user_index_, dtype=int)
    ii = np.asarray(a.item_index_, dtype=int)
    assert np.allclose(model.users_hat.value[ui], a.user_features_)
    assert np.allclose(model.items_hat.value[:, ii].T, a.item_features_)
    assert np.isfinite(model.predicted_user_item_scores).all()
    return len(ii)

def audit(kind, seed):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    users = rng.dirichlet(np.full(CONFIG['attributes'], .5), size=CONFIG['users'])
    items = rng.dirichlet(np.full(CONFIG['attributes'], .5), size=CONFIG['items']).T
    common = dict(num_users=CONFIG['users'], num_items=CONFIG['items'],
                  actual_user_representation=users.copy(), actual_item_representation=items.copy(),
                  num_items_per_iter=CONFIG['slate_size'], seed=seed)
    if kind == 'MF':
        model = ImplicitMF(**common, num_latent_factors=CONFIG['factors'],
                           model_params={'iterations': CONFIG['als_iterations'], 'rng_spec': seed})
    else:
        model = RandomRecommender(**common)
    consumed = np.zeros((CONFIG['users'], CONFIG['items']), dtype=bool)
    history, slates, rows, norms, learned_coverage = [], [], [], [], []
    for step in range(CONFIG['startup_steps'] + CONFIG['evaluation_steps']):
        startup = step < CONFIG['startup_steps']
        kwargs = dict(startup=startup, repeated_items=False, no_new_items=True)
        if kind == 'MF':
            kwargs['reset_interactions'] = False
        else:
            kwargs['disable_tqdm'] = True
        model.run(1, **kwargs)
        chosen = np.asarray(model.interactions, dtype=int).copy()
        shown = np.asarray(model.items_shown, dtype=int).copy()
        assert chosen.shape == (CONFIG['users'],)
        assert shown.shape == (CONFIG['users'], CONFIG['slate_size'])
        assert ((shown >= 0) & (shown < CONFIG['items'])).all()
        assert (shown == chosen[:, None]).any(axis=1).all()
        assert not consumed[np.arange(CONFIG['users']), chosen].any()
        assert not consumed[np.arange(CONFIG['users'])[:, None], shown].any()
        consumed[np.arange(CONFIG['users']), chosen] = True
        history.append(chosen); slates.append(shown)
        if kind == 'MF':
            assert len(model.all_interactions) == (step+1)*CONFIG['users']
            if step == CONFIG['startup_steps']-1 or not startup:
                previous = model.predicted_user_item_scores.copy()
                model.train()
                learned_coverage.append(check_fit(model))
                norms.append(float(np.linalg.norm(model.predicted_user_item_scores-previous)))
                rows.append(len(model.all_interactions))
        if (step+1) % 20 == 0:
            print(f'{kind} seed {seed}: {step+1} steps, {time.perf_counter()-start:.1f}s', flush=True)
    hist = np.stack(history)
    assert np.array_equal(model.actual_item_attributes, items)
    assert np.allclose(model.users.actual_user_profiles.value, users)
    if kind == 'MF':
        assert all(v > 0 for v in norms)
        pairs = model.all_interactions[['user', 'item']].to_numpy(dtype=int)
        assert np.array_equal(pairs[:,0], np.tile(np.arange(CONFIG['users']),hist.shape[0]))
        assert np.array_equal(pairs[:,1], hist.ravel())
    np.savez_compressed(OUT/f'{kind}_{seed}_histories.npz', choices=hist,
                        slates=np.stack(slates), actual_items=items, actual_users=users)
    result = dict(model=kind, seed=seed, config=CONFIG,
                  seconds=round(time.perf_counter()-start,3),
                  history_shape=list(hist.shape), selected_in_slate=True,
                  valid_item_ids=True, repeated_choices=0, previously_consumed_slate_items=0,
                  minimum_remaining_items=int((~consumed).sum(axis=1).min()),
                  true_attributes_fixed=True, true_preferences_fixed=True)
    if kind == 'MF':
        result.update(fits=len(rows), first_fit_interactions=rows[0], last_fit_interactions=rows[-1],
                      fitted_item_coverage_first=learned_coverage[0],
                      fitted_item_coverage_last=learned_coverage[-1],
                      index_alignment_checked_each_fit=True, scores_finite_each_fit=True,
                      score_change_norm_min=min(norms), score_change_norm_max=max(norms))
    (OUT/f'{kind}_{seed}_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

results=[]
for kind, seed in [('MF',123), ('MF',124), ('Random',123)]:
    results.append(audit(kind,seed))
summary=dict(python=sys.version,packages={p:metadata.version(p) for p in
             ['trecs','numpy','scipy','pandas','lenskit','numba']},results=results,
             limits=['Readiness audit, not scientific validation of narrowing.',
                     'Two MF seeds; no broad multi-seed inference.',
                     'ALS uses five iterations; no accuracy or convergence claim.',
                     'Fixed preferences only; drift configurations not tested.',
                     'Reference run is not a paired treatment-effect experiment.',
                     'Unseen items retain the original initialization until fitted; coverage is logged.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
