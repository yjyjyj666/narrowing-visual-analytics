"""Small adapter smoke test; no claim about recommendation quality."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mf_checked import CheckedImplicitMF
warnings.filterwarnings('ignore', category=FutureWarning)
m = CheckedImplicitMF(num_users=50, num_items=200, num_latent_factors=4,
                      seed=19, model_params={'iterations':20, 'rng_spec':19})
m.run(5, startup=True, repeated_items=False, no_new_items=True)
m.train()
a=m.als_model
u=np.asarray(a.user_index_, dtype=int); i=np.asarray(a.item_index_, dtype=int)
cold=np.ones(200, dtype=bool); cold[i]=False
assert np.allclose(m.users_hat.value[u], a.user_features_)
assert np.allclose(m.items_hat.value[:,i].T, a.item_features_)
assert np.all(m.predicted_user_item_scores[:,cold] == 0)
previous=m.predicted_user_item_scores.copy()
m.run(2, reset_interactions=False, repeated_items=False, no_new_items=True)
m.train()
assert len(m.all_interactions)==350
assert not np.allclose(previous, m.predicted_user_item_scores)
assert np.isfinite(m.predicted_user_item_scores).all()
print('PASS: index alignment, cold-item scores, retained history and model updates')
