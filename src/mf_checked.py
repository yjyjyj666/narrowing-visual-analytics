"""Local MF adapter with an explicit policy for items with no training interactions.

This does not change simulated actual preferences or use them in training.
Keep the policy fixed and disclose it in experiments. Vendor sources stay intact.
"""
import numpy as np
from trecs.models import ImplicitMF
from trecs.models.recommender import BaseRecommender


class CheckedImplicitMF(ImplicitMF):
    def train(self):
        super().train()
        if self.als_model is None:
            return
        fitted = self.als_model
        user_ids = np.asarray(fitted.user_index_, dtype=int)
        item_ids = np.asarray(fitted.item_index_, dtype=int)
        # Preserve the order in which LensKit returns factor rows.
        self.users_hat.value[user_ids, :] = fitted.user_features_
        self.items_hat.value[:, item_ids] = fitted.item_features_.T
        cold = np.ones(self.num_items, dtype=bool)
        cold[item_ids] = False
        # A zero latent vector yields a zero predicted score, replacing arbitrary
        # random scores for items on which the model has never been trained.
        self.items_hat.value[:, cold] = 0.0
        BaseRecommender.train(self)
        if not np.isfinite(self.predicted_user_item_scores).all():
            raise ValueError('MF produced non-finite predictions')


class InterleavedImplicitMF(CheckedImplicitMF):
    """Add random exploration items without repeating an item on the slate."""

    def generate_recommendations(self, k, item_indices=None):
        # T-RECS ranks items before drawing the random part of the same slate.
        self._ranked_items = super().generate_recommendations(k, item_indices)
        return self._ranked_items

    def choose_interleaved_items(self, k, item_indices):
        ranked = getattr(self, '_ranked_items', None)
        if k and ranked is not None and ranked.size:
            # Keep the eligible indices in their original order for seeded draws.
            unranked = ~(item_indices[:, :, None] == ranked[:, None, :]).any(axis=2)
            item_indices = item_indices[unranked].reshape(self.num_users, -1)
        return super().choose_interleaved_items(k, item_indices)
