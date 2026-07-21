"""Two-tower (dual-encoder) retrieval model — the architecture industry deploys.

Two small networks map users and items into a shared space; relevance is their dot
product. At serving time an item's vector is precomputed and a user's is compared
against all of them (exact here; approximate-nearest-neighbour in Stage 3). This is
the candidate-generation model job ads name.

Three design choices worth their comments:

**No user-ID embeddings.** The user tower pools the embeddings of the items in a
session's history and passes them through an MLP. It holds no per-user parameters,
so it generalises to any session with at least one interaction — including sessions
never seen in training. On a dataset where 79.6% of visitors appear once, per-user
parameters would generalise to no one; this is the same reason ALS needs a fold-in
for new users while EASE and ItemKNN handle them for free.

**The item tower sums an ID embedding with projected content** (category, parent
category, availability). The content path is what lets the model score an item with
no ID signal at all — the cold-start capability the classical models structurally
lack. ``item_tower_mode`` ablates it: ``id_only`` removes content (and cold-start
ability), ``content_only`` removes the ID.

**The logQ correction.** Training uses in-batch sampled softmax: within a batch of
(history, positive-item) pairs, every *other* row's positive serves as a negative.
Those negatives are drawn in proportion to item popularity, so popular items appear
as negatives far more often and the model learns to suppress them beyond what the
data warrants. Subtracting ``log P(sample item)`` from each logit (Yi et al. 2019)
corrects for it. ``logq_correction=False`` ablates it — the expected effect is on the
popularity bias of recommendations, not necessarily on accuracy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from reclab.features.item_features import ItemFeatures
from reclab.models.base import as_matrix, check_fitted


def _csr_to_torch(mat: sp.csr_matrix) -> torch.Tensor:
    """A scipy CSR as a torch sparse-COO tensor (for differentiable pooling)."""
    coo = mat.tocoo()
    idx = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.as_tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, mat.shape).coalesce()


class _ItemTower(nn.Module):
    def __init__(self, n_items: int, n_categories: int, n_parents: int,
                 emb_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.id_emb = nn.Embedding(n_items, emb_dim)
        # padding_idx=0 keeps the "unknown" category/parent at a fixed zero vector.
        self.cat_emb = nn.Embedding(n_categories, emb_dim, padding_idx=0)
        self.parent_emb = nn.Embedding(n_parents, emb_dim, padding_idx=0)
        self.avail_proj = nn.Linear(1, emb_dim)
        nn.init.normal_(self.id_emb.weight, std=0.01)
        nn.init.normal_(self.cat_emb.weight, std=0.01)
        nn.init.normal_(self.parent_emb.weight, std=0.01)

    def content(self, cat: torch.Tensor, parent: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        return self.cat_emb(cat) + self.parent_emb(parent) + self.avail_proj(avail.unsqueeze(-1))

    def forward(self, item_idx, cat, parent, avail) -> torch.Tensor:
        if self.mode == "content_only":
            return self.content(cat, parent, avail)
        idv = self.id_emb(item_idx)
        if self.mode == "id_only":
            return idv
        return idv + self.content(cat, parent, avail)


class _UserTower(nn.Module):
    def __init__(self, emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.mlp(pooled)


class TwoTower:
    name = "two_tower"

    def __init__(
        self,
        n_items: int,
        item_features: ItemFeatures,
        emb_dim: int = 64,
        hidden: int = 128,
        item_tower_mode: str = "id_plus_content",
        logq_correction: bool = True,
        temperature: float = 0.05,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 512,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if item_tower_mode not in ("id_plus_content", "id_only", "content_only"):
            raise ValueError(f"bad item_tower_mode: {item_tower_mode}")
        self.n_items = n_items
        self.features = item_features
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.item_tower_mode = item_tower_mode
        self.logq_correction = logq_correction
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.item_emb_: np.ndarray | None = None  # frozen (n_items, d) for scoring

    # -- feature tensors ---------------------------------------------------- #
    def _feature_tensors(self):
        f = self.features
        return (
            torch.as_tensor(f.category_ids, dtype=torch.long, device=self.device),
            torch.as_tensor(f.parent_ids, dtype=torch.long, device=self.device),
            torch.as_tensor(f.available, dtype=torch.float32, device=self.device),
        )

    def _all_item_embeddings(self, item_tower: _ItemTower) -> torch.Tensor:
        cat, parent, avail = self._feature_tensors()
        idx = torch.arange(self.n_items, device=self.device)
        return item_tower(idx, cat, parent, avail)

    # -- training ----------------------------------------------------------- #
    def fit(self, split) -> "TwoTower":
        torch.manual_seed(self.seed)
        sequences = split.train_sequences
        if sequences is None:
            raise ValueError("TwoTower.fit needs a SessionSplit carrying train_sequences")

        # One (history, target) example per session: predict the last item from
        # the rest — the same shape as the evaluation task.
        rows, cols, targets = [], [], []
        for seq in sequences:
            if len(seq) >= 2:
                r = len(targets)
                rows.extend([r] * (len(seq) - 1))
                cols.extend(int(i) for i in seq[:-1])
                targets.append(int(seq[-1]))
        if not targets:
            raise ValueError("no training sessions with >=2 items")
        targets = np.asarray(targets)
        # History as a sparse (n_examples x n_items) count matrix, so pooling is a
        # single sparse matmul per batch rather than a Python loop over sessions.
        history_mat = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(targets), self.n_items),
        )
        hist_counts = np.asarray(history_mat.sum(axis=1)).ravel()
        hist_counts[hist_counts == 0] = 1.0

        pop = np.asarray(split.train.sum(axis=0)).ravel() + 1.0
        log_sample_prob = torch.as_tensor(
            np.log(pop / pop.sum()), dtype=torch.float32, device=self.device
        )

        item_tower = _ItemTower(
            self.n_items, self.features.n_categories, self.features.n_parents,
            self.emb_dim, self.item_tower_mode,
        ).to(self.device)
        user_tower = _UserTower(self.emb_dim, self.hidden).to(self.device)
        params = list(item_tower.parameters()) + list(user_tower.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        rng = np.random.default_rng(self.seed)
        n = len(targets)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            item_tower.train()
            user_tower.train()
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < 2:  # in-batch negatives need >=2 rows
                    continue
                item_emb_all = self._all_item_embeddings(item_tower)

                # Pool each session's history in one sparse matmul: (B x n_items)
                # @ (n_items x d) / counts. Differentiable through item_emb_all.
                h_batch = _csr_to_torch(history_mat[batch])
                counts = torch.as_tensor(
                    hist_counts[batch], dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                pooled = torch.sparse.mm(h_batch, item_emb_all) / counts
                user_emb = user_tower(pooled)
                target_emb = item_emb_all[targets[batch]]

                logits = (user_emb @ target_emb.T) / self.temperature
                if self.logq_correction:
                    # Subtract log P(sampling the column's item) — the column item
                    # is the in-batch negative, sampled ~ its popularity.
                    logits = logits - log_sample_prob[targets[batch]].unsqueeze(0)
                labels = torch.arange(len(batch), device=self.device)
                loss = nn.functional.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

        item_tower.eval()
        user_tower.eval()
        with torch.no_grad():
            self.item_emb_ = self._all_item_embeddings(item_tower).cpu().numpy()
        self._user_tower = user_tower
        self._item_tower = item_tower  # retained so cold items can be embedded
        return self

    # -- scoring ------------------------------------------------------------ #
    def _user_embeddings(self, histories) -> np.ndarray:
        """Pool history-item embeddings, run the user MLP. Returns (n, d)."""
        counts = np.asarray(histories.sum(axis=1)).ravel()
        counts[counts == 0] = 1.0  # empty history -> zero pooled vector, not NaN
        pooled = np.asarray(histories @ self.item_emb_) / counts[:, None]
        with torch.no_grad():
            user = self._user_tower(torch.as_tensor(pooled, dtype=torch.float32))
        return user.cpu().numpy()

    def score(self, history) -> np.ndarray:
        check_fitted(self, "item_emb_")
        histories = as_matrix(history)
        user_emb = self._user_embeddings(histories)
        return user_emb @ self.item_emb_.T

    def embed_cold_items(self, cold_features: ItemFeatures) -> np.ndarray:
        """Content-only embeddings for items absent from training (Stage 2 cold-start).

        A cold item has no ID embedding, so it is embedded from its content alone —
        which is exactly why the content path exists. Categories unseen in training
        fall to the unknown index and contribute nothing, honestly.
        """
        check_fitted(self, "item_emb_")
        cat = torch.as_tensor(cold_features.category_ids, dtype=torch.long)
        parent = torch.as_tensor(cold_features.parent_ids, dtype=torch.long)
        avail = torch.as_tensor(cold_features.available, dtype=torch.float32)
        with torch.no_grad():
            return self._item_tower.content(cat, parent, avail).cpu().numpy()
