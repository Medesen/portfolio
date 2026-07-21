"""Cold-start evaluation — the axis on which the classical winners fail structurally.

Stage 1's split *drops* test interactions whose target item never appeared in
training, because EASE/ItemKNN/ALS cannot score an item they have never seen. Cold
start is precisely the study of those dropped interactions, so this is a second
evaluation track, not a tweak to the warm one (the plan's sanctioned exception).

Precise scope — **item cold-start, not user cold-start.** The held-out target is a
*new item* (zero training interactions); the session's history is warm. A two-tower
with a content path can score a new item from its category alone; the classical
models return 0.0 by construction, and that row is the point of the table.

Two numbers keep the story honest:
- the **cold-item share**: what fraction of evaluable test targets are cold at all —
  an advantage on a slice nobody visits is not an advantage;
- a real baseline — **most-popular-cold-item-within-the-session's-category** — which
  is what a sensible engineer ships without a neural model. Beating *that* is the bar,
  not beating zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from reclab.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from reclab.features.item_features import ItemFeatures, build_item_features


@dataclass
class ColdStartEval:
    warm_prefix: sp.csr_matrix          # (n_sessions, n_warm) warm history bag
    cold_features: ItemFeatures         # content features for the cold candidates
    cold_target_idx: np.ndarray         # target's index into the cold candidate set
    warm_item_ids: np.ndarray
    cold_item_ids: np.ndarray
    cold_share: float                   # cold targets / all evaluable test targets
    near_cold_share: float              # targets with <5 train interactions

    @property
    def n_sessions(self) -> int:
        return self.warm_prefix.shape[0]


def build_cold_start_eval(
    session_items: pd.DataFrame, warm_split, min_cold_support: int = 2
) -> ColdStartEval:
    """Assemble the cold-start track from the raw sessions and a warm split.

    A cold candidate is a post-cutoff item absent from training with at least
    ``min_cold_support`` post-cutoff occurrences (so it is evaluable, not a
    singleton). An evaluable cold session has a cold *target* (its last item) and at
    least one *warm* prefix item to encode a history from.
    """
    cutoff = warm_split.cutoff
    warm_ids = set(int(i) for i in warm_split.item_ids)
    warm_to_col = {int(i): c for c, i in enumerate(warm_split.item_ids)}

    post = session_items[session_items["ts"] >= cutoff].sort_values(
        ["session", "ts"], kind="mergesort"
    )
    # Cold items: post-cutoff, not warm, with enough support to be evaluable.
    post_cold_counts = (
        post[~post["itemid"].isin(warm_ids)]["itemid"].value_counts()
    )
    cold_ids = np.sort(post_cold_counts.index[post_cold_counts >= min_cold_support].to_numpy())
    cold_to_idx = {int(i): k for k, i in enumerate(cold_ids)}

    # Each post-cutoff session's target is its last item; prefix is the rest.
    is_last = ~post["session"].duplicated(keep="last")
    targets = post[is_last]
    prefixes = post[~is_last]

    # Share of evaluable test targets that are cold (need a warm prefix to encode).
    warm_prefix_sessions = set(
        prefixes[prefixes["itemid"].isin(warm_ids)]["session"].unique()
    )
    evaluable = targets[targets["session"].isin(warm_prefix_sessions)]
    is_cold_target = ~evaluable["itemid"].isin(warm_ids)
    cold_share = float(is_cold_target.mean())

    # Near-cold: targets whose item has fewer than 5 training interactions.
    train_counts = np.asarray(warm_split.train.sum(axis=0)).ravel()
    near_cold = 0
    for item in evaluable["itemid"]:
        if int(item) in warm_to_col and train_counts[warm_to_col[int(item)]] < 5:
            near_cold += 1
    near_cold_share = near_cold / max(len(evaluable), 1)

    # Keep only sessions with a cold target that clears min support and a warm prefix.
    rows, cols, tgt_idx, keep_sessions = [], [], [], []
    row = 0
    for session, target_item in zip(targets["session"], targets["itemid"]):
        if int(target_item) not in cold_to_idx:
            continue
        warm_hist = prefixes[(prefixes["session"] == session)
                             & (prefixes["itemid"].isin(warm_ids))]["itemid"]
        if warm_hist.empty:
            continue
        for it in warm_hist:
            rows.append(row)
            cols.append(warm_to_col[int(it)])
        tgt_idx.append(cold_to_idx[int(target_item)])
        keep_sessions.append(session)
        row += 1

    warm_prefix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(tgt_idx), len(warm_split.item_ids)),
    )
    cold_features = build_item_features(
        cutoff, cold_ids,
        category_vocab=None, parent_vocab=None,  # fit fresh; unseen cats -> unknown
    )
    return ColdStartEval(
        warm_prefix=warm_prefix,
        cold_features=cold_features,
        cold_target_idx=np.asarray(tgt_idx),
        warm_item_ids=warm_split.item_ids,
        cold_item_ids=cold_ids,
        cold_share=cold_share,
        near_cold_share=near_cold_share,
    )


def evaluate_two_tower_cold(model, cold: ColdStartEval, ks=(10, 20, 50)) -> pd.DataFrame:
    """Rank each cold target within the full warm+cold candidate universe.

    The two-tower embeds warm items from training and cold items from content, so a
    new item competes head-to-head with the whole catalogue — the honest test.
    """
    warm_emb = model.item_emb_  # (n_warm, d)
    cold_emb = model.embed_cold_items(cold.cold_features)  # (n_cold, d)
    all_emb = np.vstack([warm_emb, cold_emb])
    n_warm = warm_emb.shape[0]

    user_emb = model._user_embeddings(cold.warm_prefix)  # (n_sessions, d)
    rows = []
    for k in ks:
        hits, ndcgs = [], []
        for s in range(cold.n_sessions):
            scores = user_emb[s] @ all_emb.T
            # Mask warm history so it can't crowd the top; cold items compete freely.
            seen = cold.warm_prefix[s].indices
            scores[seen] = -np.inf
            ranked = np.argsort(-scores)[:k]
            target = n_warm + int(cold.cold_target_idx[s])  # cold items sit after warm
            hits.append(hit_rate_at_k(ranked, {target}, k))
            ndcgs.append(ndcg_at_k(ranked, {target}, k))
        rows.append({"model": model.name, "k": k, "recall": float(np.mean(hits)),
                     "ndcg": float(np.mean(ndcgs)), "n_sessions": cold.n_sessions})
    return pd.DataFrame(rows)


def evaluate_category_popularity_cold(
    cold: ColdStartEval, session_items: pd.DataFrame, warm_split, ks=(10, 20, 50)
) -> pd.DataFrame:
    """Baseline: recommend the most popular cold items in the session's category.

    The category of a session is the most common category among its warm prefix
    items (as-of cutoff); cold candidates in that category are ranked by their
    post-cutoff frequency. This is what a sensible engineer ships without a model —
    the bar the two-tower must clear, not zero.
    """
    warm_feats = build_item_features(warm_split.cutoff, warm_split.item_ids,
                                     category_vocab=cold.cold_features.category_vocab,
                                     parent_vocab=cold.cold_features.parent_vocab)
    warm_cat = warm_feats.category_ids  # (n_warm,)
    cold_cat = cold.cold_features.category_ids  # (n_cold,)

    post = session_items[session_items["ts"] >= warm_split.cutoff]
    cold_pop = post["itemid"].value_counts()
    cold_freq = np.array([cold_pop.get(int(i), 0) for i in cold.cold_item_ids])

    rows = []
    for k in ks:
        hits, ndcgs = [], []
        for s in range(cold.n_sessions):
            hist_cats = warm_cat[cold.warm_prefix[s].indices]
            hist_cats = hist_cats[hist_cats != 0]
            pred_cat = np.bincount(hist_cats).argmax() if len(hist_cats) else 0
            # Rank cold candidates: same-category first, then by frequency.
            score = cold_freq + (cold_cat == pred_cat) * (cold_freq.max() + 1)
            ranked = np.argsort(-score)[:k]
            target = int(cold.cold_target_idx[s])
            hits.append(hit_rate_at_k(ranked, {target}, k))
            ndcgs.append(ndcg_at_k(ranked, {target}, k))
        rows.append({"model": "category_popularity", "k": k, "recall": float(np.mean(hits)),
                     "ndcg": float(np.mean(ndcgs)), "n_sessions": cold.n_sessions})
    return pd.DataFrame(rows)
