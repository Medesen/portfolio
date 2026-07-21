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
    session_items: pd.DataFrame, warm_split, warm_features: ItemFeatures,
    min_cold_support: int = 2,
) -> ColdStartEval:
    """Assemble the cold-start track from the raw sessions and a warm split.

    A cold candidate is a post-cutoff item absent from training with at least
    ``min_cold_support`` post-cutoff occurrences (so it is evaluable, not a
    singleton). An evaluable cold session has a cold *target* (its last item) and at
    least one *warm* prefix item to encode a history from.

    ``warm_features`` supplies the category vocabulary. Cold items must be encoded in
    the *same* vocabulary the two-tower trained on, or ``embed_cold_items`` would
    index the wrong rows of the category-embedding table — so the cold features reuse
    the warm vocabulary rather than fitting a fresh one.
    """
    cutoff = warm_split.cutoff
    warm_ids = set(int(i) for i in warm_split.item_ids)
    warm_to_col = {int(i): c for c, i in enumerate(warm_split.item_ids)}
    n_warm = len(warm_split.item_ids)

    # "Strictly cold" = zero interactions before the cutoff (genuinely new items),
    # not merely absent from the k-core-filtered warm vocabulary. An item that
    # existed pre-cutoff but was filtered out for low support is warm-but-dropped,
    # a different thing, and counting it as cold would badly inflate the share.
    pre_cutoff_items = set(session_items.loc[session_items["ts"] < cutoff, "itemid"].unique())

    post = session_items[session_items["ts"] >= cutoff].sort_values(
        ["session", "ts"], kind="mergesort"
    )
    is_warm = post["itemid"].isin(warm_ids)
    is_cold_item = ~post["itemid"].isin(pre_cutoff_items)

    # Cold candidates: strictly-new items with enough post-cutoff support to evaluate.
    post_cold_counts = post.loc[is_cold_item, "itemid"].value_counts()
    cold_ids = np.sort(post_cold_counts.index[post_cold_counts >= min_cold_support].to_numpy())
    cold_to_idx = {int(i): k for k, i in enumerate(cold_ids)}

    # Each post-cutoff session's target is its last item; prefix is the rest.
    is_last = ~post["session"].duplicated(keep="last")
    targets = post[is_last]
    prefixes = post[~is_last]

    # Warm prefix items, mapped to columns once (vectorised — no per-session scan).
    warm_prefix_pairs = prefixes[prefixes["itemid"].isin(warm_ids)].copy()
    warm_prefix_pairs["col"] = warm_prefix_pairs["itemid"].map(warm_to_col).astype(int)
    sessions_with_warm = set(warm_prefix_pairs["session"].unique())

    # Share statistics over all evaluable targets (those with a warm prefix).
    evaluable = targets[targets["session"].isin(sessions_with_warm)]
    cold_share = float((~evaluable["itemid"].isin(pre_cutoff_items)).mean())
    # Near-cold: target items with fewer than 5 pre-cutoff interactions (0 for the
    # strictly-cold). The cliff is rarely sharp, and where it sits is informative.
    pre_counts = session_items.loc[session_items["ts"] < cutoff, "itemid"].value_counts()
    item_pre_counts = evaluable["itemid"].map(pre_counts).fillna(0.0)
    near_cold_share = float((item_pre_counts < 5).mean())

    # Evaluable cold sessions: a cold target that clears min support and a warm prefix.
    cold_targets = evaluable[evaluable["itemid"].isin(cold_to_idx)].copy()
    keep_sessions = cold_targets["session"].to_numpy()
    row_of_session = {int(s): i for i, s in enumerate(keep_sessions)}

    kept_prefix = warm_prefix_pairs[warm_prefix_pairs["session"].isin(set(keep_sessions))]
    rows = kept_prefix["session"].map(row_of_session).to_numpy()
    cols = kept_prefix["col"].to_numpy()
    tgt_idx = cold_targets["itemid"].map(cold_to_idx).to_numpy()

    warm_prefix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(tgt_idx), n_warm),
    )
    cold_features = build_item_features(
        cutoff, cold_ids,
        category_vocab=warm_features.category_vocab,  # SAME vocab the two-tower trained on
        parent_vocab=warm_features.parent_vocab,
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
    """Rank each cold target among the **cold candidates only** (the plan's
    "restricted to cold items").

    This is the fair test against the category-popularity baseline, which also ranks
    among cold items: given that a *new* item is to be recommended, does the model's
    content-based score rank the right one? Ranking cold items inside the full warm
    catalogue instead would just measure that a content-only embedding cannot out-
    score established items' ID embeddings — true, but a different and less useful
    question (noted in the README).
    """
    cold_emb = model.embed_cold_items(cold.cold_features)  # (n_cold, d)
    user_emb = model._user_embeddings(cold.warm_prefix)    # (n_sessions, d)
    scores = user_emb @ cold_emb.T                         # (n_sessions, n_cold)

    rows = []
    for k in ks:
        hits, ndcgs = [], []
        for s in range(cold.n_sessions):
            ranked = np.argsort(-scores[s])[:k]
            target = int(cold.cold_target_idx[s])
            hits.append(hit_rate_at_k(ranked, {target}, k))
            ndcgs.append(ndcg_at_k(ranked, {target}, k))
        rows.append({"model": model.name, "k": k, "recall": float(np.mean(hits)),
                     "ndcg": float(np.mean(ndcgs)), "n_sessions": cold.n_sessions})
    return pd.DataFrame(rows)


def evaluate_category_popularity_cold(
    cold: ColdStartEval, session_items: pd.DataFrame, warm_split,
    warm_features: ItemFeatures, ks=(10, 20, 50),
) -> pd.DataFrame:
    """Baseline: recommend the most popular cold items in the session's category.

    The category of a session is the most common category among its warm prefix
    items (as-of cutoff); cold candidates in that category are ranked by their
    post-cutoff frequency. This is what a sensible engineer ships without a model —
    the bar the two-tower must clear, not zero. Warm and cold categories share the
    one vocabulary, so matching a session's category to a cold item's is meaningful.
    """
    warm_cat = warm_features.category_ids  # (n_warm,)
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
