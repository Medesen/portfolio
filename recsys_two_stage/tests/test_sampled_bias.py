"""The Krichene & Rendle (2020) reversal, proven by construction.

Their result: sampled-negative evaluation can rank models in the *opposite*
order to full-catalogue evaluation. Most write-ups cite this. Here it is a test
that fails if the effect ever stops reproducing, because the whole project rests
on the claim that the protocol changes the answer.

The construction
----------------
Catalogue of M=10,000 items, n=100 sampled negatives, metric HitRate@10.

  Full-catalogue HR@10 hits iff the target's true rank <= 10.
  Sampled HR@10 hits iff <= 9 of the 100 sampled negatives outrank the target,
  i.e. roughly iff true rank <= ~900 — sampling cannot tell "rank 12" from
  "rank 5", so a just-missed target looks like a hit.

Two models, with per-session target ranks chosen directly:

  Model A: half its targets at rank 5, half at rank 5,000.
  Model B: every target at rank 12.

  Full-catalogue: A hits on its rank-5 half -> HR@10 = 0.5;  B never hits -> 0.0.
                  => A beats B.
  Sampled:        A's rank-5,000 targets still look like misses (~50 negatives
                  above), so A ~ 0.5; B's rank-12 targets almost always land in
                  the sampled top-10, so B ~ 1.0.
                  => B beats A.

The ordering flips. That is the finding.
"""

from __future__ import annotations

import numpy as np

from reclab.evaluation.metrics import hit_rate_at_k
from reclab.evaluation.sampled import sampled_metrics_from_scores

M = 10_000
N_NEG = 100
N_SESSIONS = 400


def scores_with_target_ranks(target_ranks: np.ndarray, seed: int):
    """Build score matrices placing each session's target at a chosen true rank.

    Every session gets a distinct random score permutation of the catalogue; the
    target item for session i is the item sitting at position ``target_ranks[i]``
    in that session's descending order. Returns (scores, targets, seen).
    """
    rng = np.random.default_rng(seed)
    scores = rng.random((N_SESSIONS, M))
    targets = np.empty(N_SESSIONS, dtype=int)
    for i, rank in enumerate(target_ranks):
        order = np.argsort(-scores[i])  # best-first item indices
        targets[i] = order[rank - 1]  # rank is 1-indexed
    seen = np.zeros((N_SESSIONS, M), dtype=bool)  # no history, keeps it clean
    return scores, targets, seen


def full_catalogue_hr10(scores, targets):
    """HR@10 ranking every item — the honest metric."""
    hits = []
    for row in range(len(targets)):
        ranked = np.argsort(-scores[row])
        hits.append(hit_rate_at_k(ranked, {int(targets[row])}, 10))
    return float(np.mean(hits))


def sampled_hr10(scores, targets, seen, seed):
    per = sampled_metrics_from_scores(
        scores, targets, seen, n_negatives=N_NEG, ks=(10,), sampler="uniform", seed=seed
    )
    return float(np.mean(per[("hit_rate", 10)]))


def test_sampled_evaluation_reverses_model_ranking():
    half = N_SESSIONS // 2
    ranks_a = np.concatenate([np.full(half, 5), np.full(N_SESSIONS - half, 5000)])
    ranks_b = np.full(N_SESSIONS, 12)

    sa, ta, seen_a = scores_with_target_ranks(ranks_a, seed=1)
    sb, tb, seen_b = scores_with_target_ranks(ranks_b, seed=2)

    full_a = full_catalogue_hr10(sa, ta)
    full_b = full_catalogue_hr10(sb, tb)
    samp_a = sampled_hr10(sa, ta, seen_a, seed=10)
    samp_b = sampled_hr10(sb, tb, seen_b, seed=20)

    # Full-catalogue: A clearly beats B (0.5 vs 0.0).
    assert full_a > full_b + 0.3
    assert full_b == 0.0

    # Sampled: B clearly beats A — the reversal.
    assert samp_b > samp_a + 0.3

    # State the flip as the property under test, not just the two inequalities.
    full_winner = "A" if full_a > full_b else "B"
    sampled_winner = "A" if samp_a > samp_b else "B"
    assert full_winner != sampled_winner


def test_sampled_cannot_distinguish_near_miss_from_hit():
    # The mechanism behind the reversal: a rank-12 target (a full-catalogue miss
    # at k=10) reads as a hit under sampling most of the time.
    ranks = np.full(N_SESSIONS, 12)
    scores, targets, seen = scores_with_target_ranks(ranks, seed=3)
    assert full_catalogue_hr10(scores, targets) == 0.0
    assert sampled_hr10(scores, targets, seen, seed=30) > 0.8


def test_full_catalogue_recovers_true_ordering_of_clean_models():
    # Sanity companion: when one model genuinely ranks targets higher everywhere,
    # full-catalogue evaluation prefers it — the honest metric behaves.
    good, _, _ = scores_with_target_ranks(np.full(N_SESSIONS, 3), seed=4)
    bad, tb, _ = scores_with_target_ranks(np.full(N_SESSIONS, 400), seed=5)
    gt = np.array([np.argsort(-good[i])[2] for i in range(N_SESSIONS)])
    assert full_catalogue_hr10(good, gt) > full_catalogue_hr10(bad, tb)
