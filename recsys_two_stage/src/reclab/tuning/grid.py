"""Hyperparameter tuning on a temporal validation window.

The entire premise of Dacrema et al. (2019) is that classical baselines lose
only when they are left untuned. So every model here gets a real grid search —
and it is run on a validation window that is *itself* a temporal hold-out carved
from the end of the training period, never on the test split.

The nesting:

    |<----------- train ----------->|<-- val -->|<--- test --->|
    |<-------- tune here --------------------->|              |
    start                          v_cut       cutoff        end

A model's hyperparameters are chosen to maximise validation NDCG, then the model
is refit on the *whole* training period (train + val) with those settings and
evaluated once on test. The test split plays no part in selection, so the
reported numbers carry no winner's-curse bias.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from reclab.evaluation.full_catalogue import evaluate
from reclab.splitting.protocols import SessionSplit, temporal_split


def validation_split(
    session_items: pd.DataFrame,
    test_days: int,
    val_days: int,
    min_session_items: int = 2,
    min_item_sessions: int = 10,
    filter_scope: str = "train",
) -> SessionSplit:
    """A temporal split whose test window sits *inside* the training period.

    Built by discarding the real test period entirely and then applying the same
    temporal-split machinery to what remains, with a ``val_days`` hold-out. Because
    it never sees a post-cutoff row, tuning on it cannot leak the test period.
    """
    cutoff = session_items["ts"].max().normalize() - pd.Timedelta(days=test_days)
    bounds = session_items.groupby("session")["ts"].max()
    train_sessions = set(bounds.index[bounds < cutoff])
    train_only = session_items[session_items["session"].isin(train_sessions)]
    return temporal_split(
        train_only,
        test_days=val_days,
        min_session_items=min_session_items,
        min_item_sessions=min_item_sessions,
        filter_scope=filter_scope,
    )


@dataclass
class TuningResult:
    model: str
    best_params: dict
    best_score: float
    metric: str
    k: int
    table: pd.DataFrame  # every grid point with its validation score

    def __str__(self) -> str:
        params = ", ".join(f"{key}={value}" for key, value in self.best_params.items())
        return (
            f"{self.model}: best {self.metric}@{self.k}={self.best_score:.4f} "
            f"at {params}  ({len(self.table)} configs tried)"
        )


def grid_search(
    model_cls,
    param_grid: dict[str, list],
    val_split: SessionSplit,
    metric: str = "ndcg",
    k: int = 20,
    fixed: dict | None = None,
) -> TuningResult:
    """Exhaustive grid search selecting on validation ``metric``@``k``.

    ``param_grid`` maps each parameter to the list of values to try; ``fixed``
    holds parameters kept constant (seeds, thread counts). Returns the winning
    configuration and a table of every grid point, so the search is auditable
    rather than asserted.
    """
    fixed = fixed or {}
    names = list(param_grid)
    combinations = list(itertools.product(*(param_grid[name] for name in names)))
    if not combinations:
        raise ValueError("empty parameter grid")

    rows = []
    best_score, best_params = -float("inf"), None
    for combo in combinations:
        params = dict(zip(names, combo))
        model = model_cls(**params, **fixed).fit(val_split.train)
        result = evaluate(model, val_split, ks=(k,))
        score = float(result.per_session[(metric, k)].mean())
        rows.append({**params, f"val_{metric}@{k}": score})
        if score > best_score:
            best_score, best_params = score, params
        del model

    model_name = getattr(model_cls, "name", model_cls.__name__)
    return TuningResult(
        model=model_name,
        best_params=best_params,
        best_score=best_score,
        metric=metric,
        k=k,
        table=pd.DataFrame(rows).sort_values(f"val_{metric}@{k}", ascending=False),
    )
