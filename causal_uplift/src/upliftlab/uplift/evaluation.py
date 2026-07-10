"""Evaluating uplift models on a randomized test set: Qini, deciles, targeting.

You cannot score an uplift model by accuracy — the individual treatment effect is
never observed. What you *can* do, on held-out randomized data, is check that the
customers the model ranks highest really do respond more to treatment than the
ones it ranks lowest. That is what the Qini curve measures, and what the
decile table and the targeting simulation translate into money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:  # numpy 1.x
    _trapz = np.trapz


def qini_curve(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: cumulative incremental outcomes as customers are added by score.

    Sorting customers from highest to lowest predicted uplift, the curve value at
    depth k is ``Y_t(k) - Y_c(k) * N_t(k)/N_c(k)`` — treated responses minus
    control responses rescaled to the treated count seen so far. Returns
    ``(x, q)`` with ``x`` the fraction of the population targeted (0..1).
    """
    score, y, t = np.asarray(score, float), np.asarray(y, float), np.asarray(t, float)
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]
    ct, cc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - yc * (ct / np.where(cc == 0, np.nan, cc))
    q = np.nan_to_num(q, nan=0.0)
    n = len(y)
    x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    q = np.concatenate([[0.0], q])
    return x, q


def qini_coefficient(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict:
    """Area between the model's Qini curve and the random-targeting diagonal.

    Positive means the ranking beats random. ``q_total`` is the endpoint
    (total incremental outcomes at full targeting); ``qini_norm`` expresses the
    coefficient as a fraction of a perfect-ranking reference for interpretability.
    """
    x, q = qini_curve(score, y, t)
    area_model = _trapz(q, x)
    q_total = q[-1]
    area_random = 0.5 * q_total                       # triangle under the diagonal
    perfect = _perfect_area(score, y, t, q_total)
    denom = perfect - area_random
    return {
        "qini": float(area_model - area_random),
        "area_model": float(area_model),
        "area_random": float(area_random),
        "q_total": float(q_total),
        "qini_norm": float((area_model - area_random) / denom) if denom > 0 else float("nan"),
    }


def _perfect_area(score, y, t, q_total) -> float:
    """Area under an oracle Qini curve (upper bound) — used only to normalise."""
    # An oracle that front-loads all incremental response: rises to q_total as
    # fast as the treated responders allow, then stays flat. Approximated by
    # sorting on the realised (treated-response minus control-baseline) signal.
    t = np.asarray(t, float)
    base = np.asarray(y, float) - y[t == 0].mean()
    oracle = np.where(t == 1, base, -base)
    x, q = qini_curve(oracle, y, t)
    return _trapz(q, x)


def uplift_by_group(score, y, t, n_groups: int = 10) -> pd.DataFrame:
    """Observed uplift within score-ranked groups (group 1 = highest predicted).

    A well-ranked model produces a roughly monotone decreasing observed-uplift
    column — the empirical check that the score means something.
    """
    df = pd.DataFrame({"score": np.asarray(score, float), "y": np.asarray(y, float), "t": np.asarray(t, float)})
    # Rank descending, then cut into equal groups; guard against ties collapsing bins.
    df["rank"] = df["score"].rank(method="first", ascending=False)
    df["group"] = pd.qcut(df["rank"], q=n_groups, labels=range(1, n_groups + 1))
    rows = []
    for g, gdf in df.groupby("group", observed=True):
        yt, yc = gdf.loc[gdf.t == 1, "y"], gdf.loc[gdf.t == 0, "y"]
        rows.append(
            {
                "group": int(g),
                "n": len(gdf),
                "pred_uplift": gdf["score"].mean(),
                "obs_uplift": (yt.mean() - yc.mean()) if len(yt) and len(yc) else np.nan,
                "n_treat": len(yt),
                "n_control": len(yc),
            }
        )
    return pd.DataFrame(rows)


def incremental_curve(score, y, t, ks, value: np.ndarray | None = None) -> pd.DataFrame:
    """Incremental outcome captured by targeting the top-k fraction by ``score``.

    For the top-k set, the randomized-within-subset estimate of the policy's
    incremental effect is ``(mean treated − mean control) × size``. If ``value``
    is given (e.g. spend), the incremental is additionally valued on that scale.
    """
    score = np.asarray(score, float)
    y, t = np.asarray(y, float), np.asarray(t, float)
    value = y if value is None else np.asarray(value, float)
    order = np.argsort(-score, kind="mergesort")
    n = len(score)
    rows = []
    for k in ks:
        m = max(1, int(round(k * n)))
        idx = order[:m]
        ti = t[idx]
        yt, yc = y[idx][ti == 1], y[idx][ti == 0]
        vt, vc = value[idx][ti == 1], value[idx][ti == 0]
        up = (yt.mean() - yc.mean()) if len(yt) and len(yc) else np.nan
        upv = (vt.mean() - vc.mean()) if len(vt) and len(vc) else np.nan
        rows.append(
            {
                "k": k,
                "n_targeted": m,
                "uplift_per_target": up,
                "incremental_outcome": up * m,
                "value_per_target": upv,
                "incremental_value": upv * m,
            }
        )
    return pd.DataFrame(rows)


def plot_qini(curves: dict[str, tuple[np.ndarray, np.ndarray]], out, title: str) -> str:
    """Plot one Qini curve per model plus the random-targeting diagonal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    q_total = None
    for name, (x, q) in curves.items():
        ax.plot(x * 100, q, label=name, linewidth=2)
        q_total = q[-1]
    if q_total is not None:
        ax.plot([0, 100], [0, q_total], "k--", linewidth=1, label="random targeting")
    ax.set_xlabel("% of customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("cumulative incremental visits")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
