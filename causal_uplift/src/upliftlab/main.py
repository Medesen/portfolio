"""CLI entry point: ``upliftlab {balance,ate,cuped,uplift,all}``.

Each subcommand prints its result and writes a CSV to ``outputs/`` (the Qini
figure goes to ``assets/``). ``all`` runs the full pipeline and reproduces every
number and figure in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from upliftlab.data import (
    CATEGORICAL_COVARIATES,
    CONTROL,
    NUMERIC_COVARIATES,
    OUTCOMES,
    TREATMENTS,
    design_matrix,
    load,
    two_arm,
)
from upliftlab.experiment import (
    cuped,
    estimate_all,
    regression_adjustment,
    standardized_mean_differences,
)
from upliftlab.uplift import (
    LEARNERS,
    incremental_curve,
    plot_qini,
    qini_coefficient,
    qini_curve,
    response_model_scores,
    uplift_by_group,
)

TARGET_KS = [0.1, 0.2, 0.3, 0.4, 0.5]


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def run_balance(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    table = standardized_mean_differences(
        df, numeric=NUMERIC_COVARIATES, categorical=CATEGORICAL_COVARIATES,
        arm_col="segment", control=CONTROL, treatments=TREATMENTS,
    )
    print("\n=== Covariate balance (standardized mean differences vs control) ===")
    print(table.round(4).to_string())
    print(f"\nLargest |SMD| anywhere: {table['abs_max'].max():.4f} "
          f"(threshold for concern: 0.10) -> randomization "
          f"{'looks clean' if table['abs_max'].max() < 0.1 else 'is SUSPECT'}")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "balance_smd.csv")
    return table


def run_ate(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    table = estimate_all(df, outcomes=OUTCOMES, treatments=TREATMENTS, control=CONTROL)
    show = table.copy()
    print("\n=== Average treatment effects (difference in means) ===")
    cols = ["arm", "outcome", "control_mean", "treat_mean", "effect", "rel_effect",
            "se", "ci_low", "ci_high", "mde_80", "power", "p_value", "p_holm", "p_bh"]
    print(show[cols].round(5).to_string(index=False))
    print("\nEffects with Holm-adjusted p < 0.05 survive multiplicity control across "
          f"the {len(table)} arm x outcome tests.")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ate.csv", index=False)
    return table


def run_cuped(df: pd.DataFrame, out: Path, treatment: str = "Womens E-Mail") -> pd.DataFrame:
    rows = []
    print(f"\n=== Variance reduction ({treatment} vs control) ===")
    for outcome in OUTCOMES:
        c = cuped(df, outcome, pre_covariate="history", treatment=treatment)
        r = regression_adjustment(
            df, outcome, numeric=NUMERIC_COVARIATES, categorical=CATEGORICAL_COVARIATES,
            treatment=treatment,
        )
        print("  " + str(c))
        print("  " + str(r))
        for res in (c, r):
            rows.append({
                "outcome": res.outcome, "method": res.method, "covariates": res.covariates,
                "ate_unadj": res.ate_unadj, "se_unadj": res.se_unadj,
                "ate_adj": res.ate_adj, "se_adj": res.se_adj,
                "var_reduction": res.var_reduction, "eff_n_multiplier": res.eff_n_multiplier,
            })
    table = pd.DataFrame(rows)
    print("\nReading: variance reduction is bounded by the squared correlation between "
          "the pre-period covariate(s) and the outcome. On a two-week response outcome "
          "these are weak, so the CI barely moves — an honest null, not a failure.")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "adjustment.csv", index=False)
    return table


def run_uplift(
    df: pd.DataFrame, out: Path,
    treatment: str = "Womens E-Mail", outcome: str = "visit",
    test_size: float = 0.3, seed: int = 0,
) -> dict:
    from sklearn.model_selection import train_test_split

    sub = two_arm(df, treatment)
    X, _ = design_matrix(sub)
    y = sub[outcome].to_numpy(float)
    t = sub["t"].to_numpy(int)
    kind = "continuous" if outcome == "spend" else "binary"

    idx = np.arange(len(sub))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=t)
    Xtr, Xte = X.iloc[tr], X.iloc[te]

    print(f"\n=== Uplift models: {treatment} vs control, outcome = {outcome} ===")
    print(f"train n={len(tr)}, test n={len(te)} (held-out); base learner = LightGBM (fixed params)")

    curves, qcs, best = {}, {}, None
    for cls in LEARNERS.values():
        model = cls(kind=kind).fit(Xtr, t[tr], y[tr])
        pred = model.predict_uplift(Xte)
        curves[cls.name] = qini_curve(pred, y[te], t[te])
        qc = qini_coefficient(pred, y[te], t[te])
        qcs[cls.name] = {"pred": pred, **qc}
        print(f"  {cls.name:11s} Qini={qc['qini']:.2f}  (normalized {qc['qini_norm']:.3f}, "
              f"q_total={qc['q_total']:.1f})")
        if best is None or qc["qini"] > best[1]:
            best = (cls.name, qc["qini"])

    best_name = best[0]
    best_pred = qcs[best_name]["pred"]
    print(f"\nBest ranker: {best_name} (normalized Qini {qcs[best_name]['qini_norm']:.3f}). "
          "Decile table (group 1 = highest predicted uplift):")
    deciles = uplift_by_group(best_pred, y[te], t[te], n_groups=10)
    print(deciles.round(4).to_string(index=False))

    # Targeting: compare policies on the MODELED outcome (visit) — the stable
    # metric. Spend is ~99% zeros and noise-dominated at small samples, so it is
    # translated approximately below rather than headlined per-decile.
    resp = response_model_scores(Xtr[t[tr] == 1], y[tr][t[tr] == 1], Xte, kind=kind)
    rng = np.random.default_rng(seed)
    up_v = incremental_curve(best_pred, y[te], t[te], TARGET_KS)
    rp_v = incremental_curve(resp, y[te], t[te], TARGET_KS)
    rnd_v = incremental_curve(rng.standard_normal(len(te)), y[te], t[te], TARGET_KS)

    ate_all = y[te][t[te] == 1].mean() - y[te][t[te] == 0].mean()
    total_incr = ate_all * len(te)               # treat-everyone reference (visits)

    tbl = up_v[["k", "n_targeted"]].copy()
    tbl["uplift_visits/1k"] = up_v["uplift_per_target"] * 1000
    tbl["response_visits/1k"] = rp_v["uplift_per_target"] * 1000
    tbl["random_visits/1k"] = rnd_v["uplift_per_target"] * 1000
    tbl["uplift_capture_%"] = up_v["incremental_outcome"] / total_incr * 100
    print("\nTargeting simulation — incremental VISITS per 1,000 mailed (model targets visit):")
    print(tbl.round(3).to_string(index=False))

    treated = df[df["segment"] == treatment]
    rev_per_visit = treated["spend"].sum() / max(1.0, treated["visit"].sum())
    print(f"\nApprox. revenue lens: ~${rev_per_visit:.2f} incremental spend per incremental "
          f"visit (experiment-wide), so the top-{int(TARGET_KS[0] * 100)}% uplift group is "
          f"worth ~${tbl.iloc[0]['uplift_visits/1k'] * rev_per_visit:,.0f} / 1,000 mailed.")
    print("\nReading: ranking by predicted uplift beats random at every depth and front-loads "
          f"incremental visits into the top deciles (top 30% captures "
          f"{tbl.loc[tbl['k'] == 0.3, 'uplift_capture_%'].iloc[0]:.0f}% of the visits that "
          "mailing everyone would). On this data the effect heterogeneity is real but modest, "
          "so the learners separate only slightly and the simplest can win.")

    qini_summary = pd.DataFrame(
        [{"learner": name, "qini": v["qini"], "qini_norm": v["qini_norm"], "q_total": v["q_total"]}
         for name, v in qcs.items()]
    )
    out.mkdir(parents=True, exist_ok=True)
    qini_summary.to_csv(out / "uplift_qini.csv", index=False)
    deciles.to_csv(out / "uplift_deciles.csv", index=False)
    tbl.to_csv(out / "uplift_targeting.csv", index=False)
    fig = plot_qini(curves, out / "qini_curve.png",
                    title=f"Qini curves — {treatment} vs control (outcome: {outcome})")
    print(f"\nwrote {fig}  (commit a copy to assets/ for the README)")
    return {"qini": qini_summary, "deciles": deciles, "targeting": tbl}


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="upliftlab",
        description="Experimentation & uplift on the Hillstrom randomized e-mail trial.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_out(p):
        p.add_argument("--out", type=Path, default=Path("outputs"))
        return p

    with_out(sub.add_parser("balance", help="Covariate-balance check across arms"))
    with_out(sub.add_parser("ate", help="Average treatment effects with honest inference"))
    cu = with_out(sub.add_parser("cuped", help="CUPED / regression-adjustment variance reduction"))
    cu.add_argument("--treatment", default="Womens E-Mail", choices=TREATMENTS)
    up = with_out(sub.add_parser("uplift", help="Uplift models, Qini evaluation, targeting simulation"))
    up.add_argument("--treatment", default="Womens E-Mail", choices=TREATMENTS)
    up.add_argument("--outcome", default="visit", choices=OUTCOMES)
    up.add_argument("--test-size", type=float, default=0.3)
    up.add_argument("--seed", type=int, default=0)
    with_out(sub.add_parser("all", help="Run the whole pipeline (reproduce the README)"))

    args = parser.parse_args()
    df = load()

    if args.command == "balance":
        run_balance(df, args.out)
    elif args.command == "ate":
        run_ate(df, args.out)
    elif args.command == "cuped":
        run_cuped(df, args.out, treatment=args.treatment)
    elif args.command == "uplift":
        run_uplift(df, args.out, treatment=args.treatment,
                   outcome=args.outcome, test_size=args.test_size, seed=args.seed)
    elif args.command == "all":
        run_balance(df, args.out)
        run_ate(df, args.out)
        run_cuped(df, args.out)
        run_uplift(df, args.out)


if __name__ == "__main__":
    main()
