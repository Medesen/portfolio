"""CLI entry point: ``demandcast backtest --model seasonal_naive``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demandcast.data import load_long
from demandcast.evaluation import make_folds, run_backtest, score
from demandcast.models import MODELS, LgbmForecaster, Sarimax, select_skus


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="demandcast",
        description="Demand forecasting and promo-effect estimation on daily pasta sales.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Rolling-origin backtest of one model")
    bt.add_argument("--model", choices=sorted(MODELS), required=True)
    bt.add_argument("--n-folds", type=int, default=12)
    bt.add_argument("--horizon", type=int, default=28, help="trading days ahead")
    bt.add_argument(
        "--subset",
        choices=["all", "sarimax"],
        default=None,
        help="restrict to the SARIMAX SKU subset for apples-to-apples "
        "comparison (default: all, except for --model sarimax which "
        "always runs on its subset)",
    )
    bt.add_argument(
        "--train-scope",
        choices=["subset", "global"],
        default="subset",
        help="when evaluating on a subset with --model lgbm: train on the "
        "subset itself (default) or on the full 118-SKU panel (global "
        "cross-learning, still evaluated on the subset only)",
    )
    bt.add_argument(
        "--objective",
        choices=["tweedie", "poisson", "l2"],
        default=None,
        help="LightGBM point-forecast objective for the DATA_NOTES §2 ablation "
        "(lgbm only; default tweedie). poisson/l2 runs skip the quantile "
        "companions, which are identical across objectives",
    )
    bt.add_argument("--out", type=Path, default=Path("outputs"))

    pl = sub.add_parser(
        "promo-lift", help="Fixed-effects promo-lift estimation (PPML + OLS robustness)"
    )
    pl.add_argument("--out", type=Path, default=Path("outputs"))

    pf = sub.add_parser(
        "plot-forecast", help="Plot one SKU's 28-day forecast with P10-P90 band"
    )
    pf.add_argument(
        "--preds", type=Path, default=Path("outputs/backtest_lgbm_preds.csv"),
        help="predictions CSV from `backtest --model lgbm`",
    )
    pf.add_argument("--sku", default=None, help="default: auto-picked example SKU")
    pf.add_argument("--fold", type=int, default=None, help="default: last fold")
    pf.add_argument("--out", type=Path, default=Path("outputs/forecast_example.png"))

    args = parser.parse_args()

    if args.command == "plot-forecast":
        from demandcast.analysis.plots import plot_forecast

        preds = pd.read_csv(args.preds, parse_dates=["date"])
        out = plot_forecast(preds, load_long(), sku=args.sku, fold=args.fold, out=args.out)
        print(f"wrote {out}")
        return

    if args.command == "promo-lift":
        from demandcast.analysis.promo_lift import estimate_promo_lift

        long = load_long()
        result = estimate_promo_lift(long)
        print(result.ppml)
        print(result.ols_log1p)
        print("\nPPML lift by brand:")
        print(result.by_brand.round(2).to_string())
        args.out.mkdir(parents=True, exist_ok=True)
        result.by_brand.to_csv(args.out / "promo_lift_by_brand.csv")
        return

    if args.command == "backtest":
        long = load_long()

        subset = args.subset or ("sarimax" if args.model == Sarimax.name else "all")
        if subset == "sarimax":
            long_eval = long[long["sku"].isin(select_skus(long))]
        else:
            long_eval = long

        if args.objective and args.model != LgbmForecaster.name:
            parser.error("--objective only applies to --model lgbm")
        objective = args.objective or "tweedie"
        if args.train_scope == "global" and (
            args.model != LgbmForecaster.name or subset == "all"
        ):
            parser.error(
                "--train-scope global requires --model lgbm and a subset "
                "evaluation (it widens the training pool beyond the subset)"
            )

        folds = make_folds(long_eval["date"], n_folds=args.n_folds, horizon=args.horizon)
        if args.model == LgbmForecaster.name:
            # full frame = promo/calendar schedule only
            model = LgbmForecaster(
                full_long=long,
                objective=objective,
                quantiles=(0.1, 0.5, 0.9) if objective == "tweedie" else (),
            )
        else:
            model = MODELS[args.model](long)
        preds = run_backtest(
            long_eval, model, folds,
            train_long=long if args.train_scope == "global" else None,
        )
        scores = score(preds, long_eval)

        model_tag = model.name
        if args.model == LgbmForecaster.name and objective != "tweedie":
            model_tag = f"{model.name}-{objective}"
        tag = model_tag if subset == "all" else f"{model_tag}_subset-{subset}"
        scope_note = ""
        if args.train_scope == "global":
            tag += "_train-global"
            scope_note = ", trained on the full panel"
        args.out.mkdir(parents=True, exist_ok=True)
        preds.to_csv(args.out / f"backtest_{tag}_preds.csv", index=False)
        scores.to_csv(args.out / f"backtest_{tag}_scores.csv")

        print(
            f"\n{model_tag} — {args.n_folds} folds x {args.horizon} trading days"
            f" — {long_eval['sku'].nunique()} SKUs ({subset}{scope_note})"
        )
        print(scores.round(3).to_string())

        if any(c.startswith("y_q") for c in preds.columns):
            from demandcast.evaluation.metrics import score_quantiles

            qs = score_quantiles(preds)
            print("\nquantile forecasts:")
            print(qs.round(3).to_string())
            cov = qs.attrs.get("coverage_p10_p90")
            if cov is not None:
                print(f"P10-P90 empirical coverage: {cov:.3f} (target 0.80)")


if __name__ == "__main__":
    main()
