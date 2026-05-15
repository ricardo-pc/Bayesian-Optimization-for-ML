"""
BO runner for Upworthy zero-shot classification threshold tuning.

Unlike the NCF/MF runners, this does NOT need a GPU or SLURM — each trial
takes milliseconds (it's pure pandas + scipy, no model inference).  Run it
locally on your laptop:

    python jobs/run_bo_upworthy.py

All three methods are run back-to-back and saved to separate CSVs so the
notebook can compare convergence curves:
  results/upworthy/bo_trials.csv      — Bayesian Optimisation
  results/upworthy/random_trials.csv  — random search baseline
  results/upworthy/grid_trials.csv    — grid search baseline

After this script finishes, open notebooks/02_upworthy_bo.ipynb for plots.

Typical runtime: ~60 seconds total (all three methods, 60 trials each).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bo import BayesianOptimizer
from src.black_box_upworthy import (
    UpworthyBlackBox,
    UPWORTHY_PARAM_SPACE,
    decode_params,
    best_upworthy_params,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run BO (+ baselines) for Upworthy classification threshold tuning"
    )
    p.add_argument(
        "--upworthy-repo",
        type=str,
        default=None,
        help=(
            "Optional path to the A-B-testing-analysis-upworthy repo root. "
            "By default the bundled CSVs in data/upworthy/ are used."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "upworthy"),
        help="Directory where trial CSVs are written",
    )
    p.add_argument("--budget",    type=int, default=60, help="Evaluations per method")
    p.add_argument("--n-init",    type=int, default=10, help="Random init before BO")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["bo", "random", "grid", "all"],
        help="Which method(s) to run (default: all)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Grid search helper
# ---------------------------------------------------------------------------

def run_grid_search(
    black_box: UpworthyBlackBox,
    results_dir: Path,
    budget: int,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluate a uniform grid over the 2D search space.

    Grid size is chosen as the largest n x n grid with n^2 <= budget.
    For budget=60 this gives a 7x7 = 49 point grid.
    """
    import math as _math
    n_side = int(_math.sqrt(budget))    # e.g. 7 for budget=60

    # Param bounds
    thresh_lo, thresh_hi = UPWORTHY_PARAM_SPACE[0]["bounds"]
    log_mcs_lo, log_mcs_hi = UPWORTHY_PARAM_SPACE[1]["bounds"]

    thresholds   = np.linspace(thresh_lo,   thresh_hi,   n_side)
    log_min_cats = np.linspace(log_mcs_lo, log_mcs_hi, n_side)

    records = []
    best_so_far = -np.inf

    # Override results dir so grid trials go to a separate CSV
    black_box.results_dir = results_dir
    black_box._csv_path   = results_dir / "grid_trials.csv"
    black_box._trial_count = 0
    if black_box._csv_path.exists():
        black_box._csv_path.unlink()

    for i, thresh in enumerate(thresholds):
        for j, lmcs in enumerate(log_min_cats):
            params = {
                "confidence_threshold": float(thresh),
                "log_min_cat_size":     float(lmcs),
            }
            f_stat = black_box(params)
            best_so_far = max(best_so_far, f_stat)
            records.append({
                "trial":                black_box._trial_count,
                "confidence_threshold": thresh,
                "log_min_cat_size":     lmcs,
                "min_cat_size":         decode_params(params)["min_cat_size"],
                "f_stat":               f_stat,
                "best_so_far":          best_so_far,
                "phase":                "grid",
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\nUpworthy repo : {args.upworthy_repo}")
    print(f"Results dir   : {args.results_dir}")
    print(f"Budget        : {args.budget} trials per method")
    print(f"n_init        : {args.n_init}")
    print(f"Seed          : {args.seed}")
    print(f"Method(s)     : {args.method}\n")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Bayesian Optimisation
    # -------------------------------------------------------------------
    if args.method in ("bo", "all"):
        print("=" * 55)
        print("METHOD 1: Bayesian Optimisation")
        print("=" * 55)

        bo_box = UpworthyBlackBox(
            upworthy_repo=args.upworthy_repo,
            results_dir=results_dir,
            verbose=True,
        )
        # Route to bo-specific CSV
        bo_box._csv_path = results_dir / "bo_trials.csv"
        if bo_box._csv_path.exists():
            bo_box._csv_path.unlink()
        bo_box._trial_count = 0

        bo = BayesianOptimizer(
            objective   = bo_box,
            param_space = UPWORTHY_PARAM_SPACE,
            n_init      = args.n_init,
            budget      = args.budget,
            seed        = args.seed,
            verbose     = True,
        )
        bo_results = bo.run()

        best = best_upworthy_params(bo)
        print("\n" + "=" * 55)
        print("BO COMPLETE")
        print(f"  Best F-statistic      : {bo.best_value:.4f}")
        print(f"  confidence_threshold  : {best['confidence_threshold']:.3f}")
        print(f"  min_cat_size          : {best['min_cat_size']}")
        print("=" * 55 + "\n")

    # -------------------------------------------------------------------
    # Random search baseline
    # -------------------------------------------------------------------
    if args.method in ("random", "all"):
        print("=" * 55)
        print("METHOD 2: Random Search Baseline")
        print("=" * 55)

        rng = np.random.default_rng(args.seed + 1)
        lo  = np.array([p["bounds"][0] for p in UPWORTHY_PARAM_SPACE])
        hi  = np.array([p["bounds"][1] for p in UPWORTHY_PARAM_SPACE])
        names = [p["name"] for p in UPWORTHY_PARAM_SPACE]

        rand_box = UpworthyBlackBox(
            upworthy_repo=args.upworthy_repo,
            results_dir=results_dir,
            verbose=False,           # quieter — random search is boring to watch
        )
        rand_box._csv_path    = results_dir / "random_trials.csv"
        if rand_box._csv_path.exists():
            rand_box._csv_path.unlink()
        rand_box._trial_count = 0

        rand_records = []
        best_so_far  = -np.inf

        for t in range(args.budget):
            x = rng.uniform(lo, hi)
            params = {name: float(x[i]) for i, name in enumerate(names)}
            f_stat = rand_box(params)
            best_so_far = max(best_so_far, f_stat)
            rand_records.append({
                "trial":                t + 1,
                "confidence_threshold": params["confidence_threshold"],
                "log_min_cat_size":     params["log_min_cat_size"],
                "min_cat_size":         decode_params(params)["min_cat_size"],
                "f_stat":               f_stat,
                "best_so_far":          best_so_far,
                "phase":                "random",
            })
            if (t + 1) % 10 == 0:
                print(f"  random trial {t+1:3d} | best so far F = {best_so_far:.4f}")

        rand_df = pd.DataFrame(rand_records)
        print(f"\nRandom search best F : {rand_df['f_stat'].max():.4f}")

    # -------------------------------------------------------------------
    # Grid search baseline
    # -------------------------------------------------------------------
    if args.method in ("grid", "all"):
        print("\n" + "=" * 55)
        print("METHOD 3: Grid Search Baseline")
        print("=" * 55)

        grid_box = UpworthyBlackBox(
            upworthy_repo=args.upworthy_repo,
            results_dir=results_dir,
            verbose=False,
        )
        grid_df = run_grid_search(grid_box, results_dir, args.budget, args.seed)
        print(f"\nGrid search best F : {grid_df['f_stat'].max():.4f}")

    # -------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------
    if args.method == "all":
        print("\n" + "=" * 55)
        print("SUMMARY")
        print("=" * 55)
        print(f"  BO best F         : {bo.best_value:.4f}")
        print(f"  Random best F     : {rand_df['f_stat'].max():.4f}")
        print(f"  Grid best F       : {grid_df['f_stat'].max():.4f}")
        print(f"\nFull results in   : {results_dir}")
        print("Open notebooks/02_upworthy_bo.ipynb for plots and analysis.")


if __name__ == "__main__":
    main()
