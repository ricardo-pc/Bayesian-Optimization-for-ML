"""
run_ncf_bo.py — Runner script for NCF Bayesian Optimisation

Run from the STAT 238 repo root. Points at the cs289-ranking repo via --cs289-repo.

Usage:
  # dry run (no GPU needed — returns fake NDCG values, tests the loop)
  python run_ncf_bo.py --cs289-repo ~/path/to/cs289-ranking --dry-run

  # real run on SCF GPU
  python run_ncf_bo.py --cs289-repo ~/cs289-ranking --device cuda --budget 30

  # real run on CPU (slow — for MF only, not NCF)
  python run_ncf_bo.py --cs289-repo ~/cs289-ranking --device cpu --budget 20
"""

import argparse
from pathlib import Path

import pandas as pd

from src.bo import BayesianOptimizer
from src.black_box_ncf import NCFBlackBox, NCF_PARAM_SPACE, best_ncf_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs289-repo", type=str, required=True,
                        help="absolute path to the cs289-ranking repo root")
    parser.add_argument("--budget",     type=int, default=30,
                        help="total number of BO trials (including random init)")
    parser.add_argument("--n-init",     type=int, default=5,
                        help="number of random initial trials before BO starts")
    parser.add_argument("--epochs",     type=int, default=20,
                        help="training epochs per trial")
    parser.add_argument("--device",     type=str, default="cuda",
                        help="cuda / mps / cpu")
    parser.add_argument("--dry-run",    action="store_true",
                        help="skip actual training — returns fake NDCG for testing")
    parser.add_argument("--seed",       type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cs289_repo = Path(args.cs289_repo).expanduser().resolve()
    print(f"cs289-ranking repo : {cs289_repo}")
    print(f"device             : {args.device}")
    print(f"budget             : {args.budget}  (n_init={args.n_init})")
    print(f"dry_run            : {args.dry_run}\n")

    # Black-box objective: wraps train.py, returns val NDCG@10
    black_box = NCFBlackBox(
        cs289_repo  = cs289_repo,
        device      = args.device,
        epochs      = args.epochs,
        dry_run     = args.dry_run,
        verbose     = True,
    )

    # BO loop
    bo = BayesianOptimizer(
        objective   = black_box,
        param_space = NCF_PARAM_SPACE,
        n_init      = args.n_init,
        budget      = args.budget,
        seed        = args.seed,
        verbose     = True,
    )

    results: pd.DataFrame = bo.run()

    # Summary
    print("\nBO complete.")
    print(f"Best val NDCG@10 : {bo.best_value:.5f}")
    print(f"Best config      : {best_ncf_params(bo)}")
    print(f"\nFull trial log saved to results/ncf/trials.csv")
    print(results[["trial", "objective", "best_so_far", "runtime_s", "phase"]].to_string(index=False))


if __name__ == "__main__":
    main()
