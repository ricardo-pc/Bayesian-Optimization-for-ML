"""
run_mf_bo.py — Runner script for MF Bayesian Optimisation

MF is fast enough to run on CPU. Run this on your MacBook overnight
while NCF BO runs on SCF.

Usage:
  # dry run (test the loop)
  python run_mf_bo.py --cs289-repo ~/path/to/cs289-ranking --dry-run

  # real run on MacBook (CPU or Apple Silicon MPS)
  python run_mf_bo.py --cs289-repo ~/path/to/cs289-ranking --device cpu --budget 20
  python run_mf_bo.py --cs289-repo ~/path/to/cs289-ranking --device mps --budget 20
"""

import argparse
from pathlib import Path

import pandas as pd

from src.bo import BayesianOptimizer
from src.black_box_mf import MFBlackBox, MF_PARAM_SPACE, best_mf_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs289-repo", type=str, required=True)
    parser.add_argument("--budget",     type=int, default=20)
    parser.add_argument("--n-init",     type=int, default=5)
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--seed",       type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cs289_repo = Path(args.cs289_repo).expanduser().resolve()
    print(f"cs289-ranking repo : {cs289_repo}")
    print(f"device             : {args.device}")
    print(f"budget             : {args.budget}  (n_init={args.n_init})")
    print(f"dry_run            : {args.dry_run}\n")

    black_box = MFBlackBox(
        cs289_repo = cs289_repo,
        device     = args.device,
        epochs     = args.epochs,
        dry_run    = args.dry_run,
        verbose    = True,
    )

    bo = BayesianOptimizer(
        objective   = black_box,
        param_space = MF_PARAM_SPACE,
        n_init      = args.n_init,
        budget      = args.budget,
        seed        = args.seed,
        verbose     = True,
    )

    results: pd.DataFrame = bo.run()

    print("\nBO complete.")
    print(f"Best val NDCG@10 : {bo.best_value:.5f}")
    print(f"Best config      : {best_mf_params(bo)}")
    print(f"\nFull trial log saved to results/mf/trials.csv")
    print(results[["trial", "objective", "best_so_far", "runtime_s", "phase"]].to_string(index=False))


if __name__ == "__main__":
    main()
