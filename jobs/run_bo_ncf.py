"""
Standalone BO runner for NCF hyperparameter tuning.

Run this from an interactive SLURM GPU session — NOT from the login node.

Usage (on SCF GPU node after srun):
    conda activate cs289-ranking-gpu
    cd ~/Bayesian-Optimization-for-ML        # or wherever this repo lives on SCF
    python jobs/run_bo_ncf.py

Results are saved to results/ncf/trials.csv after every single trial,
so the run is safe to interrupt and resume (see --resume flag below).

After the run finishes, load results/ncf/trials.csv in 01_ncf_bo.ipynb
for all plots and analysis.
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Make sure src/ is importable regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bo import BayesianOptimizer
from src.black_box_ncf import NCFBlackBox, NCF_PARAM_SPACE, best_ncf_params, decode_params


# ---------------------------------------------------------------------------
# Configuration — edit these or override via CLI flags
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run BO for NCF hyperparameter tuning")
    p.add_argument("--cs289-repo", type=str,
                   default=os.path.expanduser("~/cs289-ranking"),
                   help="Path to cs289-ranking repo root (default: ~/cs289-ranking)")
    p.add_argument("--device", type=str, default="cuda",
                   help="PyTorch device: cuda | mps | cpu")
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs per trial (default 20; use 5 for a timing test)")
    p.add_argument("--n-init", type=int, default=5,
                   help="Random initialisations before BO starts")
    p.add_argument("--budget", type=int, default=28,
                   help="Total evaluations including n_init")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=str,
                   default=str(REPO_ROOT / "results" / "ncf"),
                   help="Where to write trials.csv")
    p.add_argument("--timing-test", action="store_true",
                   help="Run 2 trials with --epochs 3 just to measure wall time per trial")
    return p.parse_args()


def main():
    args = parse_args()

    if args.timing_test:
        print("=" * 55)
        print("TIMING TEST MODE: 2 trials, 3 epochs each")
        print("=" * 55)
        args.epochs = 3
        args.n_init = 2
        args.budget = 2

    print(f"\nCS289 repo : {args.cs289_repo}")
    print(f"Device     : {args.device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Budget     : {args.budget}  ({args.n_init} random + {args.budget - args.n_init} BO)")
    print(f"Results    : {args.results_dir}")
    print(f"Seed       : {args.seed}")

    # Verify GPU is accessible before wasting time on random init
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("\n[ERROR] CUDA not available. Are you on a GPU node?")
            print("  Run:  srun --partition=jsteinhardt --gres=gpu:1 --mem=32G --time=06:00:00 --pty bash")
            sys.exit(1)
        print(f"\nGPU       : {torch.cuda.get_device_name(0)}")
        print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()

    # Remove poisoned CSV if it exists and has only zero-NDCG rows
    results_csv = Path(args.results_dir) / "trials.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows and all(float(r["val_ndcg"]) == 0.0 for r in rows):
            print(f"[WARNING] Found {len(rows)} trials all with NDCG=0 in {results_csv}")
            print("          Looks like a failed cuda run. Deleting and starting fresh.\n")
            results_csv.unlink()

    # ---------------------------------------------------------------------------
    # Black box
    # ---------------------------------------------------------------------------
    black_box = NCFBlackBox(
        cs289_repo  = args.cs289_repo,
        results_dir = args.results_dir,
        device      = args.device,
        epochs      = args.epochs,
        verbose     = True,
    )

    # ---------------------------------------------------------------------------
    # BO loop
    # ---------------------------------------------------------------------------
    bo = BayesianOptimizer(
        objective   = black_box,
        param_space = NCF_PARAM_SPACE,
        n_init      = args.n_init,
        budget      = args.budget,
        seed        = args.seed,
        verbose     = True,
    )

    bo_results = bo.run()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    best = best_ncf_params(bo)

    print("\n" + "=" * 55)
    print("BO COMPLETE")
    print("=" * 55)
    print(f"Best val NDCG@10 : {bo.best_value:.5f}")
    print(f"Best config (decoded):")
    for k, v in best.items():
        print(f"  {k:15s}: {v}")
    print()

    # Print the train.py commands for the CS 289A sparsity sweep
    mlp_str = " ".join(str(h) for h in best["mlp_layers"])
    print("CS 289A sparsity sweep — copy these commands:")
    print()
    for density in [1.0, 0.8, 0.6, 0.4, 0.2]:
        print(f"# density={density}")
        print(
            f"python src/train.py --model ncf --density {density} "
            f"--emb-dim {best['emb_dim']} --mlp-layers {mlp_str} "
            f"--lr {best['lr']:.2e} --l2 {best['l2']:.2e} --alpha {best['alpha']:.4f} "
            f"--epochs 20 --device {args.device}"
        )
        print()

    print(f"Full trial log: {results_csv}")


if __name__ == "__main__":
    main()
