"""
NCF black-box objective for Bayesian Optimisation.

Interface between BayesianOptimizer (src/bo.py) and the CS 289A training
pipeline (cs289-ranking/src/train.py).  See documents/ncf_bridge.md for the
full interface specification.

How it fits in the BO loop
--------------------------
BayesianOptimizer calls  objective(params)  at every trial, where params is
a dict keyed by the names in NCF_PARAM_SPACE (the encoded/GP-space values).

NCFBlackBox.__call__:
  1. Decodes the encoded GP-space vector to real NCF hyperparameters.
  2. Shells out to  python src/train.py  in the cs289-ranking repo.
  3. Parses the final stdout line to extract val NDCG@10.
  4. Appends the trial to  results/ncf/trials.csv.
  5. Returns NDCG@10 (higher = better — BO maximises).

Parameter encoding  (ncf_bridge.md, "Encoding Categorical Parameters")
-----------------------------------------------------------------------
The GP kernel requires a continuous input vector.  Categorical / ordinal
parameters are mapped to real numbers before being passed to the GP:

  emb_dim_x  ∈ [0, 3]         → {0:32, 1:64, 2:128, 3:256}   (round then lookup)
  mlp_x      ∈ [0, 2]         → one of three MLP configs       (round then lookup)
  log_lr     ∈ [log 1e-4, log 1e-2]  → lr = exp(log_lr)
  log_l2     ∈ [log 1e-6, log 1e-3]  → l2 = exp(log_l2)
  alpha      ∈ [0.5, 5.0]     → as-is  (WMF confidence scale)

Fixed (not tuned by BO, matches ncf_bridge.md):
  --model ncf  --density 1.0  --epochs 20  --batch-size 1024  --n-neg 4

Output format from train.py (last two stdout lines):
  Best val  NDCG@10 = 0.4821  HR@10 = 0.7103
  Checkpoint saved to checkpoints/ncf_density1.0.pt
"""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Search space definition  (pass this to BayesianOptimizer as param_space)
# ---------------------------------------------------------------------------

NCF_PARAM_SPACE = [
    # emb_dim: ordinal encoding — 0→32, 1→64, 2→128, 3→256
    {"name": "emb_dim_x", "type": "continuous", "bounds": (0.0, 3.0)},
    # mlp_layers: ordinal by depth — 0→[128,64], 1→[256,128,64], 2→[256,128,64,32]
    {"name": "mlp_x",     "type": "continuous", "bounds": (0.0, 2.0)},
    # learning rate in log scale: exp maps [-9.21, -4.61] → [1e-4, 1e-2]
    {"name": "log_lr",    "type": "continuous", "bounds": (math.log(1e-4), math.log(1e-2))},
    # L2 weight decay in log scale: exp maps [-13.82, -6.91] → [1e-6, 1e-3]
    {"name": "log_l2",    "type": "continuous", "bounds": (math.log(1e-6), math.log(1e-3))},
    # WMF confidence scale: c_ui = 1 + alpha * rating
    {"name": "alpha",     "type": "continuous", "bounds": (0.5, 5.0)},
]

# ---------------------------------------------------------------------------
# Decoding tables
# ---------------------------------------------------------------------------

_EMB_DIM_MAP: dict[int, int] = {0: 32, 1: 64, 2: 128, 3: 256}

_MLP_MAP: dict[int, list[int]] = {
    0: [128, 64],
    1: [256, 128, 64],
    2: [256, 128, 64, 32],
}


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def decode_params(encoded: dict) -> dict:
    """
    Convert the GP-space encoded dict to actual NCF hyperparameters.

    Parameters
    ----------
    encoded : dict
        Keys: emb_dim_x, mlp_x, log_lr, log_l2, alpha  (from BayesianOptimizer)

    Returns
    -------
    dict with keys: emb_dim (int), mlp_layers (list[int]),
                    lr (float), l2 (float), alpha (float)
    """
    emb_idx = int(round(float(encoded["emb_dim_x"])))
    emb_idx = max(0, min(3, emb_idx))   # clamp to valid range
    emb_dim = _EMB_DIM_MAP[emb_idx]

    mlp_idx = int(round(float(encoded["mlp_x"])))
    mlp_idx = max(0, min(2, mlp_idx))
    mlp_layers = _MLP_MAP[mlp_idx]

    lr  = float(np.exp(encoded["log_lr"]))
    l2  = float(np.exp(encoded["log_l2"]))
    alpha = float(encoded["alpha"])

    return {
        "emb_dim":   emb_dim,
        "mlp_layers": mlp_layers,
        "lr":        lr,
        "l2":        l2,
        "alpha":     alpha,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NCFBlackBox:
    """
    Callable black-box objective for NCF hyperparameter tuning.

    Wraps  cs289-ranking/src/train.py  and returns val NDCG@10.

    Parameters
    ----------
    cs289_repo : str or Path
        Absolute path to the cs289-ranking repo root.
        train.py is expected at  <cs289_repo>/src/train.py.
    results_dir : str or Path
        Directory where  trials.csv  is written after each evaluation.
        Defaults to  results/ncf/  relative to this repo.
    device : str
        PyTorch device string: 'cuda', 'mps', or 'cpu'.
        Use 'cuda' on SCF — do not run BO with 'cpu', each trial takes ~50 min.
    epochs : int
        Number of training epochs per trial.  20 matches the proposal.
    data_dir : str or None
        Path to the ml-1m raw data folder inside the cs289-ranking repo.
        If None, defaults to  <cs289_repo>/data/raw/ml-1m.
    dry_run : bool
        If True, skip the actual subprocess call and return a random NDCG in
        [0.35, 0.55].  Use for testing the BO loop without a GPU.
    verbose : bool
        Print decoded params and subprocess stdout for each trial.
    penalty : float
        NDCG value returned when train.py fails (so BO can continue safely).
        Defaults to 0.0.
    """

    def __init__(
        self,
        cs289_repo: str | Path,
        results_dir: str | Path | None = None,
        device: str = "cuda",
        epochs: int = 20,
        data_dir: str | None = None,
        dry_run: bool = False,
        verbose: bool = True,
        penalty: float = 0.0,
    ) -> None:
        self.cs289_repo = Path(cs289_repo).resolve()
        self.device     = device
        self.epochs     = epochs
        self.dry_run    = dry_run
        self.verbose    = verbose
        self.penalty    = penalty

        if results_dir is None:
            # Default: results/ncf/ relative to this file's grandparent (repo root)
            results_dir = Path(__file__).resolve().parent.parent / "results" / "ncf"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.results_dir / "trials.csv"
        self._trial_count = self._count_existing_trials()

        if data_dir is None:
            self._data_dir = str(self.cs289_repo / "data" / "raw" / "ml-1m")
        else:
            self._data_dir = data_dir

        # Validate repo exists
        train_script = self.cs289_repo / "src" / "train.py"
        if not self.dry_run and not train_script.exists():
            raise FileNotFoundError(
                f"train.py not found at {train_script}.\n"
                f"Set cs289_repo to the correct path or use dry_run=True for testing."
            )

    def _count_existing_trials(self) -> int:
        """Read how many trials are already in trials.csv (for numbering continuations)."""
        if not self._csv_path.exists():
            return 0
        with open(self._csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)   # subtract header

    def __call__(self, encoded: dict) -> float:
        """
        Evaluate one NCF configuration and return val NDCG@10.

        Parameters
        ----------
        encoded : dict
            Encoded GP-space parameter dict from BayesianOptimizer.
            Keys: emb_dim_x, mlp_x, log_lr, log_l2, alpha.

        Returns
        -------
        float : val NDCG@10  (higher is better)
        """
        self._trial_count += 1
        decoded = decode_params(encoded)

        if self.verbose:
            print(f"\n[NCF trial {self._trial_count}]  decoded: {decoded}")

        t0 = time.perf_counter()

        if self.dry_run:
            # Return a plausible fake NDCG for testing the BO loop locally
            rng = np.random.default_rng(self._trial_count)
            ndcg = float(rng.uniform(0.35, 0.55))
            hr   = float(rng.uniform(0.65, 0.80))
            if self.verbose:
                print(f"  [dry_run]  NDCG@10 = {ndcg:.4f}  HR@10 = {hr:.4f}")
        else:
            ndcg, hr = self._run_train(decoded)

        runtime = time.perf_counter() - t0

        self._log_trial(self._trial_count, encoded, decoded, ndcg, hr, runtime)

        if self.verbose:
            print(f"  NDCG@10 = {ndcg:.4f}  HR@10 = {hr:.4f}  ({runtime:.0f}s)")

        return ndcg

    def _run_train(self, decoded: dict) -> tuple[float, float]:
        """
        Shell out to train.py and parse NDCG@10 and HR@10 from stdout.

        train.py output (last two lines):
            Best val  NDCG@10 = 0.4821  HR@10 = 0.7103
            Checkpoint saved to checkpoints/ncf_density1.0.pt

        Parameters
        ----------
        decoded : dict
            Actual NCF hyperparameter values (emb_dim, mlp_layers, lr, l2, alpha).

        Returns
        -------
        (ndcg, hr) : (float, float)
            val NDCG@10 and HR@10.  Returns (penalty, 0.0) on failure.
        """
        cmd = [
            "python", "src/train.py",
            "--model",      "ncf",
            "--density",    "1.0",
            "--epochs",     str(self.epochs),
            "--batch-size", "1024",
            "--device",     self.device,
            "--n-neg",      "4",
            "--data-dir",   self._data_dir,
            "--emb-dim",    str(decoded["emb_dim"]),
            "--mlp-layers", *[str(h) for h in decoded["mlp_layers"]],
            "--lr",         str(decoded["lr"]),
            "--l2",         str(decoded["l2"]),
            "--alpha",      str(decoded["alpha"]),
        ]

        if self.verbose:
            print(f"  cmd: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.cs289_repo),   # run from repo root so src/ imports work
                timeout=7200,               # 2-hour hard cap per trial
            )
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] trial {self._trial_count} timed out after 2 hours")
            return self.penalty, 0.0
        except Exception as e:
            print(f"  [ERROR] subprocess failed: {e}")
            return self.penalty, 0.0

        if self.verbose and result.stdout:
            # Print last 5 lines of stdout (epoch table + final summary)
            tail = "\n".join(result.stdout.strip().splitlines()[-5:])
            print(f"  stdout (tail):\n{tail}")

        if result.returncode != 0:
            print(f"  [ERROR] train.py exited with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            return self.penalty, 0.0

        # Parse "Best val  NDCG@10 = 0.4821  HR@10 = 0.7103"
        ndcg_match = re.search(r"NDCG@10\s*=\s*([0-9.]+)", result.stdout)
        hr_match   = re.search(r"HR@10\s*=\s*([0-9.]+)",   result.stdout)

        if ndcg_match is None:
            print(f"  [ERROR] could not parse NDCG@10 from stdout")
            print(f"  stdout: {result.stdout[-300:]}")
            return self.penalty, 0.0

        ndcg = float(ndcg_match.group(1))
        hr   = float(hr_match.group(1)) if hr_match else 0.0

        return ndcg, hr

    def _log_trial(
        self,
        trial_num: int,
        encoded: dict,
        decoded: dict,
        ndcg: float,
        hr: float,
        runtime: float,
    ) -> None:
        """
        Append one trial row to results/ncf/trials.csv.

        CSV columns (from ncf_bridge.md):
            trial, emb_dim, mlp_layers, lr, l2, alpha, val_ndcg, hr, runtime_s
        """
        write_header = not self._csv_path.exists()

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "trial", "emb_dim", "mlp_layers", "lr", "l2", "alpha",
                    "val_ndcg", "hr", "runtime_s",
                    # Encoded values for reference / reproducibility
                    "emb_dim_x", "mlp_x", "log_lr", "log_l2",
                ])
            writer.writerow([
                trial_num,
                decoded["emb_dim"],
                str(decoded["mlp_layers"]),
                f"{decoded['lr']:.2e}",
                f"{decoded['l2']:.2e}",
                f"{decoded['alpha']:.4f}",
                f"{ndcg:.5f}",
                f"{hr:.5f}",
                f"{runtime:.1f}",
                # Encoded
                f"{encoded['emb_dim_x']:.4f}",
                f"{encoded['mlp_x']:.4f}",
                f"{encoded['log_lr']:.4f}",
                f"{encoded['log_l2']:.4f}",
            ])


# ---------------------------------------------------------------------------
# Convenience: baseline runners (random search and grid search)
# Used in 01_ncf_bo.ipynb to build comparison curves at the same budget.
# ---------------------------------------------------------------------------

def random_search_ncf(
    black_box: NCFBlackBox,
    n_trials: int,
    seed: int = 0,
) -> list[float]:
    """
    Random search baseline: sample n_trials configs uniformly at random.

    Returns list of NDCG@10 values in evaluation order.
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_trials):
        encoded = {
            "emb_dim_x": rng.uniform(0.0, 3.0),
            "mlp_x":     rng.uniform(0.0, 2.0),
            "log_lr":    rng.uniform(math.log(1e-4), math.log(1e-2)),
            "log_l2":    rng.uniform(math.log(1e-6), math.log(1e-3)),
            "alpha":     rng.uniform(0.5, 5.0),
        }
        ndcg = black_box(encoded)
        results.append(ndcg)

    return results


def best_ncf_params(bo: "BayesianOptimizer") -> dict:  # type: ignore[name-defined]
    """
    Decode the best params found by BayesianOptimizer into actual NCF values.

    BayesianOptimizer.best_params returns the encoded GP-space dict
    (emb_dim_x, mlp_x, log_lr, log_l2, alpha).  This function converts that
    to the actual hyperparameters you can pass directly to train.py.

    Parameters
    ----------
    bo : BayesianOptimizer
        A completed BO run (bo.run() has been called).

    Returns
    -------
    dict with keys: emb_dim, mlp_layers, lr, l2, alpha
    """
    return decode_params(bo.best_params)


def grid_search_ncf(black_box: NCFBlackBox) -> list[float]:
    """
    Coarse grid search over the NCF search space.

    Grid points (4 × 3 × 2 × 2 × 2 = 96 configs — too many at 4min/trial,
    so we use a reduced 2 × 2 × 2 × 2 × 2 = 32 grid):
        emb_dim     : {64, 256}           (2 levels)
        mlp_layers  : {[128,64], [256,128,64]}  (2 levels)
        lr          : {1e-3, 5e-4}        (2 levels)
        l2          : {1e-5, 1e-4}        (2 levels)
        alpha       : {1.0, 2.5}          (2 levels)

    Returns list of NDCG@10 values in evaluation order.
    """
    grid = [
        # (emb_dim_x, mlp_x, log_lr, log_l2, alpha)
        (emb, mlp, math.log(lr), math.log(l2), alpha)
        for emb in [1.0, 3.0]           # 64, 256
        for mlp in [0.0, 1.0]           # [128,64], [256,128,64]
        for lr  in [1e-3, 5e-4]
        for l2  in [1e-5, 1e-4]
        for alpha in [1.0, 2.5]
    ]

    results = []
    for (emb, mlp, log_lr, log_l2, alpha) in grid:
        encoded = {
            "emb_dim_x": emb,
            "mlp_x":     mlp,
            "log_lr":    log_lr,
            "log_l2":    log_l2,
            "alpha":     alpha,
        }
        ndcg = black_box(encoded)
        results.append(ndcg)

    return results
