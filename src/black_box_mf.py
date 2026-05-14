"""
MF black-box objective for Bayesian Optimisation.

Mirrors black_box_ncf.py but for Matrix Factorization (Model A).
MF has no MLP, so the search space is 4D instead of 5D (no mlp_layers).
MF is fast enough to tune on CPU — each trial takes ~20-30 min on MacBook.

Parameter encoding
------------------
  emb_dim_x  in [0, 3]              -> {0:32, 1:64, 2:128, 3:256}
  log_lr     in [log 1e-4, log 1e-2] -> lr = exp(log_lr)
  log_l2     in [log 1e-6, log 1e-3] -> l2 = exp(log_l2)
  alpha      in [0.5, 5.0]           -> as-is (WMF confidence scale)

Fixed:
  --model mf  --density 1.0  --epochs 15  --batch-size 1024  --n-neg 4
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
import time
import threading
from pathlib import Path

import numpy as np


MF_PARAM_SPACE = [
    {"name": "emb_dim_x", "type": "continuous", "bounds": (0.0, 3.0)},
    {"name": "log_lr",    "type": "continuous", "bounds": (math.log(1e-4), math.log(1e-2))},
    {"name": "log_l2",    "type": "continuous", "bounds": (math.log(1e-6), math.log(1e-3))},
    {"name": "alpha",     "type": "continuous", "bounds": (0.5, 5.0)},
]

_EMB_DIM_MAP: dict[int, int] = {0: 32, 1: 64, 2: 128, 3: 256}


def decode_params(encoded: dict) -> dict:
    """Convert GP-space encoded dict to actual MF hyperparameters."""
    emb_idx = int(round(float(encoded["emb_dim_x"])))
    emb_idx = max(0, min(3, emb_idx))

    return {
        "emb_dim": _EMB_DIM_MAP[emb_idx],
        "lr":      float(np.exp(encoded["log_lr"])),
        "l2":      float(np.exp(encoded["log_l2"])),
        "alpha":   float(encoded["alpha"]),
    }


class MFBlackBox:
    """
    Callable black-box objective for MF hyperparameter tuning.

    Wraps cs289-ranking/src/train.py --model mf and returns val NDCG@10.
    Designed to run on CPU — MF is simple enough that GPU is not required.

    Parameters
    ----------
    cs289_repo : str or Path
        Absolute path to the cs289-ranking repo root.
    results_dir : str or Path or None
        Where to write trials.csv. Defaults to results/mf/ in this repo.
    device : str
        'cpu' or 'mps' for MacBook, 'cuda' for SCF.
    epochs : int
        Training epochs per trial. 15 is sufficient for MF.
    dry_run : bool
        Return fake NDCG values for testing without running train.py.
    verbose : bool
        Print trial progress to stdout.
    penalty : float
        NDCG returned on failure so BO can continue.
    """

    def __init__(
        self,
        cs289_repo: str | Path,
        results_dir: str | Path | None = None,
        device: str = "cpu",
        epochs: int = 15,
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
            results_dir = Path(__file__).resolve().parent.parent / "results" / "mf"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path    = self.results_dir / "trials.csv"
        self._trial_count = self._count_existing_trials()

        self._data_dir = str(self.cs289_repo / "data" / "raw" / "ml-1m")

        if not self.dry_run:
            train_script = self.cs289_repo / "src" / "train.py"
            if not train_script.exists():
                raise FileNotFoundError(
                    f"train.py not found at {train_script}.\n"
                    f"Check --cs289-repo or use --dry-run for testing."
                )

    def _count_existing_trials(self) -> int:
        if not self._csv_path.exists():
            return 0
        with open(self._csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)

    def __call__(self, encoded: dict) -> float:
        self._trial_count += 1
        decoded = decode_params(encoded)

        if self.verbose:
            print(f"\n[MF trial {self._trial_count}]  decoded: {decoded}")

        t0 = time.perf_counter()

        if self.dry_run:
            rng  = np.random.default_rng(self._trial_count + 100)
            ndcg = float(rng.uniform(0.35, 0.50))
            hr   = float(rng.uniform(0.60, 0.75))
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
        cmd = [
            "python", "src/train.py",
            "--model",      "mf",
            "--density",    "1.0",
            "--epochs",     str(self.epochs),
            "--batch-size", "1024",
            "--device",     self.device,
            "--n-neg",      "4",
            "--data-dir",   self._data_dir,
            "--emb-dim",    str(decoded["emb_dim"]),
            "--lr",         str(decoded["lr"]),
            "--l2",         str(decoded["l2"]),
            "--alpha",      str(decoded["alpha"]),
        ]

        if self.verbose:
            print(f"  cmd: {' '.join(cmd)}")

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.cs289_repo),
            )

            def _drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            for line in proc.stdout:
                stdout_lines.append(line)
                if self.verbose:
                    print(f"  {line}", end="", flush=True)

            proc.wait()
            stderr_thread.join(timeout=5)

        except Exception as e:
            print(f"  [ERROR] subprocess failed: {e}")
            return self.penalty, 0.0

        if proc.returncode != 0:
            print(f"  [ERROR] train.py exited with code {proc.returncode}")
            if stderr_lines:
                print(f"  stderr: {''.join(stderr_lines[-20:])}")
            return self.penalty, 0.0

        stdout_text  = "".join(stdout_lines)
        ndcg_match   = re.search(r"NDCG@10\s*=\s*([0-9.]+)", stdout_text)
        hr_match     = re.search(r"HR@10\s*=\s*([0-9.]+)",   stdout_text)

        if ndcg_match is None:
            print("  [ERROR] could not parse NDCG@10 from stdout")
            return self.penalty, 0.0

        return float(ndcg_match.group(1)), float(hr_match.group(1)) if hr_match else 0.0

    def _log_trial(self, trial_num, encoded, decoded, ndcg, hr, runtime):
        write_header = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "trial", "emb_dim", "lr", "l2", "alpha",
                    "val_ndcg", "hr", "runtime_s",
                    "emb_dim_x", "log_lr", "log_l2",
                ])
            writer.writerow([
                trial_num,
                decoded["emb_dim"],
                f"{decoded['lr']:.2e}",
                f"{decoded['l2']:.2e}",
                f"{decoded['alpha']:.4f}",
                f"{ndcg:.5f}",
                f"{hr:.5f}",
                f"{runtime:.1f}",
                f"{encoded['emb_dim_x']:.4f}",
                f"{encoded['log_lr']:.4f}",
                f"{encoded['log_l2']:.4f}",
            ])


def best_mf_params(bo) -> dict:
    """Decode BayesianOptimizer.best_params into actual MF hyperparameters."""
    return decode_params(bo.best_params)
