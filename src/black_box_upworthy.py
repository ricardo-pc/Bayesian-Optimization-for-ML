"""
Upworthy zero-shot classification black-box for Bayesian Optimisation.

Background
----------
The STAT 230A pipeline ran facebook/bart-large-mnli on 55,092 Upworthy
headlines and saved the top predicted category + confidence score for each
headline to data/processed/categories_raw.csv.  That inference step is
already done — this black box reads those saved probabilities and applies
configurable post-processing parameters to produce a category assignment,
then measures how *meaningful* the resulting categorisation is.

Why is this a BO problem?
--------------------------
BART always assigns a label, even when it is only 51 % confident.  Two
decisions affect whether the resulting categories are useful for explaining
click-through rate (CTR):

  1. confidence_threshold (continuous, [0.05, 0.90])
     Headlines where BART is below this threshold are labelled "other" and
     excluded from the ANOVA.  Low threshold → noisy categories (lots of
     mis-classifications, high within-group variance → low F).  High
     threshold → clean but tiny groups (→ degenerate or undefined ANOVA).
     The sweet spot is non-obvious and depends on the data distribution.

  2. log_min_cat_size (log-encoded, [log 10, log 400])
     After applying the threshold, we only include categories with at least
     min_cat_size = exp(log_min_cat_size) headlines in the ANOVA.  Too small
     → individual outlier headlines inflate variance; too large → we exclude
     real categories that happen to be rare.  This parameter interacts with
     the threshold: a stricter threshold produces smaller groups, so the
     minimum must be tuned jointly.

Objective: one-way ANOVA F-statistic on log_ctr
-------------------------------------------------
After category assignment we test whether the group means differ:

    H0 : mu_category1 = mu_category2 = ... = mu_k
    F  = (between-group variance) / (within-group variance)

High F means the categories are predictive of CTR — knowing the topic of a
headline tells us something real about how likely it is to get clicked.
Low F means the categories are noise.  We maximise F.

The F-statistic is returned as 0.0 (penalty) when fewer than 2 groups pass
the size filter (degenerate case — ANOVA is undefined).

Parameter space
---------------
  confidence_threshold  in [0.05, 0.90]  (raw, no encoding)
  log_min_cat_size      in [log 10, log 400]  -> min_cat_size = exp(x)

The log-encoding for min_cat_size follows the same philosophy as log_lr in
the NCF/MF experiments: differences in category size are multiplicative (the
gap between 10 and 30 matters as much as the gap between 100 and 300), so
the SE kernel should operate on the log scale.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


# ---------------------------------------------------------------------------
# Parameter space definition (matches BayesianOptimizer format)
# ---------------------------------------------------------------------------

UPWORTHY_PARAM_SPACE = [
    {
        "name":   "confidence_threshold",
        "type":   "continuous",
        "bounds": (0.05, 0.90),
    },
    {
        "name":   "log_min_cat_size",
        "type":   "continuous",
        "bounds": (math.log(10), math.log(400)),
    },
]

# The 7 BART candidate labels used in classify_categories.py
CATEGORIES = [
    "politics and government",
    "social justice and inequality",
    "health and medicine",
    "environment and climate",
    "science and technology",
    "economics and labor",
    "lifestyle and culture",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_params(encoded: dict) -> dict:
    """
    Convert GP-space encoded dict to human-readable Upworthy pipeline params.

    Parameters
    ----------
    encoded : dict with keys 'confidence_threshold', 'log_min_cat_size'

    Returns
    -------
    dict with keys 'confidence_threshold', 'min_cat_size'
    """
    return {
        "confidence_threshold": float(encoded["confidence_threshold"]),
        "min_cat_size":         int(round(math.exp(float(encoded["log_min_cat_size"])))),
    }


def best_upworthy_params(bo) -> dict:
    """Decode BayesianOptimizer.best_params into human-readable pipeline params."""
    return decode_params(bo.best_params)


# ---------------------------------------------------------------------------
# Black-box class
# ---------------------------------------------------------------------------

class UpworthyBlackBox:
    """
    Callable black-box objective for Upworthy category threshold tuning.

    Loads the pre-computed BART classification results once at construction,
    then efficiently applies different threshold / min-size configurations
    without re-running the model.  Each trial takes milliseconds.

    Parameters
    ----------
    upworthy_repo : str or Path
        Absolute path to the A-B-testing-analysis-upworthy repo root.
        Must contain:
          data/processed/categories_raw.csv   (BART output)
          data/processed/confirmatory_clean.csv  (CTR data)
    results_dir : str or Path or None
        Where to write trials.csv after each evaluation.
        Defaults to results/upworthy/ relative to this repo root.
    min_groups : int
        Minimum number of non-"other" category groups required for a valid
        ANOVA.  If fewer groups pass the size filter, returns penalty (0.0).
    penalty : float
        Objective value returned when the configuration is degenerate
        (fewer than min_groups groups, or NaN F-statistic).
    verbose : bool
        Print trial info to stdout.
    """

    def __init__(
        self,
        upworthy_repo: str | Path,
        results_dir: str | Path | None = None,
        min_groups: int = 2,
        penalty: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.upworthy_repo = Path(upworthy_repo).resolve()
        self.min_groups    = min_groups
        self.penalty       = penalty
        self.verbose       = verbose

        if results_dir is None:
            results_dir = Path(__file__).resolve().parent.parent / "results" / "upworthy"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.results_dir / "trials.csv"

        # Count any trials already logged (for resume-friendly numbering)
        self._trial_count = self._count_existing_trials()

        # ------------------------------------------------------------------
        # Load data once — all trials share this in-memory DataFrame
        # ------------------------------------------------------------------
        cats_path  = self.upworthy_repo / "data" / "processed" / "categories_raw.csv"
        clean_path = self.upworthy_repo / "data" / "processed" / "confirmatory_clean.csv"

        if not cats_path.exists():
            raise FileNotFoundError(
                f"BART output not found at {cats_path}.\n"
                f"Run classify_categories.py first, or check --upworthy-repo."
            )
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Clean data not found at {clean_path}.\n"
                f"Check --upworthy-repo path."
            )

        cats  = pd.read_csv(cats_path)
        clean = pd.read_csv(clean_path, usecols=["headline", "log_ctr"])

        # Inner join on headline text — both files have 55,092 rows
        self.df = clean.merge(cats, on="headline", how="inner").reset_index(drop=True)

        n = len(self.df)
        if self.verbose:
            print(f"[Upworthy] Loaded {n:,} headlines with log_ctr and BART scores.")

    def _count_existing_trials(self) -> int:
        if not self._csv_path.exists():
            return 0
        with open(self._csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def __call__(self, encoded: dict) -> float:
        """
        Evaluate the ANOVA F-statistic for a given configuration.

        Steps
        -----
        1. Apply confidence_threshold: headlines below threshold → "other".
        2. Group non-"other" headlines by assigned category.
        3. Filter groups with fewer than min_cat_size headlines.
        4. Run one-way ANOVA on log_ctr across surviving groups.
        5. Return F-statistic (0.0 if fewer than 2 groups survive).

        Parameters
        ----------
        encoded : dict
            Must have keys 'confidence_threshold' and 'log_min_cat_size'
            (in the GP search space, i.e. log_min_cat_size is log-encoded).

        Returns
        -------
        float : F-statistic (higher = categories more predictive of CTR).
        """
        self._trial_count += 1
        decoded     = decode_params(encoded)
        conf_thresh = decoded["confidence_threshold"]
        min_cat     = decoded["min_cat_size"]

        t0 = time.perf_counter()

        # Step 1 — apply threshold: headlines below → labelled "other"
        # We never modify self.df; work on a view via boolean mask
        mask_assigned = self.df["category_confidence"] >= conf_thresh
        coverage      = float(mask_assigned.mean())

        # Step 2 — collect log_ctr arrays per category (exclude "other")
        groups: dict[str, np.ndarray] = {}
        for cat, grp in self.df[mask_assigned].groupby("category_raw"):
            if len(grp) >= min_cat:
                groups[cat] = grp["log_ctr"].values

        n_groups = len(groups)

        # Step 3 — degenerate case: not enough groups for ANOVA
        if n_groups < self.min_groups:
            f_stat  = self.penalty
            p_value = float("nan")
        else:
            # Step 4 — one-way ANOVA  [scipy.stats.f_oneway]
            f_stat, p_value = f_oneway(*groups.values())
            f_stat  = float(f_stat)  if not np.isnan(f_stat)  else self.penalty
            p_value = float(p_value) if not np.isnan(p_value) else float("nan")

        runtime = time.perf_counter() - t0

        if self.verbose:
            cats_str = ", ".join(f"{c[:12]}" for c in sorted(groups))
            print(
                f"\n[Upworthy trial {self._trial_count}]"
                f"  threshold={conf_thresh:.3f}  min_cat={min_cat}"
                f"  coverage={coverage:.2%}  groups={n_groups}"
            )
            if n_groups >= self.min_groups:
                print(f"  active cats: {cats_str}")
                print(f"  F = {f_stat:.2f}  p = {p_value:.4f}  ({runtime*1000:.0f}ms)")
            else:
                print(f"  [degenerate]  F = {f_stat:.2f}  ({runtime*1000:.0f}ms)")

        self._log_trial(
            trial_num   = self._trial_count,
            encoded     = encoded,
            decoded     = decoded,
            n_groups    = n_groups,
            coverage    = coverage,
            f_stat      = f_stat,
            p_value     = p_value,
            runtime     = runtime,
        )

        return f_stat

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _log_trial(
        self,
        trial_num: int,
        encoded:   dict,
        decoded:   dict,
        n_groups:  int,
        coverage:  float,
        f_stat:    float,
        p_value:   float,
        runtime:   float,
    ) -> None:
        write_header = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "trial",
                    "confidence_threshold",
                    "min_cat_size",
                    "n_groups",
                    "coverage",
                    "f_stat",
                    "p_value",
                    "runtime_ms",
                    # encoded (GP-space) values — needed to rebuild GP in notebook
                    "log_min_cat_size",
                ])
            writer.writerow([
                trial_num,
                f"{decoded['confidence_threshold']:.4f}",
                decoded["min_cat_size"],
                n_groups,
                f"{coverage:.4f}",
                f"{f_stat:.4f}",
                f"{p_value:.6f}" if not np.isnan(p_value) else "nan",
                f"{runtime * 1000:.1f}",
                f"{encoded['log_min_cat_size']:.4f}",
            ])
