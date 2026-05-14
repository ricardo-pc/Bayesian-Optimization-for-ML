"""
Bayesian Optimisation loop.

Implements the standard BO loop described in Frazier (2018) and applied
in Snoek, Larochelle, Adams (2012) to ML hyperparameter tuning.

Algorithm
---------
1. Draw n_init configurations uniformly at random and evaluate the black-box.
2. Repeat until the evaluation budget is exhausted:
   a. Condition the GP on all observations (fits hyperparams via log-MLL).
   b. Find x_next = argmax EI(x) over the search space.
   c. Evaluate y_next = f(x_next)  [the expensive black-box call].
   d. Append (x_next, y_next) to the observation set.
3. Return the full trial history and the best configuration found.

Normalisation
-------------
Inputs are normalised to [0, 1]^d before being passed to the GP (the GP
does not know about the original search space — it always sees unit-cube
inputs).  This keeps kernel length scales interpretable and avoids the
optimiser needing very different l values per dimension.

Outputs (y values) are standardised to zero mean and unit variance before
GP fitting, so the output scale alpha and noise sigma^2 stay in a
comparable range regardless of the objective's magnitude.

Both transforms are inverted before results are reported.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Callable, Sequence

from .gp import GaussianProcess
from .acquisition import next_best_candidate


class BayesianOptimizer:
    """
    Model-agnostic Bayesian Optimisation loop.

    Works with any black-box objective function that accepts a dict of
    hyperparameter values and returns a scalar (higher = better).

    Parameters
    ----------
    objective : Callable[[dict], float]
        Black-box function to maximise.  Must accept a dict whose keys
        match the names in `param_space` and return a scalar float.
    param_space : list of dicts
        Each entry describes one hyperparameter::

            {"name": str, "type": "continuous" | "integer",
             "bounds": (lo, hi)}

        All parameters are treated as continuous by the GP; integers are
        rounded when passed to the objective.
    n_init : int
        Number of random initial evaluations before BO starts.
    budget : int
        Total number of objective evaluations (including n_init).
    gp_kwargs : dict, optional
        Extra keyword arguments forwarded to GaussianProcess().
    acq_kwargs : dict, optional
        Extra keyword arguments forwarded to next_best_candidate().
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        objective: Callable[[dict], float],
        param_space: list[dict],
        n_init: int = 5,
        budget: int = 30,
        gp_kwargs: dict | None = None,
        acq_kwargs: dict | None = None,
        seed: int | None = 0,
        verbose: bool = True,
    ) -> None:
        self.objective = objective
        self.param_space = param_space
        self.n_init = n_init
        self.budget = budget
        self.gp_kwargs = gp_kwargs or {}
        self.acq_kwargs = acq_kwargs or {}
        self.seed = seed
        self.verbose = verbose

        self._rng = np.random.default_rng(seed)

        # Derived quantities from param_space
        self._names: list[str] = [p["name"] for p in param_space]
        self._types: list[str] = [p["type"] for p in param_space]
        self._lo = np.array([p["bounds"][0] for p in param_space], dtype=float)
        self._hi = np.array([p["bounds"][1] for p in param_space], dtype=float)
        self._d = len(param_space)

        # Bounds in normalised [0, 1]^d space — constant, used by EI maximiser
        self._unit_bounds: list[tuple[float, float]] = [(0.0, 1.0)] * self._d

        # Trial history — populated by run()
        self._X_raw: list[np.ndarray] = []   # (d,) configs in original scale
        self._y_raw: list[float] = []         # objective values in original scale
        self._runtimes: list[float] = []      # wall-clock seconds per trial

    # ------------------------------------------------------------------
    # Space transforms
    # ------------------------------------------------------------------

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map x from original bounds to [0, 1]^d."""
        return (x - self._lo) / (self._hi - self._lo)

    def _from_unit(self, z: np.ndarray) -> np.ndarray:
        """Map z from [0, 1]^d back to original bounds."""
        x = self._lo + z * (self._hi - self._lo)
        # Round integer parameters
        for i, t in enumerate(self._types):
            if t == "integer":
                x[i] = np.round(x[i])
        return x

    def _standardise_y(self, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Standardise y to zero mean, unit variance.
        Returns (y_std, mean, std).
        """
        mu = float(np.mean(y))
        sigma = float(np.std(y))
        if sigma < 1e-8:
            sigma = 1.0
        return (y - mu) / sigma, mu, sigma

    # ------------------------------------------------------------------
    # Objective evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, x_raw: np.ndarray) -> tuple[float, float]:
        """
        Evaluate the black-box objective at x_raw (original scale).

        Returns (y, runtime_seconds).
        """
        params = {
            name: (int(round(x_raw[i])) if self._types[i] == "integer" else float(x_raw[i]))
            for i, name in enumerate(self._names)
        }
        t0 = time.perf_counter()
        y = float(self.objective(params))
        runtime = time.perf_counter() - t0
        return y, runtime

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run the full Bayesian Optimisation loop.

        Phase 1 — random initialisation (n_init evaluations):
            Sample configurations uniformly at random from the search space.

        Phase 2 — BO iterations (budget - n_init evaluations):
            For each iteration:
              (a) Condition GP on all observations, fitting hyperparams.
              (b) Find x_next = argmax EI.
              (c) Evaluate f(x_next).
              (d) Append to history.

        Returns
        -------
        pd.DataFrame
            One row per trial with columns:
            trial, <param names...>, objective, best_so_far, runtime_s, phase
        """
        # ------- Phase 1: random initialisation -------
        if self.verbose:
            print(f"[BO] Phase 1: {self.n_init} random initial evaluations")

        for t in range(self.n_init):
            z = self._rng.uniform(0, 1, size=self._d)
            x_raw = self._from_unit(z)
            y, rt = self._evaluate(x_raw)
            self._X_raw.append(x_raw)
            self._y_raw.append(y)
            self._runtimes.append(rt)
            if self.verbose:
                print(f"  trial {t+1:3d} | y = {y:.5f} | {rt:.1f}s")

        # ------- Phase 2: BO iterations -------
        gp = GaussianProcess(**self.gp_kwargs)

        if self.verbose:
            print(f"\n[BO] Phase 2: {self.budget - self.n_init} BO iterations")

        for t in range(self.n_init, self.budget):
            # (a) Build GP: normalise inputs to [0,1]^d, standardise y
            X_unit = np.vstack([self._to_unit(x) for x in self._X_raw])
            y_arr = np.array(self._y_raw)
            y_std, y_mu, y_sigma = self._standardise_y(y_arr)

            gp.condition(X_unit, y_std)

            # y_best in standardised space (we are maximising)
            y_best_std = float(y_std.max())

            # (b) Maximise EI in [0,1]^d
            z_next = next_best_candidate(
                gp,
                y_best=y_best_std,
                bounds=self._unit_bounds,
                seed=int(self._rng.integers(0, 2**31)),
                **self.acq_kwargs,
            )

            # (c) Map back to original scale and evaluate
            x_next = self._from_unit(z_next)
            y_next, rt = self._evaluate(x_next)
            self._X_raw.append(x_next)
            self._y_raw.append(y_next)
            self._runtimes.append(rt)

            best_so_far = max(self._y_raw)
            if self.verbose:
                print(
                    f"  trial {t+1:3d} | y = {y_next:.5f} | "
                    f"best = {best_so_far:.5f} | {rt:.1f}s | {gp}"
                )

        # ------- Build results DataFrame -------
        records = []
        best_so_far = -np.inf
        for t, (x_raw, y, rt) in enumerate(
            zip(self._X_raw, self._y_raw, self._runtimes)
        ):
            best_so_far = max(best_so_far, y)
            row = {"trial": t + 1}
            for i, name in enumerate(self._names):
                row[name] = (
                    int(round(x_raw[i]))
                    if self._types[i] == "integer"
                    else float(x_raw[i])
                )
            row["objective"] = y
            row["best_so_far"] = best_so_far
            row["runtime_s"] = rt
            row["phase"] = "random" if t < self.n_init else "BO"
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def best_params(self) -> dict:
        """Configuration that achieved the best observed objective value."""
        if not self._y_raw:
            raise RuntimeError("No evaluations yet. Call run() first.")
        idx = int(np.argmax(self._y_raw))
        x_raw = self._X_raw[idx]
        return {
            name: (
                int(round(x_raw[i])) if self._types[i] == "integer" else float(x_raw[i])
            )
            for i, name in enumerate(self._names)
        }

    @property
    def best_value(self) -> float:
        """Best observed objective value."""
        if not self._y_raw:
            raise RuntimeError("No evaluations yet. Call run() first.")
        return float(max(self._y_raw))
