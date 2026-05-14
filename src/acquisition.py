"""
Expected Improvement (EI) acquisition function for Bayesian Optimization.

The EI acquisition function balances exploration and exploitation by measuring
the expected amount by which the next evaluation will exceed the current best
observed value f*.

Definition
----------
Given the GP posterior at a candidate point x:

    mu(x)  = posterior mean       [Lec 23, Eq. 2]
    v(x)   = posterior variance   [Lec 23, Sec 2]
    std(x) = sqrt(v(x))

the Expected Improvement is:

    EI(x) = E[ max(f(x) - f*, 0) ]
           = (mu(x) - f*) * Phi(Z)  +  std(x) * phi(Z)

where
    Z      = (mu(x) - f*) / std(x)
    Phi    = standard normal CDF
    phi    = standard normal PDF
    f*     = best observed value so far  (max of y_1, ..., y_n)

Interpretation
--------------
The first term  (mu(x) - f*) * Phi(Z)  drives exploitation:
    it is large when mu(x) >> f* (the mean already exceeds the best).

The second term  std(x) * phi(Z)  drives exploration:
    it is large when std(x) is large (high uncertainty — could be much better).

When std(x) = 0 (i.e., x is an already-observed point), EI = 0 by convention.

"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .gp import GaussianProcess


def expected_improvement(
    X_candidates: np.ndarray,
    gp: GaussianProcess,
    y_best: float,
    xi: float = 0.0,
) -> np.ndarray:
    """
    Expected Improvement at a batch of candidate points.

    EI(x) = (mu(x) - f* - xi) * Phi(Z)  +  std(x) * phi(Z)

        Z = (mu(x) - f* - xi) / std(x)

    The optional xi >= 0 is a small "exploration bonus" that biases EI
    toward unexplored regions. xi=0 gives the standard EI (Frazier 2018).

    Parameters
    ----------
    X_candidates : (m, d)
        Candidate configurations at which to evaluate EI.
    gp : GaussianProcess
        A conditioned GP surrogate (gp.condition() must have been called).
    y_best : float
        Best objective value observed so far, i.e. f* = max(y_1, ..., y_n).
        We assume maximisation — higher y is better.
    xi : float, optional
        Exploration bonus (default 0.0 — pure standard EI).

    Returns
    -------
    ei : (m,)
        EI value at each candidate. Always >= 0.
    """
    X_candidates = np.atleast_2d(np.asarray(X_candidates, dtype=float))

    # Posterior mean and variance from the GP  [Lec 23, Eq. 2 and Sec 2]
    mu, var = gp.predict(X_candidates)   # (m,), (m,)
    std = np.sqrt(var)                   # (m,)

    # Improvement over current best (with optional exploration bonus xi)
    improvement = mu - y_best - xi       # (m,)

    # Z = (mu(x) - f* - xi) / std(x)
    # Where std = 0, EI = 0 by convention (already observed, no uncertainty).
    ei = np.zeros_like(improvement)
    mask = std > 0.0
    if mask.any():
        Z = improvement[mask] / std[mask]
        ei[mask] = improvement[mask] * norm.cdf(Z) + std[mask] * norm.pdf(Z)

    # EI is always non-negative (it is an expectation of a non-negative quantity)
    return np.clip(ei, 0.0, None)


def next_best_candidate(
    gp: GaussianProcess,
    y_best: float,
    bounds: list[tuple[float, float]],
    n_warmup: int = 1000,
    n_restarts: int = 10,
    xi: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Find the candidate x that maximises EI over the search space.

    Strategy: evaluate EI on a large random grid (warm-up), then run
    L-BFGS-B from the top-scoring warm-up points to find the local maximum.
    Returns the best point found across all restarts.

    Parameters
    ----------
    gp : GaussianProcess
        Conditioned GP surrogate.
    y_best : float
        Best observed objective value (f*).
    bounds : list of (lo, hi) tuples, length d
        Search space bounds for each dimension.
    n_warmup : int
        Number of random points for the initial grid.
    n_restarts : int
        Number of L-BFGS-B restarts from the best warm-up points.
    xi : float
        Exploration bonus passed to expected_improvement().
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    x_next : (d,)
        Configuration that maximises EI.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # 1. Warm-up: evaluate EI on a dense random grid
    X_warmup = rng.uniform(lo, hi, size=(n_warmup, d))
    ei_warmup = expected_improvement(X_warmup, gp, y_best, xi=xi)

    # 2. Pick the top n_restarts warm-up points as L-BFGS-B starting points
    top_idx = np.argsort(ei_warmup)[-n_restarts:][::-1]
    X_starts = X_warmup[top_idx]

    def neg_ei(x: np.ndarray) -> float:
        """Negative EI (scalar) for a single point — scipy expects minimisation."""
        return -float(expected_improvement(x.reshape(1, -1), gp, y_best, xi=xi)[0])

    best_ei = -np.inf
    best_x = X_starts[0]

    for x0 in X_starts:
        res = __import__("scipy.optimize", fromlist=["minimize"]).minimize(
            neg_ei,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        candidate_ei = -res.fun
        if candidate_ei > best_ei:
            best_ei = candidate_ei
            best_x = res.x

    return np.clip(best_x, lo, hi)
