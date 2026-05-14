"""
GP regression surrogate for Bayesian Optimization.

Notation
-------------------------------------
  K        n x n kernel matrix,  K_{ij} = K(x_i, x_j)      [Lec 22, Sec 2]
  k(x)     n x 1 vector,         k_i    = K(x_i, x)         [Lec 22, Sec 2]
  mu(x)    posterior mean:        k^T (K + sigma^2 I_n)^{-1} y      [Lec 23, Eq. 2]
  v(x)     posterior variance:    K(x,x) - k^T (K + sigma^2 I_n)^{-1} k  [Lec 23, Sec 2]

Kernel (squared exponential / RBF):
    K(x, x') = alpha * exp( -||x - x'||^2 / (2 * ell^2) )

    alpha > 0  : output scale    — controls vertical scale of the function
    ell   > 0  : length scale    — controls how quickly correlations decay with distance
    sigma^2 > 0: noise variance  — how much each y_i deviates from f(x_i)  [Lec 23, Sec 2]

    Note: K(x, x) = alpha for all x (the SE kernel evaluates to alpha on the diagonal).

Hyperparameter fitting (Section 2, Lec 23):
    The marginal distribution of y given the hyperparameters is:

        y | alpha, ell, sigma^2  ~  N(0,  K + sigma^2 * I_n)

    so the log marginal likelihood is the log-density of that multivariate normal:

        log p(y) = -1/2 * y^T (K + sigma^2 I_n)^{-1} y
                   -1/2 * log |K + sigma^2 I_n|
                   -n/2 * log(2 pi)

    We maximise this over (alpha, ell, sigma^2), working in log-space so the
    optimiser never visits negative values.

Numerical implementation:
    All solves and determinants go through the Cholesky factor of (K + sigma^2 I_n)
    to avoid explicit matrix inversion and to stay numerically stable.
    A small jitter (default 1e-6) is added to the diagonal before factoring.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from typing import Tuple


class GaussianProcess:
    """
    GP regression with a squared exponential kernel.

    Usage
    -----
    gp = GaussianProcess()
    gp.condition(X_obs, y_obs)          # fits hyperparams + caches Cholesky
    mu, var = gp.predict(X_test)        # posterior mean and variance
    """

    # Bounds for log-space hyperparameters during optimisation.
    # log_alpha in [-4, 4]  =>  alpha  in [0.018, 54.6]
    # log_ell   in [-4, 4]  =>  ell    in [0.018, 54.6]
    # log_sigma2 in [-8, 2] =>  sigma2 in [3.4e-4, 7.4]
    _BOUNDS = [(-4.0, 4.0), (-4.0, 4.0), (-8.0, 2.0)]

    def __init__(self, jitter: float = 1e-6, n_restarts: int = 5) -> None:
        """
        Parameters
        ----------
        jitter : float
            Small diagonal offset added before Cholesky to ensure PD.
        n_restarts : int
            Number of random restarts for the log marginal likelihood optimisation.
        """
        self.jitter = jitter
        self.n_restarts = n_restarts

        # Hyperparameters — stored in log-space; initialised to sensible defaults.
        self._log_alpha: float = 0.0    # log(1.0)
        self._log_ell: float = 0.0      # log(1.0)
        self._log_sigma2: float = -2.0  # log(0.135)

        # Training data — set by condition()
        self._X: np.ndarray | None = None   # (n, d)
        self._y: np.ndarray | None = None   # (n,)

        # Cached after condition() so predict() is cheap
        self._chol = None           # cho_factor result for (K + sigma^2 I_n)
        self._alpha_vec: np.ndarray | None = None  # (K + sigma^2 I_n)^{-1} y

    # ------------------------------------------------------------------
    # Properties: hyperparameters in original (positive) scale
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Output scale alpha > 0."""
        return float(np.exp(self._log_alpha))

    @property
    def ell(self) -> float:
        """Length scale ell > 0."""
        return float(np.exp(self._log_ell))

    @property
    def sigma2(self) -> float:
        """Noise variance sigma^2 > 0."""
        return float(np.exp(self._log_sigma2))

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        log_alpha: float,
        log_ell: float,
    ) -> np.ndarray:
        """
        Squared exponential kernel matrix.

            K(x, x') = alpha * exp( -||x - x'||^2 / (2 * ell^2) )

        Parameters
        ----------
        X1 : (n1, d)
        X2 : (n2, d)
        log_alpha, log_ell : kernel hyperparams in log-space

        Returns
        -------
        K : (n1, n2)
        """
        alpha = np.exp(log_alpha)
        ell = np.exp(log_ell)
        # Squared Euclidean distances via broadcasting: (n1, n2, d) -> (n1, n2)
        diff = X1[:, None, :] - X2[None, :, :]   # (n1, n2, d)
        sq_dist = np.einsum("ijk,ijk->ij", diff, diff)  # (n1, n2)
        return alpha * np.exp(-sq_dist / (2.0 * ell ** 2))

    # ------------------------------------------------------------------
    # Log marginal likelihood (pure function — does not modify self)
    # ------------------------------------------------------------------

    def _lml(self, log_params: np.ndarray) -> float:
        """
        Negative log marginal likelihood at the given log-space hyperparameters.

        The marginal distribution of y is [Lec 23, Sec 2]:

            y | alpha, ell, sigma^2  ~  N(0,  K + sigma^2 * I_n)

        whose log-density is:

            log p(y) = -1/2 * y^T (K + sigma^2 I_n)^{-1} y
                       -1/2 * log |K + sigma^2 I_n|
                       -n/2 * log(2 pi)

        Computed via Cholesky:
            log |A| = 2 * sum(log diag(L))   where A = L L^T
            y^T A^{-1} y = ||L^{-1} y||^2

        Returns the *negative* LML so scipy.optimize.minimize can maximise it.

        Parameters
        ----------
        log_params : (3,)  [log_alpha, log_ell, log_sigma2]
        """
        log_alpha, log_ell, log_sigma2 = log_params
        sigma2 = np.exp(log_sigma2)
        n = self._X.shape[0]

        # K: n x n kernel matrix  [Lec 23, notation before Eq. 2]
        K = self._kernel(self._X, self._X, log_alpha, log_ell)

        # K + sigma^2 * I_n  [Lec 23, Sec 2]
        K_noisy = K + (sigma2 + self.jitter) * np.eye(n)

        try:
            chol = cho_factor(K_noisy, lower=True)
        except np.linalg.LinAlgError:
            # Not positive definite — penalise heavily
            return 1e10

        alpha_vec = cho_solve(chol, self._y)

        # -1/2 * y^T (K + sigma^2 I_n)^{-1} y  ("data fit" term)
        data_fit = -0.5 * float(self._y @ alpha_vec)

        # -1/2 * log |K + sigma^2 I_n| = -sum(log diag(L))  ("complexity" term)
        L = chol[0]
        complexity = -float(np.sum(np.log(np.diag(L))))

        # -n/2 * log(2 pi)  (normalisation constant)
        constant = -0.5 * n * np.log(2.0 * np.pi)

        lml = data_fit + complexity + constant
        return -lml  # negate for minimisation

    # ------------------------------------------------------------------
    # Hyperparameter fitting
    # ------------------------------------------------------------------

    def _fit_hyperparams(self) -> None:
        """
        Maximise the log marginal likelihood over (alpha, ell, sigma^2).

        Optimises in log-space with L-BFGS-B. Multiple random restarts
        reduce sensitivity to local optima.
        """
        rng = np.random.default_rng(seed=0)
        best_nlml = np.inf
        best_params = np.array([self._log_alpha, self._log_ell, self._log_sigma2])

        # Build starting points: current params + (n_restarts - 1) random ones
        lo = np.array([b[0] for b in self._BOUNDS])
        hi = np.array([b[1] for b in self._BOUNDS])
        starts = [best_params] + [
            rng.uniform(lo, hi) for _ in range(self.n_restarts - 1)
        ]

        for x0 in starts:
            res = minimize(
                self._lml,
                x0,
                method="L-BFGS-B",
                bounds=self._BOUNDS,
                options={"maxiter": 300, "ftol": 1e-12},
            )
            if res.success and res.fun < best_nlml:
                best_nlml = res.fun
                best_params = res.x

        self._log_alpha, self._log_ell, self._log_sigma2 = best_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def condition(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """
        Condition the GP on observations (X, y).

        Fits hyperparameters by maximising the log marginal likelihood, then
        caches the Cholesky factorisation of (K + sigma^2 I_n) for fast prediction.

        Parameters
        ----------
        X : (n, d)  input configurations
        y : (n,)    observed objective values

        Returns
        -------
        self  (for chaining)
        """
        self._X = np.atleast_2d(np.asarray(X, dtype=float))
        self._y = np.asarray(y, dtype=float).ravel()

        # 1. Fit hyperparameters [Lec 23, Sec 2: "we would need to estimate sigma...
        #    the marginal likelihood of y_1,...,y_n is important"]
        self._fit_hyperparams()

        # 2. Cache Cholesky of (K + sigma^2 I_n) with fitted hyperparams
        n = self._X.shape[0]
        K = self._kernel(self._X, self._X, self._log_alpha, self._log_ell)
        K_noisy = K + (self.sigma2 + self.jitter) * np.eye(n)
        self._chol = cho_factor(K_noisy, lower=True)

        # alpha_vec = (K + sigma^2 I_n)^{-1} y  — used in both mu and v
        self._alpha_vec = cho_solve(self._chol, self._y)

        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior mean and variance at test points.

        Following Lecture 23, Section 2:

            mu(x)  = k(x)^T (K + sigma^2 I_n)^{-1} y       [Eq. 2]
            v(x)   = K(x,x) - k(x)^T (K + sigma^2 I_n)^{-1} k(x)

        where k(x) = (K(x_1,x), ..., K(x_n,x))^T  [Lec 22, notation].

        Note: for the SE kernel K(x,x) = alpha for every x.

        Parameters
        ----------
        X_test : (m, d)  test configurations

        Returns
        -------
        mu  : (m,)  posterior mean at each test point
        var : (m,)  posterior variance at each test point (clipped to >= 0)
        """
        if self._chol is None:
            raise RuntimeError("GP has not been conditioned on data yet. Call condition() first.")

        X_test = np.atleast_2d(np.asarray(X_test, dtype=float))

        # k: (n, m)  cross-kernel matrix  k_{i,j} = K(x_i, x*_j)
        k = self._kernel(self._X, X_test, self._log_alpha, self._log_ell)

        # mu(x) = k^T (K + sigma^2 I_n)^{-1} y = k^T alpha_vec  [Lec 23, Eq. 2]
        mu = k.T @ self._alpha_vec  # (m,)

        # v(x) = K(x,x) - k^T (K + sigma^2 I_n)^{-1} k         [Lec 23, Sec 2]
        # Compute (K + sigma^2 I_n)^{-1} k via the cached Cholesky
        K_inv_k = cho_solve(self._chol, k)  # (n, m)

        # K(x*, x*) diagonal: alpha * exp(0) = alpha for SE kernel
        k_self = self.alpha * np.ones(X_test.shape[0])  # (m,)

        # k^T (K + sigma^2 I_n)^{-1} k = sum over n of k_{:,j} * K_inv_k_{:,j}
        var = k_self - np.einsum("ni,ni->i", k, K_inv_k)  # (m,)
        var = np.clip(var, 0.0, None)  # guard against small negative values

        return mu, var

    def __repr__(self) -> str:
        return (
            f"GaussianProcess("
            f"alpha={self.alpha:.4f}, "
            f"ell={self.ell:.4f}, "
            f"sigma2={self.sigma2:.4f})"
        )
