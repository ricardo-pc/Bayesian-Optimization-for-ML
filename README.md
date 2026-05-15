# Bayesian Optimisation for Machine Learning

**STAT 238 Final Project · UC Berkeley Spring 2026**
*Ricardo Perez Castillo*

A from-scratch implementation of Bayesian Optimisation (Gaussian Process surrogate +
Expected Improvement acquisition) applied to two real machine-learning hyperparameter
tuning problems, plus a Branin-function sanity check.

---

## TL;DR

| Experiment | Search space | Budget | Best objective | Comparison |
|---|---|---|---|---|
| **Branin (simulation)** | 2-D continuous | 40 trials | $-0.43$ (true min $-0.398$) | BO converges; grid search plateaus far above |
| **Experiment 1 — NCF on MovieLens 1M** | 5-D mixed | 15 trials (~5.5 GPU hrs) | NDCG@10 = **0.4056** | **+8.3 %** vs best-of-5 random init |
| **Experiment 2 — Upworthy classifier thresholds** | 2-D mixed | 60 trials | F = **1568.7** | **BO 1568.7 > Random 1514.0 > Grid 1435.2** |

Full write-up: see notebooks (suggested order below).

---

## Quick Start

```bash
# 1. Create the conda environment
conda env create -f environment.yaml
conda activate stat238-bo

# 2. Open the notebooks in order
jupyter lab notebooks/
```

Suggested reading order:

1. `notebooks/introduction.ipynb` — project motivation and high-level results
2. `notebooks/00_branin_simulation.ipynb` — BO sanity check on a known 2-D function
3. `notebooks/01_ncf_bo.ipynb` — Experiment 1 (NCF / MovieLens 1M)
4. `notebooks/02_upworthy_bo.ipynb` — Experiment 2 (Upworthy classifier)

All notebooks load pre-computed results from `results/` so they render in seconds —
no GPU required.

---

## Repository Structure

```
Bayesian-Optimization-for-ML/
├── README.md
├── environment.yaml
├── documents/
│   ├── STAT238_Final_Project_Proposal.pdf   # original proposal
│   └── ncf_bridge.md                        # NCF black-box interface spec
├── data/                                    # ← input data for the experiments
│   ├── README.md                            # provenance + column descriptions
│   └── upworthy/
│       ├── categories_raw.csv               # BART zero-shot category preds
│       └── confirmatory_clean.csv           # headline + log-CTR (trimmed)
├── src/                                     # ← BO implementation (from scratch)
│   ├── gp.py                                # GP with SE kernel, Cholesky solver,
│   │                                        # marginal-likelihood hyperparam fitting
│   ├── acquisition.py                       # Expected Improvement (closed form)
│   │                                        # + L-BFGS-B maximisation on [0,1]^d
│   ├── bo.py                                # BO loop: random init → fit GP →
│   │                                        # maximise EI → evaluate → repeat
│   ├── black_box_ncf.py                     # NCF training wrapper (CLI subprocess)
│   └── black_box_upworthy.py                # Upworthy ANOVA F-stat objective
├── jobs/                                    # ← runner scripts
│   ├── run_bo_ncf.py                        # ran on SCF GPU cluster
│   └── run_bo_upworthy.py                   # ran locally + baselines (random / grid)
├── notebooks/                               # ← analysis + figures
│   ├── introduction.ipynb
│   ├── 00_branin_simulation.ipynb
│   ├── 01_ncf_bo.ipynb
│   └── 02_upworthy_bo.ipynb
├── results/                                 # raw trial logs (CSV, one row per trial)
│   ├── ncf/trials.csv
│   └── upworthy/{bo_trials,random_trials,grid_trials}.csv
└── figures/                                 # PNGs exported from the notebooks
```

---

## Method

We model the black-box objective $f : \mathcal{X} \to \mathbb{R}$ as a Gaussian process
with zero mean and a squared-exponential kernel:

$$K(x, x') = \alpha \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right).$$

Observations are noisy, $y_i = f(x_i) + \varepsilon_i$ with
$\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.

Given $n$ observations, the GP posterior at a new point $x$ is Gaussian with

$$\mu(x) = \mathbf{k}(x)^\top \bigl(K + \sigma^2 I\bigr)^{-1} y,\qquad
v(x) = K(x,x) - \mathbf{k}(x)^\top \bigl(K + \sigma^2 I\bigr)^{-1} \mathbf{k}(x).$$

Hyperparameters $(\alpha, \ell, \sigma^2)$ are refit at every BO iteration by maximising
the log marginal likelihood. The next point to evaluate is chosen by maximising
**Expected Improvement**:

$$\mathrm{EI}(x) = (\mu(x) - f^\star)\Phi(Z) + \sqrt{v(x)}\,\phi(Z),
\qquad Z = \frac{\mu(x) - f^\star}{\sqrt{v(x)}}$$

where $f^\star$ is the best observation so far. EI is maximised over the unit cube via
multi-start L-BFGS-B (`scipy.optimize.minimize`).

The full BO loop is:

```
1. Evaluate f at n_init random configurations  →  dataset D
2. For t = n_init + 1, …, T:
     a. Fit GP on D (refit hyperparams via marginal likelihood)
     b. x_t = argmax_x EI(x ; GP)
     c. y_t = f(x_t)              ← expensive black-box call
     d. D ← D ∪ {(x_t, y_t)}
3. Return x* = argmax_t y_t
```

Inputs are normalised to $[0,1]^d$ before the GP sees them; outputs are standardised to
zero mean and unit variance during fitting and inverted before reporting. See
`src/bo.py` for details.

---

## Reproducing the results

**Branin and Upworthy** run locally in seconds to minutes — open the notebooks and
re-run all cells. The Upworthy experiment depends on the two CSVs in
`data/upworthy/` (~11 MB total, included in this repo — see `data/README.md`).

**NCF** requires a GPU. The exact run used an RTX 2080 Ti on the Berkeley SCF cluster,
with each trial taking ~22 minutes (10 epochs on MovieLens 1M).

```bash
# On a GPU node (after activating an env with torch installed)
python jobs/run_bo_ncf.py --budget 15 --n-init 5
```

The runner appends to `results/ncf/trials.csv` after every trial so it is safe to
interrupt and resume.

---

## References

- Frazier, P. (2018). *A Tutorial on Bayesian Optimization.* arXiv:1807.02811.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization of
  Machine Learning Algorithms.* NeurIPS 2012.
- Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics.*
  Cambridge University Press.
- Guntuboyina, A. (2026). *STAT 238 Lecture Notes, Lectures 22–23.* UC Berkeley.
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017).
  *Neural Collaborative Filtering.* WWW 2017.
- Salganik, M. J. et al. (2020). *Measuring the predictability of life outcomes with a
  scientific mass collaboration.* PNAS 117(15).
