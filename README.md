# Bayesian Optimization for Expensive Black-Box ML Pipelines

STAT 238, UC Berkeley · Spring 2026

**Core question:** How much more sample-efficient is Bayesian Optimization compared to
random search and grid search when each function evaluation takes minutes?

- Full methodology and design decisions: [`documents/STAT238_Final_Project_Proposal.pdf`](documents/STAT238_Final_Project_Proposal.pdf)
- NCF black-box interface spec: [`documents/ncf_bridge.md`](documents/ncf_bridge.md)

---

## Experiments

Three experiments, in priority order:

**0. Simulation — Branin function**  
Verify the BO implementation on a 2D benchmark with a known global minimum before
touching real data. Visualize how the GP surrogate and EI acquisition function evolve
across iterations. The Branin function is:

$$f(x_1, x_2) = a(x_2 - bx_1^2 + cx_1 - r)^2 + s(1-t)\cos(x_1) + s$$

with standard constants $a=1,\ b=5.1/(4\pi^2),\ c=5/\pi,\ r=6,\ s=10,\ t=1/(8\pi)$
and global minimum $f^\star \approx 0.397$ at three locations.

**Experiment 1 — NCF hyperparameter tuning (MovieLens 1M)**  
Tune embedding dimension, MLP hidden layers, learning rate, L2 regularization, and WMF
confidence scaling for Neural Collaborative Filtering. Objective: validation NDCG@10.
Approximately 4 min per trial on GPU, 25–30 trials. Best config feeds directly into the
CS 289A sparsity analysis.

**Experiment 2 — Upworthy zero-shot classification pipeline**  
Tune the candidate label set, number of categories $K$, and confidence threshold for
`facebook/bart-large-mnli`. Objective: $F$-statistic from one-way ANOVA of $\log(\text{CTR})$
on inferred labels. Approximately 30 s per trial, 60–80 trials.

---

## Folder Structure

```
Bayesian-Optimization-for-ML/
├── documents/
│   ├── STAT238_Final_Project_Proposal.pdf
│   ├── ProjectTopics238Spring2026.pdf
│   └── ncf_bridge.md          # how BO calls train.py and parses NDCG@10
├── notebooks/
│   ├── 00_branin_simulation.ipynb   # sanity check on known 2D benchmark
│   ├── 01_ncf_bo.ipynb              # Experiment 1: NCF on MovieLens 1M
│   └── 02_upworthy_bo.ipynb         # Experiment 2: Upworthy zero-shot pipeline
├── src/
│   ├── gp.py                  # GP surrogate: SE kernel, marginal likelihood, posterior
│   ├── acquisition.py         # Expected Improvement (EI), closed form
│   ├── bo.py                  # BO loop: initialize → fit GP → maximize EI → evaluate
│   ├── black_box_ncf.py       # calls cs289-ranking/src/train.py, parses NDCG@10
│   └── black_box_upworthy.py  # BART zero-shot pipeline, returns F-statistic
├── results/
│   ├── ncf/                   # trial logs: trials.csv with (config, NDCG@10, runtime)
│   └── upworthy/              # trial logs: trials.csv with (config, F-statistic, runtime)
├── figures/                   # convergence curves, GP surrogate plots
└── environment.yml
```

---

## Methods

### GP Surrogate (Lecture 23, STAT 238)

We model the black-box objective $f : \mathcal{X} \to \mathbb{R}$ as a Gaussian process
with mean zero and squared exponential (SE) kernel:

$$K(x, x') = \alpha \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

where $\alpha > 0$ is the output scale and $\ell > 0$ is the lengthscale. Observations
are noisy: $y_i = f(x_i) + \varepsilon_i$ with $\varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$.

Given $n$ observations $(x_1, y_1), \ldots, (x_n, y_n)$, define

$$K = \bigl(K(x_i, x_j)\bigr)_{n \times n}, \qquad \mathbf{k}(x) = \bigl(K(x_i, x)\bigr)_{n \times 1}.$$

The posterior distribution of $f(x)$ given the data is (Lecture 23, Eq. 2):

$$f(x) \mid \text{data} \sim \mathcal{N}\!\left(\mu(x),\ v(x)\right)$$

with

$$\mu(x) = \mathbf{k}(x)^\top \bigl(K + \sigma^2 I_n\bigr)^{-1} y$$

$$v(x) = K(x, x) - \mathbf{k}(x)^\top \bigl(K + \sigma^2 I_n\bigr)^{-1} \mathbf{k}(x).$$

Hyperparameters $(\alpha, \ell, \sigma^2)$ are fit at each BO iteration by maximizing the
log marginal likelihood:

$$\log p(y) = -\frac{1}{2}\, y^\top \bigl(K + \sigma^2 I_n\bigr)^{-1} y
              - \frac{1}{2} \log \bigl|K + \sigma^2 I_n\bigr|
              - \frac{n}{2} \log 2\pi.$$

### Acquisition Function — Expected Improvement

Let $f^\star = \max_{i} y_i$ be the best observation so far. The Expected Improvement
acquisition function is:

$$\mathrm{EI}(x) = \bigl(\mu(x) - f^\star\bigr)\,\Phi(Z) + \sqrt{v(x)}\,\phi(Z), \qquad
Z = \frac{\mu(x) - f^\star}{\sqrt{v(x)}}$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF. The term $\mu(x) - f^\star$
drives exploitation; $\sqrt{v(x)}$ drives exploration. EI is maximized numerically using
BoTorch.

### BO Loop

1. Draw $n_0 = 5$ random initial configurations and evaluate the black box.
2. Repeat until budget exhausted:
   - Fit GP hyperparameters $(\alpha, \ell, \sigma^2)$ by maximizing $\log p(y)$.
   - Find $x_{\text{next}} = \arg\max_x \mathrm{EI}(x)$ via BoTorch.
   - Evaluate $y_{\text{next}} = f(x_{\text{next}})$ (expensive black-box call).
   - Append $(x_{\text{next}}, y_{\text{next}})$ to the observation set.

### Baselines

Random search and grid search at matched evaluation budgets (25–30 trials), compared via
convergence curves: best $y$ seen so far vs. trial number.

### NCF Search Space

The GP operates on a 5-dimensional continuous vector $x \in \mathcal{X}$:

| Parameter | Encoding | Range |
|---|---|---|
| Embedding dim $d$ | $\log_2(d) - 5$ (ordinal) | $\{0, 1, 2, 3\}$ |
| MLP config | ordinal by depth | $\{0, 1, 2\}$ |
| Learning rate $\eta$ | $\log(\eta)$ | $[\log 10^{-4},\ \log 10^{-2}]$ |
| L2 weight decay $\lambda$ | $\log(\lambda)$ | $[\log 10^{-6},\ \log 10^{-3}]$ |
| WMF scale $\alpha_{\text{wmf}}$ | as-is | $[0.5,\ 5.0]$ |

Log-transforming $\eta$ and $\lambda$ ensures the SE kernel assigns meaningful distances
across several orders of magnitude.

---

## References

- Frazier (2018) — A Tutorial on Bayesian Optimization. *arXiv:1807.02811*
- Snoek, Larochelle, Adams (2012) — Practical Bayesian Optimization of Machine Learning Algorithms. *NeurIPS 2012*
- Hennig, Osborne, Kersting (2022) — *Probabilistic Numerics*. Cambridge University Press
- Guntuboyina (2026) — STAT 238 Lecture Notes, Lectures 22–23. UC Berkeley
