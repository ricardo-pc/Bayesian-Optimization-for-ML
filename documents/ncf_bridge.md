# NCF Bridge — How BO Calls the CS 289A Training Pipeline

This document is the interface spec between the STAT 238 BO code and the CS 289A
training pipeline. The BO loop treats `train.py` as a black-box function: it passes
in a hyperparameter config, waits for the process to finish, and reads back the
validation NDCG@10.

---

## CS 289A Repo Location

```
~/cs289-ranking/                      (on SCF)
~/Documents/.../cs289-ranking/        (local)
```

The relevant script is `src/train.py`. It accepts all hyperparameters as CLI flags
and prints the final val NDCG@10 to stdout.

---

## Hyperparameter Search Space

| Parameter | Flag | Type | Search Range | Notes |
|---|---|---|---|---|
| Embedding dimension | `--emb-dim` | integer | {32, 64, 128, 256} | size of p_u and q_i |
| MLP hidden layers | `--mlp-layers` | categorical | see below | NCF tower sizes |
| Learning rate | `--lr` | log-continuous | [1e-4, 1e-2] | Adam lr |
| L2 weight decay | `--l2` | log-continuous | [1e-6, 1e-3] | ridge on all params |
| WMF confidence scale | `--alpha` | continuous | [0.5, 5.0] | c_ui = 1 + alpha * rating |

MLP layer options:
```python
MLP_CONFIGS = [
    [128, 64],
    [256, 128, 64],
    [256, 128, 64, 32],
]
```

Fixed (not tuned by BO):
- `--model ncf`
- `--density 1.0` — BO tunes on full-density data only
- `--epochs 20`
- `--batch-size 1024`
- `--device cuda`
- `--n-neg 4`

---

## How to Call train.py from Python

```python
import subprocess
import re

def evaluate_ncf(config: dict, cs289_repo: str) -> float:
    """
    Run one NCF training trial and return val NDCG@10.

    config keys: emb_dim (int), mlp_layers (list), lr (float), l2 (float), alpha (float)
    cs289_repo: absolute path to the cs289-ranking repo root
    """
    cmd = [
        "python", "src/train.py",
        "--model",      "ncf",
        "--density",    "1.0",
        "--epochs",     "20",
        "--batch-size", "1024",
        "--device",     "cuda",
        "--emb-dim",    str(config["emb_dim"]),
        "--mlp-layers", *[str(h) for h in config["mlp_layers"]],
        "--lr",         str(config["lr"]),
        "--l2",         str(config["l2"]),
        "--alpha",      str(config["alpha"]),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cs289_repo,   # run from repo root so src/ imports work
    )

    # Parse NDCG@10 from the last output line:
    # "Best val  NDCG@10 = 0.4821  HR@10 = 0.7103"
    match = re.search(r"NDCG@10\s*=\s*([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse NDCG@10 from output:\n{result.stdout}\n{result.stderr}")

    return float(match.group(1))
```

---

## Expected Runtime

| Setup | Time per trial |
|---|---|
| SCF A4000/A5000 (cuda) | ~3-5 min |
| Local CPU (MacBook) | ~45-60 min — do not use for BO |

Total BO budget: 25-30 trials → ~2-2.5 hours on SCF GPU.

---

## What train.py Outputs

The last two lines of stdout are always:
```
Best val  NDCG@10 = 0.4821  HR@10 = 0.7103
Checkpoint saved to checkpoints/ncf_density1.0.pt
```

The checkpoint is saved inside the cs289-ranking repo at `checkpoints/ncf_density1.0.pt`.
After BO finishes, the best config is rerun manually to produce the final checkpoint used
in the CS 289A sparsity sweep.

---

## Encoding Categorical Parameters for the GP

The GP kernel operates on continuous vectors. Categorical parameters (emb_dim, mlp_layers)
need to be encoded before being passed to the GP.

Suggested encoding:
```python
# emb_dim: map to log scale so distances are meaningful
emb_dim_to_x = {32: 0.0, 64: 1.0, 128: 2.0, 256: 3.0}

# mlp_layers: ordinal by total parameter count
mlp_to_x = {
    (128, 64):        0.0,
    (256, 128, 64):   1.0,
    (256, 128, 64, 32): 2.0,
}

# lr, l2: log-transform so the GP sees a roughly uniform scale
x_lr = log(lr)   # range: [log(1e-4), log(1e-2)] = [-9.2, -4.6]
x_l2 = log(l2)   # range: [log(1e-6), log(1e-3)] = [-13.8, -6.9]

# alpha: use as-is, range [0.5, 5.0]
```

Full input vector to GP: `x = [emb_dim_x, mlp_x, x_lr, x_l2, alpha]` — shape (5,)

---

## Results Logging

Each trial should be logged to `results/ncf/trials.csv`:
```
trial, emb_dim, mlp_layers, lr, l2, alpha, val_ndcg, runtime_s
1, 64, "[256,128,64]", 0.001, 1e-05, 1.0, 0.4231, 243
2, 128, "[256,128,64]", 0.0005, 1e-05, 2.0, 0.4512, 251
...
```

This log is what gets plotted as the convergence curve (best NDCG seen so far vs trial number).

---

## Connection to CS 289A Sparsity Analysis

BO runs once at density=1.0 to find θ* = best hyperparameter config.
θ* is then used to retrain NCF at all 5 density levels:

```
density = 1.0  →  retrain NCF at θ*  →  test NDCG@10
density = 0.8  →  retrain NCF at θ*  →  test NDCG@10
density = 0.6  →  retrain NCF at θ*  →  test NDCG@10
density = 0.4  →  retrain NCF at θ*  →  test NDCG@10
density = 0.2  →  retrain NCF at θ*  →  test NDCG@10
```

The 5 test NDCG@10 values form the NCF line in the main figure of the CS 289A paper.
