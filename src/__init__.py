# src package — shared BO engine
# Modules are imported on demand so the package can be used while
# acquisition.py and bo.py are still being written.
from .gp import GaussianProcess

def __getattr__(name):
    if name == "expected_improvement":
        from .acquisition import expected_improvement
        return expected_improvement
    if name == "BayesianOptimizer":
        from .bo import BayesianOptimizer
        return BayesianOptimizer
    raise AttributeError(f"module 'src' has no attribute {name!r}")
