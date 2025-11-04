import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for simple drift watching."""
    cuts = np.quantile(expected.dropna(), np.linspace(0,1,bins+1))
    cuts[0] -= 1e-9; cuts[-1] += 1e-9
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e = np.where(e_counts==0, 1e-6, e_counts) / max(e_counts.sum(), 1)
    a = np.where(a_counts==0, 1e-6, a_counts) / max(a_counts.sum(), 1)
    idx = ((a - e) * np.log(a/e)).sum()
    return float(idx)