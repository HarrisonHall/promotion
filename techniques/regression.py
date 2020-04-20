"""
Technique for regressions
"""

import pandas as pd
from sklearn.linear_model import LinearRegression as LR


def lr_weights(x : pd.DataFrame, y : pd.DataFrame):
    """
    Assumes last column is desired outcome.
    """
    reg = LR().fit(x, y)
    return reg#.coef_[0]

def lr_estimate(x : pd.DataFrame, y : list) -> list:
    vals = list(x.dot(y))
    return [0 if v < .5 else 1 for v in vals]

def lr_apply(x, reg) -> list:
    z = reg.predict(x)
    red = [a[0] for a in z]
    return [0 if v < .5 else 1 for v in red]
