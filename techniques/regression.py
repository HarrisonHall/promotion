"""
Technique for regressions
"""

import pandas as pd
from sklearn.linear_model import LinearRegression as LR


def lr_weights(x : pd.DataFrame, y : pd.DataFrame) -> list:
    """
    Assumes last column is desired outcome.
    """
    reg = LR().fit(x, y)
    input(str(y))
    return reg.coef_[0]

def lr_estimate(x : pd.DataFrame, y : list) -> list:
    vals = list(x.dot(y))
    print(x.dot(y))
    input()
    return [0 if v < .5 else 1 for v in vals]
