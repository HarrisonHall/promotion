"""
Logistic regression technique
"""

from sklearn.linear_model import LogisticRegression as lr
import pandas as pd


def log_weights(x : pd.DataFrame, y : pd.DataFrame):
    clf = lr(random_state=1).fit(x,y)
    return clf

def log_apply(x, clf):
    z = clf.predict(x)
    return [0 if v < .5 else 1 for v in z]
