"""
Quadratic Discriminant Analysis
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda

def qd_model(x, y):
    q = qda()
    return q.fit(x,y)

def qd_estimate(x, q, raw=False):
    if raw:
        return [v for v in q.predict(x)]
    return [0 if v < .5 else 1 for v in q.predict(x)]
