"""
Quadratic features regression
"""

from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression as LR

def q_model(x, y, n=2):
    if n <= 2:
        x = pf(degree=n).fit_transform(x)
    else:
        x = pf(degree=n, interaction_only=True).fit_transform(x)
    q = LR().fit(x,y)
    del x
    return q
    
def q_estimate(x, mod, n=2, raw=False):
    if n <= 2:
        x = pf(degree=n).fit_transform(x)
    else:
        x = pf(degree=n, interaction_only=True).fit_transform(x)
    p = mod.predict(x)
    del x
    if raw:
        return p
    return [0 if v[0] < .5 else 1 for v in p]
