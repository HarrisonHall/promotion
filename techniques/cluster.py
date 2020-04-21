"""
Clustering techniques
"""

from sklearn.cluster import KMeans as km


def c_model(x, y, n=2):
    means = km(n_clusters=n)
    means.fit(x)
    return means

def c_apply(x, means, raw=False) -> list:
    v = means.fit_predict(x)
    num_ones = sum([1 if a==1 else 0 for a in v])
    num_zeros = sum([1 if a==0 else 0 for a in v])
    # since more not promoted than promoted,
    # smaller cluster should be 1's
    if num_ones < num_zeros:
        return v
    else:
        return [1 if a == 0 else 0 for a in v]
