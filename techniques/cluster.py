"""
Clustering techniques
"""

from sklearn.cluster import KMeans as km
from sklearn.cluster import AgglomerativeClustering as ag
from sklearn.cluster import SpectralClustering as sc
from sklearn.cluster import Birch as b
from sklearn.cluster import DBSCAN as db
from sklearn.cluster import FeatureAgglomeration as fa


def c_model(x, y, n=2, use="k"):
    if use == "k":
        means = km(n_clusters=n)
        means = means.fit(x)
        #means.fit(x)
        return means
    if use == "a":
        # killed
        means = ag(n_clusters=n).fit(x)
        print("fit")
        return means
    if use == "s":
        # killed
        means = sc(n_clusters=2).fit(x)
        return means
    if use == "b":
        means = b(n_clusters=2).fit(x)
        return means
    if use == "m":
        means = ms().fit(x)
        return means
    if use == "f":
        means = fa(n_clusters=2).fit(x)
        return means
        

def c_apply(x, means, raw=False) -> list:
    v = means.predict(x)
    num_ones = sum([1 if a==1 else 0 for a in v])
    num_zeros = sum([1 if a==0 else 0 for a in v])
    # since more not promoted than promoted,
    # smaller cluster should be 1's
    if num_ones < num_zeros:
        return v
    else:
        return [1 if a == 0 else 0 for a in v]
