"""
K-nearest-neighbor technique. 
"""

from sklearn.neighbors import NearestNeighbors as nn
from sklearn.neighbors import KDTree as kd


def kd_model(x, y):
    tree = kd(x)
    return tree

def kd_apply(x, y, tree, k=1):
    #avg = lambda a: sum(a)/len(a)
    r = [0]*len(x)
    for i in range(len(x)):
        #print(x[i:i+1])
        d, j = tree.query(x[i:i+1])
        r[i] = int(y["is_promoted"][j[0]])
    return r
