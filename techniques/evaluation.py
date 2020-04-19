"""
Evaluation
"""


def true_positives(true : list, estimate : list) -> float:
    f = [1 if a == b else 0 for a, b in zip(true, estimate)]
    return sum(f)   
