"""
Evaluation
"""

def num_correct(true : list, estimate : list) -> float:
    f = [1 if (a == b) else 0 for a, b in zip(true, estimate)]
    return sum(f)

def percent_correct(true : list, estimate : list) -> float:
    f = [1 if (a == b) else 0 for a, b in zip(true, estimate)]
    return sum(f)/len(f)

def true_positives(true : list, estimate : list) -> float:
    f = [1 if a == 1 and b == 1 else 0 for a, b in zip(true, estimate)]
    return sum(f)

def true_negatives(true : list, estimate : list) -> float:
    f = [1 if a == 0 and b == 0 else 0 for a, b in zip(true, estimate)]
    return sum(f)

def false_positives(true : list, estimate : list) -> float:
    f = [1 if a == 1 and b == 0 else 0 for a, b in zip(true, estimate)]
    return sum(f)

def false_negatives(true : list, estimate : list) -> float:
    f = [1 if a == 0 and b == 1 else 0 for a, b in zip(true, estimate)]
    return sum(f)

def precision(true, estimate):
    tp = true_positives(true, estimate)
    fp = false_positives(true, estimate)
    return tp/(tp+fp)

def recall(true, estimate):
    tp = true_positives(true, estimate)
    fn = false_negatives(true, estimate)
    return tp/(tp+fn)

def f1(true, estimate):
    p = precision(true, estimate)
    r = recall(true, estimate)
    return 2*p*r/(p+r)

def accuracy(true, estimate):
    tp = true_positives(true, estimate)
    fn = false_negatives(true, estimate)
    tn = true_negatives(true, estimate)
    fp = false_positives(true, estimate)
    return (tp + tn)/(tp + fn + tn + fp)

