"""
Tools
"""

from techniques.evaluation import *

def print_title(title):
    print(
        "\n"+
        "="*len(title)+"\n"+
        title + "\n"
        + "="*len(title)
    )

def print_measures(vals, y2):
    #print("Number correct: ",num_correct(vals,y2["is_promoted"]))
    #print("Percent correct: ", percent_correct(vals,y2["is_promoted"]))
    print("TP: ",true_positives(vals,y2["is_promoted"]))
    print("TN: ",true_negatives(vals,y2["is_promoted"]))
    print("FP: ",false_positives(vals,y2["is_promoted"]))
    print("FN: ",false_negatives(vals,y2["is_promoted"]))
    print("Precision: ",precision(vals,y2["is_promoted"]))
    print("Accuracy: ",accuracy(vals,y2["is_promoted"]))
    print("Recall: ",recall(vals,y2["is_promoted"]))
    print("F1: ",f1(vals,y2["is_promoted"]))
