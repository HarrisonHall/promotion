"""
Harrison Hall
CPSC 4820 - P5

Main file for promotion.
"""

import argparse

from techniques.tools import *
from techniques.reading import *
from techniques.regression import *
from techniques.logistic import *
from techniques.evaluation import *
from techniques.cluster import *
from techniques.knn import *


default_train_fname = "data/train.csv"
default_test_fname = "data/test.csv"
default_promotion = "is_promoted"

"""
solutions = pd.DataFame({
    "method": [],
    "is_promoted": []
})
"""


parser = argparse.ArgumentParser(description="Commands for promotion")
parser.add_argument(
    "-S", "-s", "--split",
    action="store_true",
    help="Split the training data into a test set"
)
parser.add_argument(
    "-R", "-r", "--regression",
    action="store_true",
    help="Do linear regression"
)
parser.add_argument(
    "-L", "-l", "--logistic",
    action="store_true",
    help="Do logistic regression"
)
parser.add_argument(
    "-C", "-c", "--clustering",
    action="store_true",
    help="Do clustering"
)
parser.add_argument(
    "-K", "-k", "--knn",
    action="store_true",
    help="Do K-nearest-neighbor"
)
parser.add_argument(
    "-?", "--willibepromoted",
    action="store_true",
    help="Ask question"
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.split:
        print_title("Command: Split Data")
        a, b = split_data(read_data(input("Splitting data name >> ")))
        write_csv(a, input("Training filename >> "))
        write_csv(b, input("Testing filename >> "))
    else:
        print_title("Promotion Procedure")
        train_fname = input("Training data file >> ")
        test_fname = input("Testing data file >> ")
        
        # Apply defaults
        train_fname = train_fname if train_fname != "" else default_train_fname
        test_fname = test_fname if test_fname != "" else default_test_fname
        
        train_data = make_numeric(read_data(train_fname))
        x, y = split_on(train_data, default_promotion)
        test_data = make_numeric(read_data(test_fname))
        x2, y2 = split_on(test_data, default_promotion)

        if args.regression:
            # Linear regression
            print_title("Linear regression")
            lr_w = lr_weights(x, y)
            vals = lr_apply(x2,lr_w)
            print_measures(vals, y2)
        if args.logistic:
            # Logistic regression
            print_title("Logistic regression")
            log_w = log_weights(x,y)
            vals = log_apply(x2, log_w)
            print_measures(vals, y2)
        if args.clustering:
            # Clustering
            print_title("Clustering")
            means = c_model(x, y)
            vals = c_apply(x, means)
            print_measures(vals, y2)
        if args.knn:
            # Knn
            print_title("K-nearest-neighbor")
            mod = kd_model(x, y)
            vals = kd_apply(x2, y, mod)
            print_measures(vals, y2)
        if args.willibepromoted:
            print("?")
    print("Finished")
