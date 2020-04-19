"""
Harrison Hall
CPSC 4820 - P5

Main file for promotion.
"""

import argparse

from techniques.reading import read_data, split_data
from techniques.reading import make_numeric, split_on
from techniques.reading import num_promoted
from techniques.writing import write_csv
from techniques.regression import lr_weights, lr_estimate
from techniques.evaluation import true_positives


default_train_fname = "data/train.csv"
default_test_fname = "data/test.csv"
default_promotion = "is_promoted"


parser = argparse.ArgumentParser(description="Commands for promotion")
parser.add_argument(
    "-S", "-s", "--split",
    action="store_true",
    help="Split the training data into a test set"
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.split:
        print("Command: Split Data")
        a, b = split_data(read_data(input("Splitting data name >> ")))
        write_csv(a, input("Training filename >> "))
        write_csv(b, input("Testing filename >> "))
    elif False:
        pass
    else:
        print("Promotion Procedure")
        train_fname = input("Training data file >> ")
        test_fname = input("Testing data file >> ")
        
        # Apply defaults
        train_fname = train_fname if train_fname != "" else default_train_fname
        test_fname = test_fname if test_fname != "" else default_test_fname
        
        train_data = make_numeric(read_data(train_fname))
        x, y = split_on(train_data, default_promotion)
        test_data = make_numeric(read_data(test_fname))
        x2, y2 = split_on(test_data, default_promotion)

        # Linear regression
        print("Linear regression")
        lr_w = lr_weights(x, y)
        print(f"Num promoted {num_promoted(train_data, 'is_promoted')}")
        print(lr_w)
        print(lr_estimate(x, lr_w))
        print(true_positives(lr_estimate(x, lr_w),y), len(train_data))
        
    print("Finished")
