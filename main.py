"""
Harrison Hall
CPSC 4820 - P5

Main file for promotion.
"""

import argparse

import warnings
from sklearn.exceptions import DataConversionWarning
from random import randint

from techniques.tools import *
from techniques.reading import *
from techniques.regression import *
from techniques.quadratic import *
from techniques.logistic import *
from techniques.evaluation import *
from techniques.cluster import *
from techniques.knn import *
from techniques.qda import *


default_train_fname = "data/train.csv"
default_test_fname = "data/test.csv"
default_write_fname = "data/evaluation.csv"
default_promotion = "is_promoted"


parser = argparse.ArgumentParser(
    description="Commands for using promotion."
)
parser.add_argument(
    "-S", "-s", "--split",
    action="store_true",
    help="Split the training data into a test set"
)
parser.add_argument(
    "-V", "-v", "--verbose",
    action="store_true",
    help="Be verbose in printing output"
)
parser.add_argument(
    "-R", "-r", "--regression",
    action="store_true",
    help="Do linear regression"
)
parser.add_argument(
    "-F", "-f", "--quadratic",
    action="store_true",
    help="Do linear regression on quadratic features"
)
parser.add_argument(
    "-U", "-u", "--cubic",
    action="store_true",
    help="Do linear regression on cubic features"
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
    "-Q", "-q", "--qda",
    action="store_true",
    help="Do quadratic discriminant analysis"
)
parser.add_argument(
    "-A", "-a", "--rall",
    action="store_true",
    help="Do regression on all"
)
parser.add_argument(
    "-?", "--willibepromoted",
    action="store_true",
    help="Find out whether or not you'll be promoted"
)
args = parser.parse_args()

if __name__ == "__main__":
    if not args.verbose:
        warnings.filterwarnings(
            action='ignore', category=DataConversionWarning
        )
    if args.split:
        print_title("Command: Split Data")
        a, b = split_data(read_data(input("Splitting data name >> ")))
        write_csv(a, input("Training filename >> "))
        write_csv(b, input("Testing filename >> "))
    else:
        print_title("Promotion Procedure")
        train_fname = input("Training data file >> ")
        test_fname = input("Testing data file >> ")
        write_fname = input("*Analysis data file >> ")
        
        # Apply defaults
        train_fname = (
            train_fname if train_fname != "" else default_train_fname
        )
        test_fname = (
            test_fname if test_fname != "" else default_test_fname
        )
        write_fname = (
            write_fname if write_fname != "" else default_write_fname
        )
        
        train_data = make_numeric(read_data(train_fname))
        x, y = split_on(train_data, default_promotion)
        test_data = make_numeric(read_data(test_fname))
        x2, y2 = split_on(test_data, default_promotion)
        x_all = pd.DataFrame()
        x2_all = pd.DataFrame()

        if args.willibepromoted:
            # Preliminary questions
            print_title("Questions")
            l = len(x2)-1
            r = float(input("Age >> "))
            x2.at[l,"age"] = r
            r = float(input("Rating >> "))
            x2.at[l,"previous_year_rating"] = r
            r = float(input("Years Worked >> "))
            x2.at[l,"length_of_service"] = r
            field = input(
                "Field (Analytics, Finance, HR, Legal, "
                "Operations, HR, Procurement, "
                "Sales & Marketing, Technology)\n>> "
            )
            x2.at[l,"Procurement"] = 0.0
            x2.at[l,field] = 1.0
            r = float(input("Number of trainings >> "))
            x2.at[l,"no_of_trainings"] = r
            r = float(input("Bachelors? >> "))
            x2.at[l,"Bachelor's"] = r
            g = input("Sex (m/f) >> ")
            x2.at[l,"m"] = 0.0
            x2.at[l,"f"] = 0.0
            x2.at[l,g] = 1.0
            x2.at[l,"employee_id"] = randint(0,1000000)
            x2.at[l,"region_7"] = 0.0
            x2.at[l,f"region_{randint(1,25)}"] = 1.0

        if args.regression:
            # Linear regression
            print_title("Linear regression")
            lr_w = lr_weights(x, y)
            vals = lr_apply(x2,lr_w)
            print_measures(vals, y2)
            x_all["linear"] = lr_apply(x,lr_w,raw=True)
            x2_all["linear"] = lr_apply(x2,lr_w,raw=True)
        if args.quadratic:
            print_title("Quadratic regression")
            w = q_model(x,y)
            vals = q_estimate(x2, w)
            print_measures(vals, y2)
            x_all["quad"] = q_estimate(x, w, raw=True)
            x2_all["quad"] = q_estimate(x2, w, raw=True)
        if args.cubic:
            print_title("Cubic regression")
            print("Skipped: too much memory")
            """ Uses too much memory
            w = q_model(x,y, n=3)
            vals = q_estimate(x2, w, n=3)
            print_measures(vals, y2)
            x_all["cub"] = q_estimate(x, w, n=3)
            x2_all["cub"] = vals
            """
        if args.logistic:
            # Logistic regression
            print_title("Logistic regression")
            log_w = log_weights(x,y)
            vals = log_apply(x2, log_w)
            print_measures(vals, y2)
            x_all["log"] = log_apply(x,log_w)
            x2_all["log"] = log_apply(x2,log_w)
        if args.clustering:
            # Clustering
            print_title("Clustering")
            means = c_model(x, y)
            vals = c_apply(x, means)
            print_measures(vals, y2)
            x_all["cluster"] = c_apply(x,means)
            x2_all["cluster"] = c_apply(x2,means)
        if args.knn:
            # Knn
            print_title("K-nearest-neighbor")
            mod = kd_model(x, y)
            vals = kd_apply(x2, y, mod)
            print_measures(vals, y2)
            x_all["knn"] = kd_apply(x, y, mod)
            x2_all["knn"] = vals
        if args.qda:
            # qda
            print_title("Quadratic Discriminant Analysis")
            model = qd_model(x,y["is_promoted"].values)
            vals = qd_estimate(x2, model)
            print_measures(vals, y2)
            x_all["qda"] = qd_estimate(x,model)
            x2_all["qda"] = vals
        if args.rall or args.willibepromoted:
            # use all
            print_title("Regression of previous")
            model = q_model(x_all, y, n=3)
            vals = q_estimate(x2_all, model, n=3)
            print_measures(vals, y2)
            if args.willibepromoted:
                print_title("Will I be promoted?")
                a = q_estimate(x2_all, model, n=3, raw=True)[-1]
                print(f"Raw {a[0]}")
                if a[0] > .5:
                    print("Promoted!")
                else:
                    print("Better luck next year.")
