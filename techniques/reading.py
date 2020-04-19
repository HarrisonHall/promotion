"""
Tools for reading data.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as ohe
from numpy import issubdtype, number


def read_data(fname : str, csv=True) -> pd.DataFrame:
    if csv:
        return pd.read_csv(fname)
    return None

def split_data(
        df : pd.DataFrame,
        train_size=.8
) -> (pd.DataFrame, pd.DataFrame):
    tr, te = train_test_split(df, train_size=train_size)
    return (tr, te)

def make_numeric(df : pd.DataFrame) -> pd.DataFrame:
    """
    Use one-hot-encoding to make string and other 
    datatypes into numbers.

    Greatly increases matrix size.
    """
    ocols = df.columns
    for column in ocols:
        if not issubdtype(df[column], number):
            df = df.join(pd.get_dummies(df[column])).drop(column,axis=1)
    return df.fillna(0)

def split_on(
        df : pd.DataFrame,
        c : str
) -> (pd.DataFrame,pd.DataFrame):
    return (df.loc[:, df.columns != c], df.loc[:, df.columns == c])

def num_promoted(df : pd.DataFrame, col : str) -> int:
    i = 0
    for j in df[col]:
        if j == 1:
            i += 1
    return i
