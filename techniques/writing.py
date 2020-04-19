"""
Writing.
"""

import pandas as pd



def write_csv(df : pd.DataFrame, path : str, index=False) -> None:
    df.to_csv(path, index=index)
