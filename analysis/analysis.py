"""
Analysis
"""

import pandas as pd


data = pd.read_csv("data/old_train.csv")
print(data.columns)

# Department and age
d = {}
ages = {}
total = 0
for i in range(len(data)):
    if data.at[i, "is_promoted"] == 1:
        total += 1
        dep = data.at[i, "department"]
        age = data.at[i, "age"]
        if dep in d:
            d[dep] += 1
        else:
            d[dep] = 1
        if age in ages:
            ages[age] += 1
        else:
            ages[age] = 1
print(d)
print(ages)

for i in range(5):
    f = {}
    for age in ages:
        f[ages[age]] = age
    best = f[max(list(f.keys()))]
    print(f"Best age is {best} with count of {ages[best]}")
    ages.pop(best)
print(f"Total {total}")
