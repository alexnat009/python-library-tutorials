import numpy as np
import pandas as pd

df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)

print(df)

"""
CATEGORICALS

pandas can include categorical data in a DataFrame.
"""

df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])

new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)

print(df["grade"])

df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)

print(df["grade"])
print(df.sort_values(by="grade"))
print(df.groupby("grade", observed=False).size())