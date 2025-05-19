import numpy as np
import pandas as pd

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
df["F"] = [0, 1, 2, 3, 4, 5]
df[df > 0] = -df
"""
For NumPy data types, np.nan represents missing data. It is bu default not included in computations
Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data
"""
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0]: dates[1], "E"] = 1

# print(df1)

"""
DataFrame.dropna() drops any rows that have missing data
"""

# print(df1.dropna(how="any"))

"""
DataFrame.fillna() fills missing data:
"""
# print(df1.fillna(value=5))

"""
DataFrame.isna() return the boolean mask where values are nan
"""
print(pd.isna(df1))  # df1.isna()
