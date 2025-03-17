import numpy as np
import pandas as pd

dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

"""
For a DataFrame, passing a single label selects a columns and yields
a Series equivalent to df.A
"""
# print(df["A"])

"""
For a DataFrame, passing a slice : selects matching rows
"""

# print(df[0:2])
# print(df['20200102':'20200104'])


# SELECTING BY LABEL
"""
Selecting a row matching a label
"""
# print(dates[0])
# print(df.loc[dates[0]])
"""
Selecting all rows : with a select column labels
"""

# print(df.loc[:, ["A", "B"]])


"""
For label slicing, both endpoints are included
"""
# print(df.loc["20200102":"20200105", ["A", "B"]])

"""
Selecting a single row and column label returns a scalar
"""

# print(df.loc[dates[0], "A"])

"""
For getting fast access to a scalar, (IN CASE OF ONE GETTING ONE CONCRETE VALUE)
you can use  df.at
"""

# print(df.at[dates[0], "A"])


# SELECTING BY POSITION
"""
selecting a row matching a label
"""
# print(df.iloc[3])

"""
Integer slices acts similar to Numpy
"""
# print(df.iloc[3:5, 0:2])

"""
Lists of integer position locations
"""
# print(df.iloc[[0, 4, 2], [2, 1]])

"""
For slicing rows explicitly
"""
# print(df.iloc[1:3, :])

"""
For slicing columns explicitly
"""

# print(df.iloc[:, 1:3])

"""
For getting a value explicitly
"""

# print(df.iloc[1, 1])

"""
For getting fast access to a scalar use iat method
"""
# print(df.iat[1, 1])


# BOOLEAN INDEXING

"""
Select rows where df.A is greater than 0
"""

# print(df[df["A"] > 0])

"""
Selecting values from a DataFrame where a boolean condition is met
"""

# print(df[df > 0])

"""
Using isin() method for filtering
"""

df1 = df.copy()

df1["E"] = ["one", "one", "two", "three", "four", "three"]

# print(df1)

# print(df1[df1["E"].isin(["two", "four"])])


# SETTING

"""
Setting a new column automatically aligns the data by the indexes:
"""

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20200101', periods=6))
# print(s1)

df["F"] = s1
df.at[dates[0], "A"] = 0  # Setting values by label
df.iat[0, 1] = 10  # Setting values by position
df.loc[:, "D"] = np.array([5] * len(df))
# print(df)

"""
A where operation with setting:
"""

df1 = df.copy()

df1[df1 > 0] = -df1

print(df1)