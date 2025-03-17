import numpy as np
import pandas as pd

dates = pd.date_range("20200101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

"""
Operations in general exclude missing data
"""

# print(df.mean())  # Calculate the mean value for each column

# print(df.mean(axis=1))  # Calculate the mean for each row

"""
Operation with another Series or DataFrame with a different index or column will align
the result with the union of the index or column labels. In addition, pandas automatically broadcasts
along the specified dimension and will fill unaligned labels with np.nan
"""

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
# print(s)
#
# print(df.sub(s, axis="columns"))


# USER DEFINED FUNCTIONS

"""
DataFrame.agg() and DataFrame.transform() applies a user defined function that reduces
or broadcasts its result respectively
"""
# print(df.mean())
# print(df.agg(lambda x: np.mean(x) * 5.6))
# print(df.transform(lambda x: x * 100))

"""
value counts
"""

s = pd.Series(np.random.randint(0, 7, size=10))
# print(s)
# print(s.value_counts())


"""
String Methods:
Series is equipped with a set of string processing methods in the str attribute that make it
easy to operate on each element of the array 
"""

s = pd.Series(["A", "B", "C", "AbA", "Bacca", np.nan, "CABA", "dog", "cat"])
print(s.str.lower())
