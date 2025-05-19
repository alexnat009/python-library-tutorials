import numpy as np
import pandas as pd

"""
Creating Series by passing a list of values, letting pandas create a default RangeIndex
"""
s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

"""
Creating DataFrame by passing a NumPy array with a datatime index using data_range()
and labeled columns:
"""

dates = pd.date_range('20130101', periods=6)
# print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
# print(df)

"""
Creating DataFrame by passing a dictionary of objects where the keys are the column
labels and  values are the column values
"""
df = pd.DataFrame({
    "A": 1.0,
    "B": pd.Timestamp("20240101"),
    "C": pd.Series(1, index=list(range(4)), dtype='float32'),
    "D": np.array([3] * 4, dtype='int32'),
    "E": pd.Categorical(['test', "train", "test", "train"]),
    "F": "foo"
}
)

# print(df)
"""
The columns of the resulting DataFrame have different dtypes
"""
# print(df.dtypes)

