import numpy as np
import pandas as pd

arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]

index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])

# print(index)

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])

df2 = df[:4]

# print(df)
# print(df2)

stacked = df2.stack(future_stack=True)
# print(stacked)
# print(stacked.index)

# print(stacked.unstack())
# print(stacked.unstack(0))
# print(stacked.unstack(1))

df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)

print(df)

"""
pivot_table() pivots a DataFrame specifying the values, index and columns
"""

pivotDf = pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])

print(pivotDf)