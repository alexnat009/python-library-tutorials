import numpy as np
import pandas as pd


def printLine():
	print("----------------------------------------")


"""
DataFrame is a 2D labeled data structure with  columns of potentially different types.
"""

# Creating a DataFrame
"""
You can create a DataFrame from:
Dict of Series or dicts - The resulting index will be the union of the indexes of the various Series
"""
d = {
	"one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
	"two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}
df = pd.DataFrame(d)
print(df)
print(pd.DataFrame(d, index=["d", "b", "a"]))
printLine()
"""
Dict of ndarrays/lists - All ndarrays must share the same length, if the index is passed, 
                         it must be the same length as the arrays.  
"""

d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
print(pd.DataFrame(d))
print(pd.DataFrame(d, index=["a", "b", "c", "d"]))
printLine()
"""
From structured or record array
                            - This case is handled identically to a dict of arrays
"""
data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "O")])
data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]
print(pd.DataFrame(data))

print(pd.DataFrame(data, index=["first", "second"]))
print(pd.DataFrame(data, columns=["C", "A", "B"]))
printLine()
"""
From a list of dicts
"""
data = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
print(pd.DataFrame(data))
print(pd.DataFrame(data, index=["first", "second"]))
print(pd.DataFrame(data, columns=["a", "b"]))
printLine()
