import pandas as pd
import numpy as np

np.random.seed(0)
"""
Series is a one-dimensional labeled array capable of holding any data type
The axis labels are collectively referred to as the index
s = pd.Series(data=None,index=None)
"""

"""
You can create Series from:
ndarray
dict    (if index isn't specified the keys of the dict will be used as index)
scalar value or constant
"""
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
"""
Series act similarly to a ndarray, but operations such as slicing will also slice index
"""

print(s.iloc[0])
print(s.iloc[:3])
print(s[s > s.median()])
print(s.iloc[[4, 1, 3]])
print(np.exp(s))
# Line a NumPy array, a pandas Series has a one, concrete dtype

print(s.dtype)
print(s.array)  # to get the data as an array backing the Series, resulting in a NumpyExtensionArray
print(s.to_numpy())  # to get the data as a NumPy array

# Series is dict-like
"""
Seires is like a fixed-size dict in which you can get and set values by index label
"""
print(s['a'])
s["e"] = 12.0
print(s)

print("e" in s)
print("f" in s)
# print(s["f"]) # KeyError
# You can use get() method to avoid KeyError and return a default value
print(s.get("f", np.nan))

# Vectorized operations and label alignment with Series
"""
You don't need to loop through each element to apply an operation, you can apply it to the whole Series
"""
print(s + s)
print(s * 2)
"""
Main difference between Series and ndarray is the operations between Series automatically align the data based on label.
Thus, you can write computations without giving considerations to whether the Series involved have the same labels
"""
print(s.iloc[1:] + s.iloc[:-1])

"""
The result of an operation between unaligned Series will have the union of the indexes involved.
If a label is not found in one Series or the other, the result will be marked as missing NaN.
"""

# Series can also have a name attribute

s = pd.Series(np.random.randn(5), name="Random Numbers")
print(s)
s2 = s.rename("Random Numbers 2")
print(s2.name)