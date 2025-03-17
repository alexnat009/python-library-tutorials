import pandas as pd
import numpy as np

"""
CONCAT

pandas provide various facilities fro easily combining together Series and DataFrame
objects with various kinds of set logic for the indexes and relational algebra functionality
in the case of join/merge-type operations
"""

"""
Concatenations pandas objects together row-wise with concat()
"""

df = pd.DataFrame(np.random.randn(10, 4))

pieces = [df[:3], df[3:7], df[7:]]
# print(pd.concat(pieces))


"""
JOIN

merge() enables SQL style join types along specified columns
"""

left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})

# print(left, right, sep="\n")

# print(pd.merge(left, right, on="key"))

left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
# print(left, right, sep="\n")

# print(pd.merge(left, right, on="key"))
