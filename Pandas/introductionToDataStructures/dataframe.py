import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt


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

"""
From a dict of tuples
You can automatically create a MultiIndexed frame by passing a tuples dictionary.
"""
data = {
	("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
	("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
	("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
	("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
	("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
}
print(pd.DataFrame(data))
printLine()

"""
From a series
The result will be a DataFrame with the same index as input Series, and with one column
whose name is the original name of the Series
"""

ser = pd.Series(range(3), index=list("abc"), name="ser")
print(pd.DataFrame(ser))
printLine()

"""
From a list of namedtuples
The field names of the first namedtuple in the list determine the columns of the DataFrame
The remaining namedtuples are simply unpacked and their values are fed into the rows of the DataFrame
If ant of those tuples is shorter than the first namedtuple then the later columns in the corresponding row
are marked as missing values. If any are longer than the first namedtuple, a ValueError is raised 
"""

Point2d = namedtuple("Point2D", "x y")
print(pd.DataFrame([Point2d(0, 0), Point2d(0, 3), Point2d(2, 3)]))

Point3d = namedtuple("Point3D", "x y z")
print(pd.DataFrame([Point3d(0, 0, 0), Point3d(0, 3, 5), Point2d(2, 3)]))
printLine()

# Alternate constructors

"""
DataFrame.from_dict

DataFrame.from_dict() takes a dict of dicts or a dict of array-like sequences and returns
a DataFrame. It operates like the DataFrame constructor except for the orient parameter which is 
'columns' by default, but which can be set to 'index' in order to use the dict keys as row labels
"""

print(pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])])))
print(
	pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]), orient='index', columns=["one", "two", "three"]))
printLine()
"""
DataFrame.from_records

DataFrame.from_records() takes a list of tuples or an ndarray with structured dtype. It works
analogously to the normal DataFrame constructor, except that the resulting DataFrame index
may be a specific field of the structured dtype
"""

data = np.array([(3, 'a'), (3, 'b'), (1, 'c'), (0, 'd')], dtype=[('col_1', 'i4'), ('col_2', 'U1')])
print(pd.DataFrame.from_records(data))
printLine()

# Column selection, addition, deletion

"""
You can treat a DataFrame semantically like a dict of like-indexed Series objects. 
Getting, setting, and deleting columns works with the same syntax as the analogous dict operations:
"""

print(df["one"])
df["three"] = df["one"] * df["two"]
df["flag"] = df["one"] > 2
print(df)

"""
Columns can be deleted or popped like with a dict
"""

del df["two"]
three = df.pop("three")
print(df)
print(three)

"""
When inserting a scalar value, it will naturally be propagated to fill the column
"""

df["foo"] = "bar"
print(df)

"""
When inserting a Series that does not have the same index as the DataFrame, it will
be conformed to the DataFrame's index

You can insert raw ndarrays but their length must match the length of the DataFrame's index
"""

df["one_trunc"] = df["one"][:1]
print(df)

"""
By Default, columns get inserted at the end. DataFrame.insert() inserts at a particular
location in the columns
"""

df.insert(1, "bar", df["one"])
print(df)
printLine()
# Assigning new columns in method chains

"""
Inspired by dplry's mutate verb, DataFrame has as assign() method that allows you to easily create
new columns that are potentially derived from existing columns

assign() always return a acopy of the data, leaving the original DataFrame untouched
"""

iris = pd.read_csv('../datasets/iris.csv')
print(iris.head())

print(iris.assign(sepal_ratio=iris["sepal_width"] / iris["sepal_length"]).head())

print(iris.assign(sepal_ratio=lambda x: x["sepal_width"] / x["sepal_length"]).head())
"""
Passing a callable, as opposed to an actual value to be inserted, is useful when you don't
have a reference to the DataFrame at hand. This is common when using assign() in a chain of operations.
For example, we can limit the DataFrame to just those observations with a Sepal Length greater than 5, calculate
the ratio, and plot
"""

# ((iris.query("sepal_length > 5")
#   .assign(sepal_ratio=lambda x: x["sepal_width"] / x["sepal_length"],
# 		  petal_ratio=lambda x: x["petal_width"] / x["petal_length"]))
#  .plot(kind="scatter", x="sepal_ratio", y="petal_ratio"))
# plt.show()

"""
Since a function is passed in, the function is computed on the DataFrame being assigned to.
Importantly, this is the DataFrame that's been filtered to those rows with sepal length greater than 5.
The filtering happens first, and then the ratio calculations.
"""

dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
dfa = dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"])
print(dfa)
printLine()

# Indexing / Selection

"""
The basics of indexing are as follows:

Operation				Syntax				Result
---------------------------------------------------
Select columns			df[col]				Series
Select row by label		df.loc[label]		Series
Select row by			df.iloc[loc]		Series 
integer location
Slice rows				df[5:10]			DataFrame
Select rows by
boolean vector			df[bool_vec]		DataFrame

"""

print(df.loc['b'])
print(df.iloc[2])
printLine()

# Data alignment and arithmetic

"""
Data alignment between DataFrame objects automatically align on both the columns and the index.
Again, the resulting objects will have the union of the column and row labels
"""

df = pd.DataFrame(np.random.randn(10, 4), columns=list("ABCD"))
df2 = pd.DataFrame(np.random.randn(7, 3), columns=list("ABC"))

print(df + df2)

"""
When doing an operation between DataFrame and Series, the default behavior is to align
the Series index on the DataFrame columns, thus broadcasting row-wise.
"""
print(df - df.iloc[0])

"""
Arithmetic operations with scalars operate element-wise
"""
print(df * 5 + 2)
print(1 / df)
print(df ** 4)

"""
Boolean operators operate element-wise as well
"""
df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print(df1 & df2)
print(df1 | df2)
print(df1 ^ df2)
print(-df1)
printLine()
# Transposing
"""
To transpose, access the T attribute or DataFrame.transpose(), similar to an ndarray
"""
print(df[:5].T)
printLine()

# DataFrame interoperability with Numpy Functions
"""
Most NumPy functions can be called directly on Series and DataFrame
"""
print(np.exp(df))
print(np.asarray(df))
ser = pd.Series([1, 2, 3, 4, 5])
print(np.exp(ser))

"""
When multiple Series are passed to a ufunc, then are aligned before performing the operation 
"""

ser1 = pd.Series([1, 2, 3], index=list("abc"))
ser2 = pd.Series([1, 3, 5], index=list("bac"))
print(ser1, ser2, np.remainder(ser1, ser2), sep="\n")

"""
As usual, the union of the two indices is taken, and non-overlapping values are filled with missing values
"""

ser3 = pd.Series([2, 4, 6], index=list("bcd"))
print(ser3, np.remainder(ser1, ser3), sep="\n")

"""
When a binary ufunc is applied to a Series and Index, the Series implementation takes precedence
ands a Series is returns
"""
ser = pd.Series([1, 2, 3])
idx = pd.Index([4, 5, 6])

print(np.maximum(ser, idx))
printLine()
"""
If possible, the ufunc is applied without converting the underlying data to an ndarray
"""

# Console display
"""
A very large DataFrame will be truncated to display them in the console. You can also get a
summary using info().
"""

print(iris)
print(iris.info())

"""
However, using DataFrame.to_string() will return a string representation of the DataFrame
in tabular form, though it won;t always fit the console width
"""
print(iris[-20:].to_string())

"""
Wide DataFrames will be printed across multiple rows by default
"""
print(pd.DataFrame(np.random.randn(3, 12)))
#

"""
You can change how much to print on a single row by setting the display.width, display.max_colwidth or expand_frame_repr

"""

# pd.set_option("expand_frame_repr", False)  # default is 80
# print(pd.DataFrame(np.random.randn(3, 12)))
