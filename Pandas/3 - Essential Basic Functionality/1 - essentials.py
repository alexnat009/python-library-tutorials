import pandas as pd
import numpy as np

np.random.seed(0)

index = pd.date_range('1/1/2000', periods=8)

s = pd.Series(np.random.randn(5), index=list("abcde"))
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list("ABC"))
print(df)
df.head()
"""
Head and tail
df.head(n=5)
df.tail(n=5)
"""

# Attributes and underlying data
"""
pandas objects have a number of attributes enabling you to access the metadata

shape: gives the axis dimensions of the object, consistent with ndarray
Axis labels:
	Series: index (only axis)
	DataFrame: index (rows) and columns

These attributes can be safely assigned to
"""

print(df[:2])

df.columns = [x.lower() for x in df.columns]
print(df.head(1))

"""
To get the actual data inside a Index or Series, use the .array property
"""

print(s.array)
print(s.index.array)

"""
If you know you need a NumPy array, use to_numpy() or numpy.asarray()
"""

print(s.to_numpy())
print(np.asarray(s))

"""
When series or Index is backed by an ExtensionArray, to_numpy() may involve copying
data and coercing values
"""

"""
In the past, Pandas recommended Series.values or DataFrame.values for extracting data from 
a Series or DataFrame. You'll still find references to these in old code bases and
online. Going forward, it's recommend to avoid .values and use to_numpy() as .values has drawbacks: 
"""

# Accelerated operatins
"""
Pandas has support for accelerating certain types of binary numerical and boolean operations using
the 'numexpr' and 'bottleneck' libraries
These libraries are especially useful when dealing with large datasets, and provide large speedups.

'numexpr' uses smart chunking, caching, and multiple cores
'bottleneck' is a set of specialized cython routines that are especially fast when dealing with arrays
that have nans 
"""

# Flexible binary operations
"""
With binary operations between pandas data structures, there are two key points of interest:
1) Broadcasting behavior between higher and lower dimensional objects
2) Missing data in computations
"""

# Matching/ broadcasting behavior
"""
DataFrame has the methods 'add()', 'sub()', 'mul()', 'div()' and related functions 'radd()',
'rsub(), for carrying out binary operations. For broadcasting behavior, Series input is of primary
interest. Using these functions, you can use to either match on the index or columns via the axis keyword
"""

df = pd.DataFrame(
	{
		"one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
		"two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
		"three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
	}
)

print(df)

row = df.iloc[1]
column = df["two"]

print(row, column, sep="\n")

print(df.sub(row, axis="columns"))  # or axis=1
print(df.sub(column, axis="index"))  # or axis=0

"""
Furthermore you can align a level of a Multi-indexed DataFrame with a series
"""
dfmi = df.copy()
dfmi.index = pd.MultiIndex.from_tuples(
	[(1, "a"), (1, "b"), (1, "c"), (2, "a")], names=["first", "second"]
)

print(dfmi)
print(dfmi.sub(column, axis=1, level="second"))

"""
Series and Index also support the 'divmod()' builtin. This function takes the floor
division and modulo operation at the same time returning a two-tuple of the same type
as the left hand side
"""

s = pd.Series(np.arange(10))
div, rem = divmod(s, 3)
# we can also do elementwise 'divmode()' -> div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

print(div, rem, sep="\n")

# Missing data / operations with fill values
"""
In Series and DataFrame, the arithmetic functions have the option of inputting a fill_value
namely a value to substitute when at most one of the values at a location are missing
For example you may wish to treat NaN as 0 unless both DataFrames are missing that value, in which case
the result will be NaN
"""
df2 = df.copy()

df2.loc["a", "three"] = 1.0

print(df, df2, sep="\n")

print("+ operation", df + df2, sep='\n')
print("pd.add()", df.add(df2, fill_value=0), sep='\n')

# Flexible comparisons
"""
Series and DataFrames hae the binary comparison methods 'eq, 'ne', lt', 'ht', 'le' and 'ge,
whose behavior is analogous to the binary arithmetic operations described above
"""

print(df.gt(df2))
print(df.ne(df2))

"""
These operations produce a pandas object of the same type as the left-hand-side input that
is of dtype boot. These boolean objects can be used in indexing operations
"""

# Boolean reductions
"""
You can apply the reductions: 'empty', 'any()', 'all()' and 'bool()' to provide a way to 
summarize a boolean result
"""
print(np.all(df > 0, axis=0))
print(np.any(df > 0, axis=0))

print(df.empty)
"""
Asserting the truthiness of a pandas object will raise an error, as the testing of
the emptiness or values is ambiguous
"""

# Comparing if objects are equivalent
print(np.all(df + df == df * 2))
"""
Notice that the boolean DataFrame 'df + df == df * 2' contains some False values.
This is because NaNs do not compare as equals
np.nan == np.nan -> False 

So, NDFrames have equals() method for testing equality, with NaNs in corresponding locations treated as equals
"""
print((df + df).equals(df * 2))
"""
Note that the Series or DataFrame index needs to be in the same order ofr equality to be True
"""

df1 = pd.DataFrame({"col": ["foo", 0, np.nan]})

df2 = pd.DataFrame({"col": [np.nan, 0, "foo"]}, index=[2, 1, 0])

print(df1.equals(df2))
print(df1.equals(df2.sort_index()))

# Comparing array-like objects
"""
You can conveniently perform element-wise comparisons when comparing a pandas data
structure with a scalar value
"""
print((pd.Series(["foo", "bar", "baz"]) == "foo"))
print((pd.Index(["foo", "bar", "baz"]) == "foo"))

"""
Pandas also handles element-wise comparisons between different array-like objects of the same length
"""

print((pd.Series(["foo", "bar", "baz"]) == pd.Index(["foo", "bar", "qux"])))
print((pd.Series(["foo", "bar", "baz"]) == np.array(["foo", "bar", "qux"])))
"""
Trying to compare Index or Series objects of different lengths will raise a ValueError:
"""

# Combining overlapping data sets
df1 = pd.DataFrame(
	{"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)

df2 = pd.DataFrame(
	{
		"A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
		"B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
	}
)

print(df1, df2, sep="\n")
print(df1.combine_first(df2))

# General DataFrame Combine
"""
The combine_first() method above calls the more general DataFrame.combine().
This method takes another DataFrame and a combiner function, aligns the input DataFrame and then
passes the combiner function pairs of Series

to reproduce combine_first() as above
"""


def combiner(x, y):
	return np.where(pd.isna(x), y, x)


print(df1.combine(df2, combiner))

