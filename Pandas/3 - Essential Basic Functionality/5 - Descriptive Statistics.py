import numpy as np
import pandas as pd

np.random.seed(0)
"""
There exists a large number of methods for computing descriptive statistics and other
related operations on Series/DataFrame. Most of these are aggregations like sum(), mean(), quantile()
but some of them, like cumsum() and cumprod() produce an object of the same size. These methods
take an axis argument, just like ndarray.{sum, std, ...}, but the axis can be specified by name or integer:
○ Series: no axis argument needed
○ DataFrame: "index" (axis=0,default), "columns" (axis=1)
"""
df = pd.DataFrame(np.random.randn(4, 3), columns=["one", "two", "three"], index=list("abcd"))
df.loc["a", "one"] = None
df.loc["d", "three"] = None

print(df)

print(df.mean(0))
print(df.mean(1))

"""
All such methods have a skipna option signaling whether to exclude missing data (True by default)
"""
print(df.sum(0, skipna=False))
print(df.sum(1, skipna=False))

"""
Combined with the broadcasting / arithmetic behavior, one can describe various statistical procedures, 
like standardization very concisely
"""

ts_stand = (df - df.mean()) / df.std()
print(ts_stand.std())

xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
print(xs_stand.std(1))

"""
Note that methods like cumsum() and cumprod() preserve the location of NaN values. This is somewhat different
from expanding() and rolling() since NaN behavior is furthermore dictated by a min_periods parameter"""

print(df.cumsum())

"""
Some of the common functions. Each also take an optional level parameter which applies only if the object has a 
hierarchical index
	Function			Description
	count				Number of non-NA observations
	sum					Sum of values
	mean				Mean of values
	median				Arithmetic median of values
	min					Minimum
	max					Maximum
	mode				Mode
	abs					Absolute value
	prod				Product of values
	std					Bessel-corrected sample standard deviation
	var					Unbiased variance
	sem				 	Standard error of the mean
	skew				Sample skewness (3rd moment)
	kurt 				Sample kurtosis (4th moment)
	quantile			Sample quantile (value at %)
	cumsum				Cumulative sum
	cumprod				Cumulative product
	cummax				Cumulative maximum
	cummin				Cumulative minimum
	
Note that by change some Numpy methods, like mean,std and sum, will exclude NAs on Series
input by default
"""
print(np.mean(df["one"]))
print(np.mean(df["one"].to_numpy()))

"""
Series.nunique() will return the number of unique non-NA values in a Series
"""

series = pd.Series(np.random.randn(500))
series[20:500] = np.nan
series[10:20] = 5
print(series.nunique())

# Summarizing data: Describe

"""
There is a convenient describe() function which computes a variety of summary statistics about
a Series or the columns of a DataFrame (excluding NAs)
"""

series = pd.Series(np.random.randn(1000))
series[::2] = np.nan
print(series.describe())

df = pd.DataFrame(np.random.randn(1000, 5), columns=list("abcde"))
df.iloc[::2] = np.nan
print(df.describe())

"""
You can select specific percentiles to include in the output
"""
print(series.describe(percentiles=[0.05, 0.25, 0.75, 0.95]))

"""
For a non-numerical Series object, describe() will give a simple summary of the number of 
unique values and most frequently occurring values
"""
series = pd.Series(["a", "a", "b", "b", "a", "a", np.nan, "c", "d", "a"])
print(series.describe())

"""
On a mixed-type DataFrame object, describe() will restrict the summary to include only numerical
columns or, if none are, only categorical columns
This behavior can be controlled by providing a list of types as include/exclude arguments
you can also pass all to include every value
"""

# Index of min/max values
"""
The idxmin() and idxmax() functions on Series and DataFrame compute the index labels with the minimum and 
maximum corresponding values, they return the first matching index in case of multiple min/max values
These are called argmin and argmax in NumPy
"""

series = pd.Series(np.random.randn(5))
print(series)

print(series.idxmax(), series.idxmin())
df = pd.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
print(df)
print(df.idxmin(axis=0))
print(df.idxmax(axis=1))

# Value counts (histogramming) / mode
"""
The value_counts() Series method computes a histogram of a 1D array of values. It can also be used as a function
on regular arrays
"""

data = np.random.randint(0, 7, 50)
print(data)
series = pd.Series(data)
print(series.value_counts())

"""
The value_counts() method can be used to count combinations across multiple columns. By default all columns are used
but a subset can be selected using the subset argument
"""

data = {"a": [1, 1, 3, 4], "b": ["x", "x", "y", "y"]}

df = pd.DataFrame(data)
print(df.value_counts())

"""
Similarly you can get the most frequently occurring value(s) i.e. the mode, of the values in a Series/DataFrame
"""

series = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])
print(series.mode())

df = pd.DataFrame(
	{
		"A": np.random.randint(0, 7, size=50),
		"B": np.random.randint(-10, 15, size=50),
	}
)
print(df.mode())

# Discretization and quantiling
"""
Continuous values can be discretized using the cut() (bins based on values) and qcut() (bins based on sample qunatiles)
"""

arr = np.random.randint(-10, 10, 10)
print(arr)
factor = pd.cut(arr, labels=["low", "medium", "high"], bins=[-10, -5, 5, 10], include_lowest=True)
print(factor)

"""
qcut() computes sample quantiles. For examples, we could slice up some normally distributed data into
equal-size quartiles
"""
factor = pd.qcut(arr, labels=["low", "medium", "high"], q=[0, 0.25, 0.75, 1])
print(factor)
