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

