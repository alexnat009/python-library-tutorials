import numpy as np
import pandas as pd

dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

"""
Use DataFrame.head() and DataFrame.tail() to view the top and bottom rows of the frame respectively
Default value of rows  n = 5 for head and tail
"""
# print(df.head())
# print(df.tail(3))

"""
Display the DataFrame.index nad DataFrame.columns
"""
# print(df.index)
# print(df.columns)

"""
Return a NumPy representation of the underlying data with DataFrame.to_numpy() 
without the index or columns labels
"""
# print(df.to_numpy())

"""
Describe shows a quick statistic summary of your data
"""
# print(df.describe())

"""
Transposing your data
"""

# print(df.T)

"""
DataFrame.sort_index() sorts by an axis and DataFrame.sort_values() sorts by values:
"""

# print(df.sort_index(axis=1, ascending=False))
# print(df.sort_values(axis=1, by=2, key=lambda x: x.astype(str)))

