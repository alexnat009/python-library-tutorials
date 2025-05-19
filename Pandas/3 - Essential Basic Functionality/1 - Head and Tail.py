import numpy as np
import pandas as pd

# Head and Tail
np.random.seed(0)

index = pd.date_range('1/1/2000', periods=6)

s = pd.Series(np.random.randn(5), index=list("abcde"))
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list("ABC"))
print(df)
df.head()
"""
Head and tail
df.head(n=5)
df.tail(n=5)
"""
