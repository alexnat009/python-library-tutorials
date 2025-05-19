import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

print(df)

# print(df.groupby("A").groups)
# print(df.groupby("A").get_group("bar"))
# print(df.groupby("A")[["C", "D"]].sum())

print(df.groupby(["A", "B"]).sum())
