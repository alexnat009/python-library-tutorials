import pandas as pd
import numpy as np

"""
pandas has simple, powerful, and efficient functionality for performing resampling operations
during frequency conversion. This is extremely common in, but not limited to, financial applications
"""

rng = pd.date_range("1/1/2012", periods=100, freq="s")

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

print(ts.resample("5Min").sum())

rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")

ts = pd.Series(np.random.randn(len(rng)), rng)

print(ts)

ts_utc = ts.tz_localize("UTC")
print(ts_utc)

print(ts_utc.tz_convert("US/Eastern"))

print(rng)


print(rng + pd.offsets.BusinessDay(5))