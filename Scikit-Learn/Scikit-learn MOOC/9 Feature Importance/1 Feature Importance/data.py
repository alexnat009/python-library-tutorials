import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X = X[:10000]
y = y[:10000]


"""
The features reads as follow:

	○ MedInc: 		median income in block
	○ HouseAge:		median house age in block
	○ AveRooms:		average number of rooms
	○ AveBedrms:	average number of bedrooms
	○ Population:	block population
	○ AveOccup:		average house occupancy
	○ Latitude:		house block latitude
	○ Longitude:	house block longitude
	○ MedHouseVal:	median house value in 100$ (target)	
"""

rng = np.random.RandomState(0)
bin_var = pd.Series(rng.randint(0, 1, X.shape[0]), name="rnd_bin")
num_var = pd.Series(np.arange(X.shape[0]), name="rnd_num")
X_with_rnd_feat = pd.concat([X, bin_var, num_var], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_with_rnd_feat, y, random_state=29)

train_dataset = X_train.copy()
train_dataset.insert(0, "MedHouseVal", y_train)
