from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_validate

df = pd.read_csv("../../datasets/house_prices.csv")
X = df.drop(columns="SalePrice")
y = df["SalePrice"]
X = X.select_dtypes(np.number)
y /= 1000

lr = LinearRegression()

cv = KFold(n_splits=10)

scores = cross_val_score(
	lr,
	X,
	y,
	cv=cv,
	scoring="r2"
)
print(f"Precision score: {scores.mean():.3f} ± {scores.std():.3f}")
scores = cross_val_score(
	lr,
	X,
	y,
	cv=cv,
	scoring="neg_mean_absolute_error"
)
scores = -scores
print(f"Precision score: {scores.mean():.3f} ± {scores.std():.3f}")

scorings = ["r2", "neg_mean_absolute_error"]
cv_results = cross_validate(
	lr,
	X,
	y,
	cv=cv,
	scoring=scorings
)
scores = {
	"R2": cv_results["test_r2"],
	"MAE": -cv_results["test_neg_mean_absolute_error"],
}
scores = pd.DataFrame(scores)
print(scores)

scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]
loss_functions = ["squared_error", "absolute_error"]
scores = defaultdict(list)

for loss_function in loss_functions:
	model = HistGradientBoostingRegressor(loss=loss_function)
	cv_results = cross_validate(model, X, y, scoring=scoring)
	mse = -cv_results["test_neg_mean_squared_error"]
	mae = -cv_results["test_neg_mean_absolute_error"]
	scores["loss"].append(loss_functions)
	scores["MSE"].append(f"{mse.mean():.1f} ± {mse.std():.1f}")
	scores["MAE"].append(f"{mae.mean():.1f} ± {mae.std():.1f}")

scores = pd.DataFrame(scores)
scores.set_index("loss", inplace=True)
print(scores)
