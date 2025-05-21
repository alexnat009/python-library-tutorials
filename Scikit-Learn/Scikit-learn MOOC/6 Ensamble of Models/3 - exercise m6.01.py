import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
y *= 100  # rescale the target in k$
X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=0, test_size=0.5
)

bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(), n_jobs=2)
bagging_regressor.fit(X_train, y_train)
y_pred = bagging_regressor.predict(X_test)
scores = mean_absolute_error(y_test, y_pred)
print(
	"Basic mean absolute error of the bagging regressor:\n"
	f"{mean_absolute_error(y_test, y_pred):.2f} k$"
)

#
param_grid = {
	"estimator__max_depth": randint(3, 10),
	"n_estimators": randint(10, 30),
	"max_samples": [0.5, 0.8, 1.0],
	"max_features": [0.5, 0.8, 1.0],
}
gridSearch = RandomizedSearchCV(
	estimator=bagging_regressor,
	param_distributions=param_grid,
	scoring="neg_mean_absolute_error",
	n_iter=20
)

gridSearch.fit(X_train, y_train)
columns = [f"param_{name}" for name in param_grid.keys()] + ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(gridSearch.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = -cv_results["std_test_score"]
cv_results[columns].sort_values(by="mean_test_error")
print(cv_results)

y_pred = gridSearch.predict(X_test)
print(
	"Mean absolute error after tuning of the bagging regressor:\n"
	f"{mean_absolute_error(y_test, y_pred):.2f} k$"
)
