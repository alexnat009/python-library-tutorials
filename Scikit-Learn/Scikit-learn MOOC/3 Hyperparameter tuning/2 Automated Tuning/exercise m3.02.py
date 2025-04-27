from unittest import TestLoader

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import plotly.express as px

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y *= 100
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = Pipeline([
	("preprocessor", StandardScaler()),
	("classifier", KNeighborsRegressor())
])

param_distribution = {
	"classifier__n_neighbors": np.logspace(0, 3, num=10).astype(np.int32),
	"preprocessor__with_mean": (True, False),
	"preprocessor__with_std": (True, False)
}

model_random_search = RandomizedSearchCV(
	model,
	param_distribution,
	n_iter=20,
	scoring="neg_mean_absolute_error",
	n_jobs=2,
	verbose=1,
	random_state=1
)
model_random_search.fit(X_train, y_train)
"""
The scoring function is expected to return higher values for better models, since 
grid/random search objects maximize it. Because of that, error metrics like 'mean_absolute_error'
must be negated to work correctly
"""
print(model_random_search.best_params_)

cv_results = pd.DataFrame(model_random_search.cv_results_)
cv_results["mean_test_score"] *= -1
print(cv_results.columns)
column_name_mapping = {
	"param_classifier__n_neighbors": "n_neighbors",
	"param_preprocessor__with_mean": "centering",
	"param_preprocessor__with_std": "scaling",
	"mean_test_score": "mean test score",
}
cv_results = cv_results.rename(columns=column_name_mapping)
cv_results = cv_results[column_name_mapping.values()].sort_values(
	"mean test score"
)
column_scaler = ["centering", "scaling"]
cv_results[column_scaler] = cv_results[column_scaler].astype(np.int64)
cv_results["n_neighbors"] = cv_results["n_neighbors"].astype(np.int64)

fig = px.parallel_coordinates(
	cv_results,
	color="mean test score",
	dimensions=["n_neighbors", "centering", "scaling", "mean test score"],
	color_continuous_scale=px.colors.diverging.Tealrose
)
fig.show()


"""
Selecting the best performing models (below MEA score of ~47k$), we observe that:
	○ scaling the data is important. All the best performing models use scaled featured
	○ centering the data doesn't have a strong impact. Both approaches, centering or not, can lead to good models
	○ using some neighbors is fine but using too many is a problem. In particular no pipeline with 
	n_neighbors=1 can be found among the best models. However, scaling features has an even stronger impact
	than the choice of n_neighbors in this problem 
"""
