import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y *= 100  # rescale the target in k$

hgbr = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True, random_state=0)

param_grid = {
	"max_depth": [3, 8],
	"max_leaf_nodes": [15, 31],
	"learning_rate": [0.1, 1]
}
search_grid = GridSearchCV(
	estimator=hgbr,
	param_grid=param_grid,
	n_jobs=2,
)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_validate(search_grid, X, y, cv=cv, n_jobs=2, return_estimator=True)
score = cv_scores["test_score"]
print(f"R2 score with cross-validation: {score.mean()} ± {score.std()}")

for estimator in cv_scores["estimator"]:
	print(estimator.best_params_)
	print(f' # trees: {estimator.best_estimator_.n_iter_}')

index_columns = [f"param_{name}" for name in param_grid.keys()]
columns = index_columns + ["mean_test_score"]

inner_cv_results = []
for cv_idx, estimator in enumerate(cv_scores["estimator"]):
	search_cv_results = pd.DataFrame(estimator.cv_results_)
	search_cv_results = search_cv_results[columns].set_index(index_columns)
	search_cv_results = search_cv_results.rename(columns={"mean_test_score": f"CV {cv_idx}"})

	inner_cv_results.append(search_cv_results)

inner_cv_results = pd.concat(inner_cv_results, axis=1).T

color = {"whiskers": "black", "medians": "black", "caps": "black"}
inner_cv_results.plot.box(vert=False, color=color)
plt.xlabel("R2 score")
plt.ylabel("Parameter")
plt.title("Inner CV results with parameters\n(max_depth, max_leaf_nodes, learning_rate)")
plt.show()
