from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split

df = pd.read_csv("../datasets/penguins.csv")
print(df.columns)
columns = ["Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"]
target_name = "Species"

df_non_missing = df[columns + [target_name]].dropna()

X = df_non_missing[columns]
y = df_non_missing[target_name]

# print(y.value_counts())
print(X.describe())
# print(y.unique())

model = Pipeline([
	("preprocessor", StandardScaler()),
	("classifier", KNeighborsClassifier())
])

pprint(model.get_params())
# cv = StratifiedKFold(n_splits=10)
# cv_results = cross_validate(model, X, y, cv=10, scoring="balanced_accuracy")
# print(cv_results["test_score"].mean())

param_grid = {
	"classifier__n_neighbors": [5, 51, 101],
	"preprocessor": [StandardScaler(), "passthrough"]
}

grid_search = GridSearchCV(
	model,
	param_grid=param_grid,
	cv=10,
	scoring="balanced_accuracy",
	return_train_score=False,
	refit=False
)

grid_search.fit(X, y)
cv_results = pd.DataFrame(grid_search.cv_results_)

print(cv_results[['param_preprocessor', 'param_classifier__n_neighbors', 'mean_test_score']])
all_preprocessors = [
	"passthrough",
	StandardScaler(),
	MinMaxScaler(),
	QuantileTransformer(n_quantiles=100),
	PowerTransformer(method="box-cox"),
]

param_grid = {
	"classifier__n_neighbors": [5, 51, 101],
	"preprocessor": all_preprocessors
}

grid_search = GridSearchCV(
	model,
	param_grid=param_grid,
	cv=10,
	scoring="balanced_accuracy",
	return_train_score=True

)

# grid_search.fit(X, y)
# cv_results = pd.DataFrame(grid_search.cv_results_)
# cv_results = cv_results[['param_preprocessor', 'param_classifier__n_neighbors', 'mean_test_score']]
# print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
grid_search.fit(X_train, y_train)
cv_results = cross_validate(
	grid_search, X, y, cv=5, n_jobs=2, return_estimator=True
)

cv_results = pd.DataFrame(cv_results)
cv_test_scores = cv_results["test_score"]
print(
	"Generalization score with hyperparameters tuning:\n"
	f"{cv_test_scores.mean():.3f} ± {cv_test_scores.std():.3f}"
)

for cv_fold, estimator_in_fold in enumerate(cv_results["estimator"]):
	print(
		f"Best hyperparameters for fold #{cv_fold + 1}:\n"
		f"{estimator_in_fold.best_params_}"
	)
