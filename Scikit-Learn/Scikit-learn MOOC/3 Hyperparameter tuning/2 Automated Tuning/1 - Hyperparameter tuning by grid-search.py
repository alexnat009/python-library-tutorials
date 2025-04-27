import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, ShuffleSplit
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../../datasets/adult-census.csv")

target_name = "class"
y = df[target_name]
X = df.drop(columns=[target_name, "education-num"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(X)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer([
	("categorical_preprocessor", categorical_preprocessor, categorical_columns)
], remainder="passthrough")

model = Pipeline([
	("preprocessor", preprocessor),
	("classifier", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))
])

param_grid = {
	"classifier__learning_rate": (0.01, 0.1, 1, 10),
	"classifier__max_leaf_nodes": (3, 10, 30)
}
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
model_grid_search.fit(X_train, y_train)

accuracy = model_grid_search.score(X_test, y_test)
print(
	f"The test accuracy score of the grid-searched pipeline is: {accuracy:.2f}"
)
cv = ShuffleSplit(n_splits=2, test_size=0.2)
cv_result = cross_validate(model_grid_search, X, y, cv=cv, return_train_score=True)
scores = cv_result["train_score"]
print(
	f"The test accuracy score of the grid-searched pipeline is: {scores.mean():.2f}±{scores.std():.2f}"
)

print(model_grid_search.predict(X_test.iloc[0:5]))
print(f"The best set of parameters is: {model_grid_search.best_params_}")

"""
In addition to best parameters we can inspect all results which are stored
in the attribute 'cv_results_' of the grid-search
"""
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values("mean_test_score", ascending=False)
print(cv_results)

columns_results = [f"param_{name}" for name in param_grid.keys()]
columns_results += ["mean_test_score", "std_test_score", "rank_test_score"]

cv_results = cv_results[columns_results]


def shorten_param(param_name):
	if "__" in param_name:
		return param_name.rsplit("__", 1)[1]
	return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
print(cv_results)

pivoted_cv_results = cv_results.pivot_table(
	values="mean_test_score",
	index=["learning_rate"],
	columns=["max_leaf_nodes"]
)
print(pivoted_cv_results)

ax = sns.heatmap(
	pivoted_cv_results,
	annot=True,
	cmap="YlGnBu",
	vmin=0.7,
	vmax=0.9,
	cbar_kws={"label": "mean test accuracy"},
)
ax.invert_yaxis()
plt.show()
