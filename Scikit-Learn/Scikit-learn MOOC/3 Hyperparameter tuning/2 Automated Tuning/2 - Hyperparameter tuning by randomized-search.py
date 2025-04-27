from pprint import pprint

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, RandomizedSearchCV
from scipy.stats import loguniform

df = pd.read_csv("../../datasets/adult-census.csv")

target_name = "class"
X = df.drop(columns=[target_name, "education-num"])
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

categorical_columns_preprocessor = selector(dtype_include=object)

categorical_columns = categorical_columns_preprocessor(X)
categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer(
	[
		("categorical_preprocessor", categorical_preprocessor, categorical_columns)
	], remainder="passthrough"
)

model = Pipeline([
	("preprocessor", preprocessor),
	("classifier", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))
])

"""
With the GridSearchCV estimator, the parameters need to be specified explicitly.
Exploring large number of values for different parameters quickly becomes intractable 

Instead, we can randomly generate the parameter candidates. Such approach avoids the
regularity of the grid. Hence, adding more evaluations can increase the resolution in
each direction. This is the case frequent situation where the choice of some hyperparameters
is not very important

With a grid the danger is that the region of good hyperparameters may fall between lines of the
grid. Rather, stochastic search samples the hyperparameter 1 independently from the second
parameter and finds the optimal region
"""

"""
The RandomizedSearchCV class allows for such stochastic search. It is used similarly to
GridSearchCV but the sampling distribution need to be specified instead of the parameter values.
We can draw candidates using a log-uniform distribution because the parameters we are interested
in take positive values with a natural log scaling 
"""

# NOTE::
"""
Random search is typically beneficial compared to grid search to optimize 3
or more hyperparameters
"""


class LoguniformInt:
	def __init__(self, a, b):
		self._distribution = loguniform(a, b)

	def rvs(self, *args, **kwargs):
		return self._distribution.rvs(*args, **kwargs).astype(int)


param_distributions = {
	"classifier__l2_regularization": loguniform(1e-6, 1e3),
	"classifier__learning_rate": loguniform(0.001, 10),
	"classifier__max_leaf_nodes": LoguniformInt(2, 256),
	"classifier__min_samples_leaf": LoguniformInt(1, 100),
	"classifier__max_bins": LoguniformInt(2, 255),
}
model_random_search = RandomizedSearchCV(
	model,
	param_distributions=param_distributions,
	n_iter=10,
	cv=5,
	verbose=2,
	n_jobs=3
)
model_random_search.fit(X_train, y_train)

accuracy = model_random_search.score(X_test, y_test)

print("The best parameters are:")
pprint(model_random_search.best_params_)

column_results = [f"param_{name}" for name in param_distributions.keys()]
column_results += ["mean_test_score", "std_test_score", "rank_test_score"]

cv_results = pd.DataFrame(model_random_search.cv_results_)


def shorten_param(param_name):
	if "__" in param_name:
		return param_name.rsplit("__", 1)[1]
	return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
print(cv_results)

cv_results = pd.read_csv("../../figures/randomized_search_results.csv")

cv_results = cv_results[column_results].rename(shorten_param, axis=1)
print(cv_results.sort_values("mean_test_score", ascending=False))
