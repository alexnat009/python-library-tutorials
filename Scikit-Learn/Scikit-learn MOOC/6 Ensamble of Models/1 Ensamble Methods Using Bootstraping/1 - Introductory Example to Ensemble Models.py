import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
y *= 100  # rescale the target in k%

"""
first lets check the generalization performance of decision tree regressor with default parameters
"""

dtr = DecisionTreeRegressor(random_state=0)
cv_results = cross_validate(dtr, X, y, n_jobs=2)
scores = cv_results["test_score"]

print(
	"Base model:\n"
	"R2 score obtained by cross-validation: "
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)

"""
We obtain fair results. However, as we previously presented, this model need to be tuned to overcome
overfitting/underfitting. Indeed, the default parameters will not necessarily lead to an optimal decision tree.
Instead of using the default value, we should search via cross-validation the optimal value of the important parameters
such as max_depth, min_samples_split, min_samples_leaf

We recall that we need to tune these parameters, as decision trees tend to over fit the training data if we grow deep
tree, but there are no rule on what each parameters should be set to. Thus, not making a search could lead us to have 
an underfitted/overfitted model

Now, we make a grid-search to tune the hyperparameters that we mentioned earlier
"""

param_grid = {
	"max_depth": [5, 8, None],
	"min_samples_split": [2, 10, 30, 50],
	"min_samples_leaf": [0.01, 0.05, 0.1, 1]
}
cv = KFold(n_splits=3)

grid_search = GridSearchCV(
	DecisionTreeRegressor(random_state=0),
	param_grid=param_grid,
	cv=cv,
	n_jobs=2
)

cv_results = cross_validate(
	grid_search,
	X,
	y,
	n_jobs=2,
	return_estimator=True
)

scores = cv_results["test_score"]

print(
	"Grid search:\n"
	"R2 score obtained by cross-validation: "
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)

"""
We see that optimizing the hyperparameters will have a positive effect on the generalization
performance, However, it comes with a higher computational cost

We can create a dataframe storing the important information collected during the tuning of the parameters 
and investigate the results 
"""

"""
Lets visualize the parameters search ivs parallel coordinate figure
"""
# Combine cv_results_ from each fold's grid search into a single DataFrame
all_results = []
for estimator in cv_results["estimator"]:
	result = pd.DataFrame(estimator.cv_results_)
	all_results.append(result)

# results_df = pd.concat(all_results, ignore_index=True)
results_df = pd.concat(all_results, ignore_index=True)

# Select only the relevant columns for parallel coordinates
cols_to_plot = [
	'param_max_depth',
	'param_min_samples_split',
	'param_min_samples_leaf',
	'mean_test_score'
]

# Convert to appropriate numeric types if necessary
results_df['param_max_depth'] = results_df['param_max_depth'].apply(
	lambda x: -1 if x is None else int(x)
)
results_df['param_min_samples_split'] = results_df['param_min_samples_split'].astype(int)
results_df['param_min_samples_leaf'] = results_df['param_min_samples_leaf'].astype(float)
results_df['mean_test_score'] = results_df['mean_test_score'].astype(float)

# Create parallel coordinates plot
fig = px.parallel_coordinates(
	results_df[cols_to_plot],
	color='mean_test_score',
	labels={
		'mean_test_score': 'R² Score',
		'param_max_depth': 'Max Depth',
		'param_min_samples_split': 'Min Samples Split',
		'param_min_samples_leaf': 'Min Samples Leaf',
	},
	color_continuous_scale=px.colors.sequential.Viridis,
)

fig.update_layout(title="Parallel Coordinates: Decision Tree Hyperparameter Tuning")
fig.show()

"""
Now we will use an ensemble method called bagging.

Here we will use 20 decision trees and check the fitting time as well as the generalization performance
on the left-out testing data. It is important to note that we are not going to tune any
parameter of the decision tree
"""

estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
	estimator=estimator,
	n_estimators=20,
	random_state=0
)

cv_results = cross_validate(
	bagging_regressor,
	X,
	y,
	n_jobs=2
)

scores = cv_results["test_score"]
print(
	"Bagging:\n"
	"R2 score obtained by cross-validation: "
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)

"""
Without searching for optimal hyperparameters, the overall generalization performance of the bagging
regressor is better than a single decision tree. In addition, the computational cost is reduced
in comparison of seeking for the optimal hyperparameters.

This shows the motivation behind the use of an ensemble learner: It gices a relatively good baseline
with decent generalization performance without any parameter tuning,
"""
