import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

data_clf_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_clf_column = "Species"
data_clf = pd.read_csv("../../datasets/penguins_classification.csv")

data_reg_columns = ["Flipper Length (mm)"]
target_reg_column = "Body Mass (g)"
data_reg = pd.read_csv("../../datasets/penguins_regression.csv")


def fit_and_plot_classification(model, data, feature_names, target_names):
	model.fit(data[feature_names], data[target_names])
	if data[target_names].nunique() == 2:
		palette = ["tab:red", "tab:blue"]

	else:
		palette = ["tab:red", "tab:blue", "black"]

	DecisionBoundaryDisplay.from_estimator(
		model,
		data[feature_names],
		response_method="predict",
		cmap="RdBu",
		alpha=0.5
	)
	sns.scatterplot(
		data=data,
		x=feature_names[0],
		y=feature_names[1],
		hue=target_names,
		palette=palette
	)
	plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def fit_and_plot_regression(model, data, feature_names, target_names):
	model.fit(data[feature_names], data[target_names])
	data_test = pd.DataFrame(
		np.arange(data.iloc[:, 0].min(), data.iloc[:, 0].max()),
		columns=feature_names
	)
	target_predicted = model.predict(data_test)
	sns.scatterplot(
		x=data.iloc[:, 0], y=data[target_names], color="black", alpha=0.5
	)
	plt.plot(data_test.iloc[:, 0], target_predicted, linewidth=4)


max_depth = 30
dtc = DecisionTreeClassifier(max_depth=max_depth)
dtr = DecisionTreeRegressor(max_depth=max_depth)
fit_and_plot_classification(dtc, data_clf, data_clf_columns, target_clf_column)
plt.title(f"Shallow classification tree with max-depth of {max_depth}")
plt.subplots_adjust(right=0.75)
plt.show()

fit_and_plot_regression(dtr, data_reg, data_reg_columns, target_reg_column)
plt.title(f"Shallow regression tree with max-depth of {max_depth}")
plt.subplots_adjust(right=0.75)
plt.show()

"""
For both classification and regression setting, we observe that increasing the depth makes
the tree model more expressive. However, a tree that is too deep may overfit the training data,
creating partitions which are only correct for "outliers". The max_depth is one of the hyperparameters 
the one should optimize via cross-validation and grid_search
"""

param_grid = {"max_depth": np.arange(2, 10, 1)}
dtc_optimal = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
dtr_optimal = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid)

fit_and_plot_classification(dtc_optimal, data_clf, data_clf_columns, target_clf_column)
plt.title(f"Optimal depth found via CV: {dtc_optimal.best_params_['max_depth']}")
plt.subplots_adjust(right=0.75)
plt.show()

fit_and_plot_regression(dtr_optimal, data_reg, data_reg_columns, target_reg_column)
plt.title(f"Optimal depth found via CV: {dtr_optimal.best_params_['max_depth']}")
plt.subplots_adjust(right=0.75)
plt.show()

"""
Other hyperparameters in decision trees

The max_depth hyperparameter controls the overall complexity of the tree. This parameter is adequate
under the assumption that a tree is built symmetrically. However, there is no reason why a tree should be symmetrical.
Indeed, optimal generalization performance could be reached by growing some of the branches deeper than some others

We build a dataset where we illustrate this asymmetry. We generate a dataset composed of 2 subsets:
1) clear separation should be found by the tree
2) samples from both classes are mixed.
It implies that a decision tree need more splits to classify properly samples from the second subset that from the first 
one 
"""

data_clf_columns = ["Feature #0", "Feature #1"]
target_clf_column = "Class"
X_1, y_1 = make_blobs(n_samples=300, centers=[[0, 0], [-1, -1]], random_state=0)
X_2, y_2 = make_blobs(n_samples=300, centers=[[3, 6], [7, 0]], random_state=0)

X = np.concatenate([X_1, X_2], axis=0)
y = np.concatenate([y_1, y_2])

data_clf = np.concatenate([X, y[:, np.newaxis]], axis=1)
data_clf = pd.DataFrame(data_clf, columns=data_clf_columns + [target_clf_column])
data_clf[target_clf_column] = data_clf[target_clf_column].astype(np.int32)

sns.scatterplot(
	data=data_clf,
	x=data_clf_columns[0],
	y=data_clf_columns[1],
	hue=target_clf_column,
	palette=["tab:red", "tab:blue"],
)
plt.title("Synthetic dataset")
plt.show()

max_depth = 2
dtc = DecisionTreeClassifier(max_depth=max_depth)
fit_and_plot_classification(dtc, data_clf, data_clf_columns, target_clf_column)
plt.title(f"Decision tree with max-depth of {max_depth}")
plt.subplots_adjust(right=0.75)

"""
As expected we see that the blue blob in the lower right and the red blob on the top
are easily seperated. However, more splits are needed to better split the blob were both blue
and red data points are mixed
"""

_, ax = plt.subplots(figsize=(10, 10))
plot_tree(
	dtc,
	ax=ax,
	feature_names=data_clf_columns
)
plt.show()

max_depth = 6
dtc = DecisionTreeClassifier(max_depth=max_depth)
fit_and_plot_classification(
	dtc, data_clf, data_clf_columns, target_clf_column
)
plt.subplots_adjust(right=0.75)
plt.title(f"Decision tree with max-depth of {max_depth}")
_, ax = plt.subplots(figsize=(10, 10))
plot_tree(
	dtc,
	ax=ax,
	feature_names=data_clf_columns
)
plt.show()
"""
As expected, the left branch of the tree continue to grow while no further splits were done on
the right branch. Fixing the max_depth parameter would cut the tree horizontally at a specific level,
whether or not it would be more beneficial that a branch continue growing.

The hyperparameters 'min_samples_leaf', 'min_samples_split', 'max_leaf_nodes' or 'min_impurity_decrease' allow
growing asymmetric trees and apply a constraint at the leaves or nodes level.
"""
min_samples_leaf = 60
"""
This hyperparameter allows to have leaves with a minimum number of samples and no
further splits are searched otherwise. Therefore, these hyperparameters could be an
alternative to fix the max_depth 
"""
dtc = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
fit_and_plot_classification(
	dtc, data_clf, data_clf_columns, target_clf_column
)
plt.title(
	f"Decision tree with leaf having at least {min_samples_leaf} samples"
)

_, ax = plt.subplots(figsize=(10, 7))
plot_tree(dtc, ax=ax, feature_names=data_clf_columns)
plt.show()
