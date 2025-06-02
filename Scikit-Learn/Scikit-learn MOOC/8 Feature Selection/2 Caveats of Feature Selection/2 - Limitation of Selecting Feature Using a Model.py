import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

"""
We want to show a limitation when using a ML model to make a selection

Indeed, one can inspect a model and find relative feature importance. For instance, the parameters
coef_ for the linear models or feature_importances_ for the tree-based models carries such information. Therefore, this
method works as far as the relative feature importances given by the model is sufficient to select the meaningful 
feature
"""
data, target = make_classification(
	n_samples=5000,
	n_features=100,
	n_informative=2,
	n_redundant=5,
	n_repeated=5,
	class_sep=0.3,
	random_state=0
)

model_without_selection = RandomForestClassifier()

cv_results_without_selection = cross_validate(model_without_selection, data, target, cv=5)
cv_results_without_selection = pd.DataFrame(cv_results_without_selection)

model_with_selection = make_pipeline(
	SelectFromModel(RandomForestClassifier()),
	RandomForestClassifier()
)

cv_results_with_selection = cross_validate(
	model_with_selection, data, target, cv=5
)

cv_results_with_selection = pd.DataFrame(cv_results_with_selection)
cv_results = pd.concat(
	[cv_results_without_selection, cv_results_with_selection],
	axis=1,
	keys=["Without feature selection", "With feature selection"],
).swaplevel(axis="columns")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 6), layout='tight')  # type:figure.Figure, axes.Axes

color = {"whiskers": "black", "medians": "black", "caps": "black"}
cv_results["test_score"].plot.box(color=color, vert=False, ax=ax)
ax.set_xlabel("Accuracy")
ax.set_title("Limitation of using a random forest for feature selection")
plt.show()
