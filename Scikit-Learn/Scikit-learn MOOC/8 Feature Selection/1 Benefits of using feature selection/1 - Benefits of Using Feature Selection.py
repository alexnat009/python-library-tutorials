import numpy as np
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

X, y = make_classification(
	n_samples=5000,
	n_features=100,
	n_informative=2,
	n_redundant=0,
	n_repeated=0,
	random_state=0,
)

"""
We choose to create a dataset with two informative features among a hundred. To simplify
our example, we don't include either redundant or repeated features

We will create two ML pipelines. The former will be a random forest that will use all
available features. The latter will also be a random forest, but we will add a feature
selection step to train this classifier. The feature selection is based on a univariate test
(ANOVA F-value) between each feature and the target that we want to predict. The features with the
two most significant scores are selected.
"""

model_without_selection = RandomForestClassifier(n_jobs=2)

model_with_selection = make_pipeline(
	SelectKBest(score_func=f_classif, k=2),
	RandomForestClassifier(n_jobs=2)
)

"""
We will measure the average time spent to train each pipeline and make it predict. Besides we will compute
the testing score of the model. We will collect these results via cross-validation
"""

cv_results_without_selection = cross_validate(model_without_selection, X, y)
cv_results_without_selection = pd.DataFrame(cv_results_without_selection)
cv_results_with_selection = cross_validate(model_with_selection, X, y, return_estimator=True)
cv_results_with_selection = pd.DataFrame(cv_results_with_selection)

cv_results = pd.concat(
	[cv_results_without_selection, cv_results_with_selection],
	axis=1,
	keys=["Without feature selection", "With feature selection"]
)

cv_results = cv_results.swaplevel(axis="columns")

color = {"whiskers": "black", "medians": "black", "caps": "black"}

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 5), layout='tight')  # type:figure.Figure, axes.Axes
cv_results["fit_time"].plot.box(color=color, vert=False, ax=ax)
ax.set_xlabel("Elapsed time (s)")
ax.set_title("Time to fit the model")
plt.show()

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 5), layout='tight')  # type:figure.Figure, axes.Axes
cv_results["score_time"].plot.box(color=color, vert=False, ax=ax)
ax.set_xlabel("Elapsed time (s)")
ax.set_title("Time to make prediction")
plt.show()

"""
We can draw the same conclusion for both training and scoring elapsed time: selecting the most
informative features speed-up our pipeline

Such speed-up is beneficial only of the generalization performance in terms of metrics remain the same.
"""

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 5), layout='tight')  # type:figure.Figure, axes.Axes
cv_results["test_score"].plot.box(color=color, vert=False, ax=ax)
ax.set_xlabel("Accuracy score")
ax.set_title("Test score via cross-validation")
plt.show()

"""
We can observe that the model's generalization performance selecting a subset of features
decreases compared with the model using all available features. Since we generated the dataset, we
can infer that the decrease is because of the selection. The feature selection algorithm didn't choose the 
two informative features

We can investigate which feature have been selected during the cross-validation. We will pring
the indices of the two selected features
"""

for idx, pipeline in enumerate(cv_results_with_selection["estimator"]):
	print(f"Fold #{idx} - features selected are: {np.argsort(pipeline[0].scores_)[-2:]}")

"""
We see that the feature 53 is always selected while the other feature varies depending on the
cross-validation fold

If we would like to keep our score with similar generalization performance, we could choos another metric to perform
the test or select more features. For instance, we could select the number of features based on a specific percentile 
of the highest scores. Besides, we should keep in mind that we simplify our problem by having informative and not
informative features. Correlation between features makes the problem of feature selection even harder

Therefore, we could come with a much more complicated procedure that could tune (via cross-validation) the number
of selected features and change the way features is selected (using ML model). However, going towards these solution 
alienates the features selection's primary purpose to get a significant train/test speed-up. also if the
primary goal was to get a more performant model, performant models exclude non-informative features natively. 
"""
