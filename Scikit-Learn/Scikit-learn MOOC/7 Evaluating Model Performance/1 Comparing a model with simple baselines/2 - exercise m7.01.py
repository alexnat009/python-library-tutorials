import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../datasets/adult-census-numeric-all.csv")
X, y = df.drop(columns="class"), df["class"]

cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())

cv_results = cross_validate(
	model,
	X,
	y,
	cv=cv,
	n_jobs=2
)

test_score_model = pd.Series(
	cv_results["test_score"], name="StandardScaler + LogisticRegression"
)
print(test_score_model)

dummy = DummyClassifier(strategy="most_frequent")

cv_results_dummy = cross_validate(
	dummy,
	X,
	y,
	cv=cv,
	n_jobs=2
)
test_score_dummy = pd.Series(
	cv_results_dummy["test_score"], name="Dummy Classifier"
)
print(test_score_dummy)

all_test_scores = pd.concat(
	[test_score_model, test_score_dummy],
	axis="columns"
)
print(all_test_scores)

bins = np.linspace(0.5, 1.0, 100)
all_test_scores.plot.hist(bins=bins, edgecolor="black")
plt.legend(loc="upper left")
plt.xlabel("Accuracy (%)")
plt.title("Distribution of the CV scores")
plt.show()

"""
We observe that the two histograms are well separated. Therefore the dummy classifier with 
the strategy most_frequent has a much lower accuracy than the logistic regression classifier.
We conclude that the logistic model can successfully find predictíve information in the input features to
improve upon the baseline.
"""

dummy_stratified = DummyClassifier(strategy="stratified")
dummy_uniform = DummyClassifier(strategy="uniform")

# solution
stratified_dummy = DummyClassifier(strategy="stratified")
cv_results_stratified = cross_validate(
	stratified_dummy, X, y, cv=cv, n_jobs=2
)
test_score_dummy_stratified = pd.Series(
	cv_results_stratified["test_score"], name="Stratified class predictor"
)
uniform_dummy = DummyClassifier(strategy="uniform")
cv_results_uniform = cross_validate(
	uniform_dummy, X, y, cv=cv, n_jobs=2
)
test_score_dummy_uniform = pd.Series(
	cv_results_uniform["test_score"], name="Uniform class predictor"
)
all_test_scores = pd.concat(
	[
		test_score_model,
		test_score_dummy,
		test_score_dummy_stratified,
		test_score_dummy_uniform,
	],
	axis="columns",
)
all_test_scores.plot.hist(bins=bins, edgecolor="black")
plt.legend(loc="upper left")
plt.xlabel("Accuracy (%)")
plt.title("Distribution of the test scores")
plt.show()

"""
We see that using strategy="stratified", the results are much worse than with the most_frequent strategy. Since the
classes are imbalanced, predicting the most frequent involves that we will be right for the proportion of this class 
(~75% of the samples). However, the "stratified" strategy will randomly generate predictions by respecting the training
set's class distribution, resulting in some wrong predictions even for the most frequent class, hence we obtain
a lower accuracy

This is even more so for the strategy="uniform": this strategy assigns class labels uniformly at random. Therefore,
on a binary classification problem, the cross-validation accuracy is 50% on average, which is the weakest of the three
dummy baselines.

Defining the change level using permutation_test_score is quite computation-intensive because it requires fitting
many non-dummy models on random permutations of the data. Using dummy classifiers as baselines is often enough for
practical purposes. For imbalanced classification problems, the "most_frequent" strategy is the strongest of the
three baselines and therefore the one we should use 
"""

