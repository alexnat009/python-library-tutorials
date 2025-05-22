import warnings

from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

"""
We present a modified version of gradient boosting which uses a reduced number of splits when
building the different trees. This algorithm is called "histogram gradient boosting" in scikit-learn

We previously mentioned that random-forest is an efficient algorithm since each tree of the ensemble can be fitted
at the same time independently. Therefore, the algorithm scales efficiently with both the number of cores and
the number of samples

In gradient-boosting, the algorithm is a sequential algorithm. It requires the N-1 trees
to have been fit to be able to fit the tree at stage N. Therefore, the algorithm is quite 
computationally expensive. The most expensive part is the search for the best split in the tree
which is a brute-force approach: all possible split are evaluated and then best one is picked.

To accelerate the gradient-boosting algorithm, one could reduce the number of splits to be evaluated.
As a consequence, the generalization performance of such a tree would be reduced. However, since we are combining
several trees in a gradient-boosting, we can add more estimators to overcome this issue

We will make a naive implementation of such algorith using building blocks from scikit-learn
"""

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y *= 100  # rescale the target in k$

gbr = GradientBoostingRegressor(n_estimators=200)
cv_results_gbr = cross_validate(
	gbr,
	X,
	y,
	scoring="neg_mean_absolute_error",
	n_jobs=2
)

print("Gradient Boosting Decision Tree")
print(f"MAE (CV): {-cv_results_gbr['test_score'].mean():.3f} ± {cv_results_gbr['test_score'].std():.3f} k$")
print(f"Avg fit time: {cv_results_gbr['fit_time'].mean():.3f}s")
print(f"Avg score time: {cv_results_gbr['score_time'].mean():.3f}s")

"""
Recall that a way of accelerating the gradient boosting is to reduce the number of split
considered within the tree building. One way is to bin the data before to give them into the gradient 
gradient boosting. A transformer called KBinsDiscretized is doing such transformation. Thus we can pipeling this
preprocessing with the gradient boosting
"""

discretizer = KBinsDiscretizer(
	n_bins=256, encode="ordinal", strategy="quantile"
)
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", message="Bins whose width are too small*")
	data_trans = discretizer.fit_transform(X)
print(data_trans)

"""
We see that the discretizer transform the original data into integral values. Each value represents the bin
index when the distribution by quantile is performed. We can check the number of bins per feature 
"""
print([len(np.unique(col)) for col in data_trans.T])

gbr = make_pipeline(
	discretizer, GradientBoostingRegressor(n_estimators=200)
)
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", message="Bins whose width are too small*")

	cv_results_gbr = cross_validate(
		gbr,
		X,
		y,
		scoring="neg_mean_absolute_error",
		n_jobs=2
	)

print("Gradient Boosting Decision Tree")
print(f"MAE (CV): {-cv_results_gbr['test_score'].mean():.3f} ± {cv_results_gbr['test_score'].std():.3f} k$")
print(f"Avg fit time: {cv_results_gbr['fit_time'].mean():.3f}s")
print(f"Avg score time: {cv_results_gbr['score_time'].mean():.3f}s")

"""
Here we see that the fit time has been reduced but that the generalization performance of the
model is identical. Scikit-learn provides specific classes which are even more optimized for large dataset, called
HistGradientBoostingClassifier and HistGradientBoostingRegressor. Each feature in the dataset data is first binned by 
computing histograms, which are later used to evaluate the potential splits. The number of splits to evaluate
is then much smaller. This algorithm becomes much more efficient then gradient boostingwhen the dataset has over 10000
samples


"""

hgbr = HistGradientBoostingRegressor(
	max_iter=200, random_state=0
)
cv_results_hgbr = cross_validate(
	hgbr,
	X,
	y,
	scoring="neg_mean_absolute_error",
	n_jobs=2
)

print("Gradient Boosting Decision Tree")
print(f"MAE (CV): {-cv_results_hgbr['test_score'].mean():.3f} ± {cv_results_hgbr['test_score'].std():.3f} k$")
print(f"Avg fit time: {cv_results_hgbr['fit_time'].mean():.3f}s")
print(f"Avg score time: {cv_results_hgbr['score_time'].mean():.3f}s")

"""
The histogram gradient-boosting is the best algorithm in terms of score. It will also scale when 
the number of samples  increases, while the normal gradient-boosting will not
"""
