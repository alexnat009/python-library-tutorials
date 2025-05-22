import pandas as pd
from scipy.stats import loguniform
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y *= 100  # rescale the target in k$
X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=0
)
"""
For the sake of clarity, no nested cross-validation is used to estimate the variability
of the testing error. only the effect of the parameters on the validation set is shown
"""

"""
Random Forest

The main parameter to select in random forest is the n_estimators parameter. In general, the more
trees in the forest, the better the generalization performance would be. However, adding trees slows down
the fitting and prediction time. The goal is to balance computing time and generalization performance when
setting the number of estimators. Here we fix n_estimators=100, which is already the default value


Tuning the n_estimators for random forests generally result in a waste of computing power. We just need to 
ensure that it is large enough so that doubling its value doesn't lead to a significant improvement of 
the validation error

Instead, we can tune the hyperparameter max_features, which controls the size of random subset of features
to consider when looking ofr the best split when growing the trees: smaller values of max_features lead to more
random trees with hopefully more uncorrelated prediction errors. However if max_features is too small, prediction
can be too random, even after averaging the trees in the ensemble

If max_features is set to None, then this is equivalent to setting max_features=n_features which means
that the only source of randomness is the bagging procedure 
"""

print(f"In our case, n_features={len(X.columns)}")

"""
We can also tune the different parameters that control the depth of each tree in the forest.
Two parameters are important for this: max_depth and max_leaf_nodes. They differ in the way they 
control the tree structure. Indeed, max_depth enforces growing symmetric trees, while max_leaf_nodes
doesn't impose such constraint. If max_leaf_nodes=None then the number of leaf nodes in unlimited


The hyperparameter min_samples_leaf controls the minimum number of samples required to be at a leaf node.
This means that a split point is only done if it leaves at leas min_samples_leaf training samples in each of
the left and right branches. A small value for min_samples_leaf means that some samples can become isolated when a tree
is deep, promoting overfitting. A large value would prevent deep trees, which can lead to underfitting.

Be aware that with random forest, trees are expected to be deep since we are seeking to overfit each tree
on each bootstrap sample. Overfitting is mitigated when combining the trees altogether, whereas assembling
underfitted trees might also lead to an underfitted forest 
"""

param_distributions = {
	"max_features": [1, 2, 3, 5, None],
	"max_leaf_nodes": [10, 100, 1000, None],
	"min_samples_leaf": [1, 2, 5, 10, 20, 50, 100]
}

search_cv = RandomizedSearchCV(
	RandomForestRegressor(n_jobs=2),
	param_distributions=param_distributions,
	scoring="neg_mean_absolute_error",
	n_iter=10,
	random_state=0,
	n_jobs=2
)

search_cv.fit(X_train, y_train)

columns = [f"param_{name}" for name in param_distributions.keys()] + ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = cv_results["std_test_score"]
print(cv_results[columns].sort_values(by="mean_test_error").iloc[0])

"""
We can observe in our search that we are required to have a large number of max_leaf_nodes and thus deep trees.
This parameter seem particularly impactful with respect to the other tuning parameters, but large values of
min_samples_leaf seem to reduce the performance of the model

In practice, more iterations of random search would be necessary to precisely assert the role
of each parameters. Using n_iter=10 is good enough to quickly inspect the hyperparameter combinations
that yield models that work well enough without spending too much computational resources.

Once the RandomizedSearchCV has found the best set of hyperparameters, it uses them to refit the model
using the full training set. To estimate the generalization performance of the best model it
suffices to call .score on the unseen data
"""

error = -search_cv.score(X_test, y_test)
print(
	f"On average, our random forest regressor makes an error of {error:.2f} k$"
)

"""
Histogram Gradient-Boosting Decision Trees

for gradient-boosting, hyperparameters are coupled, so we cannot set them one after the other anymore.
The important hyperparameters are max_iter, learning_rate and max_depth or max_leaf_nodes

max_iter similarly to n_estimators in random forest, controls the number of trees in the estimator. The 
difference is that the actual number of trees trained by the model is not entirely set by the user, but
depends also on the stopping criteria: the number of trees can be lower that max_iter if adding a new tree doesn't
improve the model enough.

The depth of the trees is controlled by max_depth (pr max_leaf_nodes). We saw that boosting algorithms
fit the error of the previous tree in the ensemble. Thus, fitting grown trees would be detrimental. Indeed, the 
first tree of the ensemble would perfectly fit (overfit) the data and thus no subsequent tree would be required,
since there would be no residuals. Therefore, the tree used in gradient-boosting should have a low depth,
typically between 3-8 levels, or few leaves (2^3=8 to 2^8=256). Having very weak learners at each step helps reducing
overfitting

With this consideration in mind, the deeper the trees, the faster the residuals are corrected and then 
less learners are required. Therefore, it can be beneficial to increase max_iter if max_depth is low

Learning_rate parameter controls how much each correction contributes to the final prediction. A smaller
learning-rate means the corrections of a new tree result in small adjustments to the model prediction.
When the learning-rate is small, the model generally need more trees to achieve good performance. A higher
learning-rate makes larger adjustment with each tree, which requires fewer trees and trains faster, at the risk of
overfitting. The learning-rate need to be tuned by hyperparameter tuning to obtain th ebest results in a model with 
good generalization performance
"""

param_distributions = {
	"max_iter": [3, 10, 30, 100, 300, 1000],
	"max_leaf_nodes": [2, 5, 10, 20, 50, 100],
	"learning_rate": loguniform(0.01, 1)
}

"""
Here, we tune max_iter but be aware that it is better to set max_iter to a fixed, large enough
value and use parameters linked to early_stopping. 
"""

search_cv = RandomizedSearchCV(
	HistGradientBoostingRegressor(),
	param_distributions=param_distributions,
	scoring="neg_mean_absolute_error",
	n_iter=20,
	random_state=0,
	n_jobs=2
)

search_cv.fit(X_train, y_train)

columns = [f"param_{name}" for name in param_distributions.keys()] + ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = cv_results["std_test_score"]
print(cv_results[columns].sort_values(by="mean_test_error").iloc[0])

error = -search_cv.score(X_test, y_test)
print(f"On average, our HGBT regressor makes an error of {error:.2f} k$")

"""
The mean test score in the held-out test set is slightly better than the score of the best
model. The reason is that the final model is refitted on the whole training set and therefore, on more
data than the cross-validated models of the grid search procedure
"""

"""
Summarize Ensemble methods

Bagging & Random Forest				Boosting
-------------------------------------------------------------
fit trees independently				fit trees sequentially

each deep tree overfits				each shallow tree underfits

averaging the tree predictions		sequentially adding trees
reduces overfitting					reduces underfitting
			
generalization improves				too many trees may cause
with the number of trees			overfitting

doesn't have a learning_rate		fitting the residuals is
									controlled by the learning_rate
"""
