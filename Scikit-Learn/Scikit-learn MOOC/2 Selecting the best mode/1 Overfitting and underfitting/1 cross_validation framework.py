import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print(housing.DESCR)
print(X)

y *= 100  # transform the prices from 100 (k$) range to the thousand dollars (k$) range
print(y)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

target_predicted = regressor.predict(X)
score = mean_absolute_error(y, target_predicted)
print(f"On average, our regressor makes an error of {score:.2f} k$")

"""
We get perfect prediction with no error, which is too optimistic
We trained and predicted on the same dataset. Since our decision tree was fully grown,
every sample in the dataset is stored in a leaf. Therefore our decision tree fully memorized
the dataset given during fit and therefore made no error when predicting

This error computed about is called the 'empirical error' or 'training error'

We trained a predictive model to minimize the training error but our aim is to minimze
the error on data that has not been seen during training

This error is called the 'generalization error' or 'true' testing error
"""

"""
Thus the most basic evaluation involved:
 1) Splitting our dataset into two subsets: training and test sets
 2) fitting the model on the training set
 3) estimating the training error on the training set
 4) estimating the testing error on the testing set
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

regressor.fit(X_train, y_train)
target_predicted = regressor.predict(X_train)
score = mean_absolute_error(y_train, target_predicted)
print(f"The training error of our model is {score:.2f} k$")

target_predicted = regressor.predict(X_test)
score = mean_absolute_error(y_test, target_predicted)
print(f"The training error of our model is {score:.2f} k$")

"""
Stability of the cross-validation estimates
When doing a single train-test split we don't give any indication regarding the
robustness of the evaluation of our predictive model

Cross-validation allows estimating the robustness of a predictive model by repeating the
splitting procedure. It will give several training and testing errors and thus some estimate 
of the variability of the model generalization performance
"""

# There are different cross-validation strategies, mainly used for different data:
"""
1) Cross-validation iterators for i.i.d. data
	Assuming that some data is independent and identically distributed (i.i.d) is making the
	assumption that all samples stem from the same generative process and the generative process is assumed
	to have no memory of past generates samples. One should use following techniques:
		1) K-fold
		2) Repeated K-fold
		3) Leave One Out (LOO)
		4) Leave P Out (LPO)
		5) Shuffle & Split
		
2) Cross-validation iterators with stratification based on class labels
	Some classification problems can exhibit a large imbalance in the distribution of the target classes:
	for instance there could be several time more negative samples than positives. In such cases
	it's recommended to use stratified sampling to ensure that relative class frequencies is approximately preserved
	in each train and validation fold
		1) Stratified K-Fold
		2) Stratified Shuffle Split
		3) Predefined fold-splits / Validation-sets
		
3) Cross-validation iterators for grouped data
	The i.i.d assumption is broken if the underlying generative process yields groups of dependent samples
	Such a grouping of data is domain specific. An example would be when there is medical data collected
	from multiple patients, with multiple samples taken from each patient. And such data is likely to be
	dependent on the individual group. In this case we would like to know if a model trained on a particular
	set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the
	validation fold come from groups that are not represented at all in the paired training fold
		1) Group K-Fold
		2) StratifiedGroupKFold
		3) Leave One Group Out
		4) Leave P Groups Out
		5) Group Shuffle Split
		
	NOTE: you can use cross-validation iterators to split train and test see scikit-learn example for reference
4) Cross validation of time series data
	Time series data is characterized by the correlation between observations that are near in time. However,
	classical cross-validation techniques such as KFold and ShuffleSplit assume that samples are i.i.d and would
	result in unreasonable correlation between training and testing instances on time series data. Therefore it is 
	very important to evaluate our model for time series data on the "future" observation least like those that are
	used to train the model.
		1) Time Series Split
"""

"""
Lets take example of shuffle-split. At each iteration of this strategy we:
	1) randomly shuffle the order of the samples of a copy of the full dataset
	2) split the shuffled dataset into a training and a test set
	3) train a new model on the train set
	4) evaluate the testing error on the test set

We repeat this procedure 'n_splits' times.
"""
cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(regressor, X, y, cv=cv, scoring="neg_mean_absolute_error", verbose=2, n_jobs=3)

cv_results = pd.DataFrame(cv_results)
print(cv_results)

"""
A 'score' is a metric for which higher values mean better results.
An 'error' is a metric for which lower values mean better results

The parameter scoring in cross_validate expects a function that is a score
To make it easy, all error metrics in scikit-learn can be transformed into a score by
appending 'neg_' at the front. e.g. scoring="neg_mean_absolute_error"
"""

# let us revert the negation to get the actual error:
cv_results["test_error"] = -cv_results["test_score"]
print(cv_results)

cv_results["test_error"].plot.hist(bins=10, edgecolor="black")
plt.xlabel("Mean Absolute Error (k$)")
plt.title("Test Error Distribution")
plt.show()

print(f"The mean cross-validated testing error is: {cv_results['test_error'].mean():.2f} k$")
print(f"The standard deviation of the testing error is: {cv_results['test_error'].std():.2f} k$")

y.plot.hist(bins=20, edgecolor="black")
plt.xlabel("Median House Value (k$)")
plt.title("Target Distribution")
plt.show()

"""
We see that our model makes, on average, an error around 47 k$, when predicting houses with a
value of 50 k$ this would be a problem. Hence this indicates that our metric is not ideal
"""

# More details regarding cross_validate
"""
During cross-validation, many models are trained an evaluated. It is possible to retrieve these fitted models
for each of the folds by passing the option return_estimator=True in cross_validate
"""
cv_results = cross_validate(regressor, X, y, return_estimator=True)

print(pd.DataFrame(cv_results))

"""
In the case where you only are interested in the test score, scikit-learn provides a
'cross_val_score' function. It is identical to calling the cross_validate function and to select the 'test_score'
only
"""

scores = cross_val_score(regressor,X,y)
print(scores)