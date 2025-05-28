import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeRegressor

"""
We present hot to compare the generalization performance of a model to a minimal baseline. In
regression, we can use the DummyRegressor class to predict the mean target value observered on the 
training set without using the input features
"""

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y *= 100

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

dtr = DecisionTreeRegressor()
cv_results_dtr = cross_validate(
	dtr,
	X,
	y,
	cv=cv,
	scoring="neg_mean_absolute_error",
	n_jobs=2
)

errors_dtr = pd.Series(
	-cv_results_dtr["test_score"], name="Decision Tree Regressor"
)
print(errors_dtr.describe())

dummy = DummyRegressor(strategy="mean")
cv_dummy = cross_validate(
	dummy,
	X,
	y,
	cv=cv,
	scoring="neg_mean_absolute_error",
	n_jobs=2
)

errors_dummy = pd.Series(
	-cv_dummy["test_score"], name="Dummy Regressor"
)
print(errors_dummy.describe())

all_errors = pd.concat(
	[errors_dtr, errors_dummy],
	axis=1
)
print(all_errors)

bins = np.linspace(0, 100, 80)
all_errors.plot.hist(bins=bins, edgecolor="black")
plt.legend(loc="upper left")
plt.xlabel("Mean absolute error (k$)")
plt.title("Cross-validation testing errors")
plt.show()

"""
We see that the generalization performance of our decision tree is far from being perfect:
the price predictions are off by more than 45K$ on average. However it is much better than the mean price
baseline. So this confirms that it is possible to predict the housing price much better by using a model
that takes into account the values of the input features. Such a model makes more informed predictions and 
approximately divides the error rate by a factor of 2 compared to baseline that ignores the input features

Note that here we used the mean price as the baseline prediction. We could have used the median instead.

"""

cv_dummy_r2 = cross_validate(
	dummy,
	X,
	y,
	cv=cv,
	scoring="r2",
	return_train_score=True,
	n_jobs=2
)
r2_train_score_dummy = pd.Series(
	cv_dummy_r2["train_score"], name="Dummy Regressor Train Score"
)
print(r2_train_score_dummy.describe())

"""
The R^2 score is always 0. It can be shown that this is always the case, because of its
mathematical definition. 
"""

"""
This helps put your model's R^2 score in perspective: If your model has an r^2 score higher
than 0 then it performs better than a DummyRegressor with strategy="mean"; similarly, if the R^2 is lower
than 0 then your model is worse than the dummy regressor. For the test score, we observe something similar, 
but with an additional effect coming from the dataset variations: The mean target value measured on the testing set is
slightly different from the mean target value measured on the training set
"""

r2_test_score_dummy = pd.Series(
	cv_dummy_r2["test_score"], name="Dummy Regressor Test Score"
)
print(r2_test_score_dummy.describe())

"""
In conclusion, R^2 is a normalized metric, which makes it independent of the physical unit of the 
target variable, unlike MAE. R^2 score of 0.0 is the performance of a model that always predict the mean
observed value of the target, while 1.0 corresponds to a model that predicts exactly the observed target variable
for each given input observation. Notice that it is only possible to reach 1.0 if the target variable is a deterministic
function of the available input features. In practice, external factors often introduce variability in the target 
that cannot be explained by the available features. Therefore, the R^2 score of an optimal model is typically
less than 1.0, not due to a limitation of the ML algorithms itself, but because the chosen input features are 
fundamentally not informative enough to deterministically predict the target.

Overall, R^2 represents the proportion of the target's variability explained by the model, while MAE, which
retains the physical units of the target, can be helpful for reporting error in those units.  
"""

