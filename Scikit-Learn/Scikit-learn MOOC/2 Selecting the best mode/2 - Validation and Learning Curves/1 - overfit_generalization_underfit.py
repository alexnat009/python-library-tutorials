import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.model_selection import ValidationCurveDisplay

df = fetch_california_housing(as_frame=True)
X, y = df.data, df.target
y *= 100

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
regressor = DecisionTreeRegressor()
cv_results = cross_validate(regressor, X, y, cv=cv, scoring="neg_mean_absolute_error", return_train_score=True,
							n_jobs=3)

cv_results = pd.DataFrame(cv_results)
print(cv_results)

scores = pd.DataFrame()
scores[["train_error", "test_error"]] = -cv_results[["train_score", "test_score"]]
scores.plot.hist(bins=50, edgecolor="black")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Train and test errors distribution via cross-validation")
plt.show()

"""
Here we observe, a small training error (actually zero), meaning that the model is not under-fitting
it is flexible enough to capture any variations present in the training set

However the significantly larger testing error tells us that the model is over-fitting
the model has memorized many variations of the training set that could be considered "noisy"
because they do not generalize to help us make good predictions on the test set
"""

# Validation Curve
"""
We call hyperparameters those parameters that potentially impact the result of the learning
and subsequent predictions of a prediction
"""

max_depth = np.array([1, 5, 10, 15, 20, 25])
disp = ValidationCurveDisplay.from_estimator(
	regressor,
	X,
	y,
	param_name="max_depth",
	param_range=max_depth,
	cv=cv,
	scoring="neg_mean_absolute_error",
	negate_score=True,
	std_display_style="errorbar",
	n_jobs=3,
)

disp.ax_.set(
    xlabel="Maximum depth of decision tree",
    ylabel="Mean absolute error (k$)",
    title="Validate curve for decision tree",
)
plt.show()

"""
The validation curve can be divided into three areas:
○ max_depth < 10:
	the decision tree underfits. The training error and therefore the testing error are both high
	The model is too constrained and cannot capture much of the variability of the target variable
	
○ max_depth = 10:
	corresponds to the parameter of which the decision tree generalizes the best.
	It is flexible enough to capture a fraction of the variability of the target that 
	generalizes, while not memorizing all of the noise in the target
	
○ max_depth > 10:
	The decision tree overfits. The training error becomes very small, while the testing error
	increases. In this region, the models create decisions specifically for noisy sample harming
	its ability to generalize to test data
"""

"""
Note that for max_depth = 10, the model overfits a bit as there is a gap between the training
error and the testing error. It can also potentially underfit also a bit at the same time, because
the training error is still far from zero. meaning that the model might still be too constrained to model
interesting parts of the data. However, the testing error is minimal, and this is what really matters.
This is the best compromise we could reach by just tuning this parameters

Be aware that looking at the mean error is quite limiting. We should also look at the standard deviation
to assess the dispersion of the score. For such purpose, we can use the parameters 'std_displays_style' to
show the standard deviation of the errors as well. In this case, the variance of the errors is small compared to their
respective values, and therefore the conclusions above are quite clear. This is not necessarily always the case
"""