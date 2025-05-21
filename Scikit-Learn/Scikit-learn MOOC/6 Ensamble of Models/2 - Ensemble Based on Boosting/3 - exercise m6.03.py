import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, ValidationCurveDisplay

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y *= 100  # rescale the target in k$
X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=0, test_size=0.5
)

gbr = GradientBoostingRegressor(max_depth=5, learning_rate=0.5)
rfr = RandomForestRegressor(max_depth=None)

param_range = np.array([1, 2, 5, 10, 20, 50])
disp = ValidationCurveDisplay.from_estimator(
	estimator=gbr,
	X=X,
	y=y,
	scoring="neg_mean_absolute_error",
	negate_score=True,
	param_name="n_estimators",
	param_range=param_range,
	std_display_style="errorbar",
	n_jobs=3
)
disp.ax_.set(
	xlabel="Number of trees in the forest",
	ylabel="Mean absolute error (k$)",
	title="Validation curve for random forest",
)
plt.show()

disp = ValidationCurveDisplay.from_estimator(
	estimator=rfr,
	X=X,
	y=y,
	scoring="neg_mean_absolute_error",
	negate_score=True,
	param_name="n_estimators",
	param_range=param_range,
	std_display_style="errorbar",
	n_jobs=3
)
disp.ax_.set(
	xlabel="Number of trees in the gradient boosting model",
	ylabel="Mean absolute error (k$)",
	title="Validation curve for gradient boosting model",
)
plt.show()

gbr = GradientBoostingRegressor(n_estimators=1000, n_iter_no_change=5)
gbr.fit(X_train, y_train)
print(gbr.n_estimators_)

"""
Wee see that the number of trees used is far below 1000 with the current dataset. Training the
gradient boosting model with the entire 1000 trees would have been detrimental

Please note that one shouldn't hyperparameter tune the number of estimators for both random forest and
gradient boosting models. In this exercise we only show model performance with varying n_estimators
for educational purposes
"""

y_pred = gbr.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


"""
We observe that MAE value measure on the held out test set is close to the validation error measured to the 
right hand side of the validation curve. This is kind of reassuring, as it means that both the cross-validation
procedure and the outer train-test split roughly agree as approximations of th etrue generalization performance
of the model. We can observe that the final evaluation of the test error seems to be even slightly 
below than the cross-validated test scores. This can be explained because the final model has been trained on the full
training set while the cross-validation models have been trained on smaller subsets:
in general the larger the number of training points, the lower the test error
"""