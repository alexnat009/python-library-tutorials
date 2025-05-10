import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

df = pd.read_csv("../../datasets/ames_housing_no_missing.csv")


def return_cv_results(model, X, y, cv, **kwargs):
	return cross_validate(
		model,
		X,
		y,
		cv=cv,
		scoring="neg_mean_absolute_error",
		return_estimator=True,
		return_train_score=True,
		**kwargs
	)


features_of_interest = [
	"LotFrontage",
	"LotArea",
	"PoolArea",
	"YearBuilt",
	"YrSold",
]
target_name = "SalePrice"
X, y = (
	df[features_of_interest],
	df[target_name],
)

model = make_pipeline(
	PolynomialFeatures(degree=2, include_bias=False),
	LinearRegression()
).set_output(transform="pandas")
cv_results = return_cv_results(model, X, y, 10)
train_error = -cv_results["train_score"]
test_error = -cv_results["test_score"]
print(
	"Mean squared error of linear regression model on the train and test sets:\n"
	f"Train set:{train_error.mean():.2e} ± {train_error.std():.2e}\n"
	f"Test set:{test_error.mean():.2e} ± {test_error.std():.2e}"
)

model_first_fold: Pipeline = cv_results["estimator"][0]
feature_names = model_first_fold[-1].feature_names_in_
print(feature_names)

coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_linear_regression = pd.DataFrame(coefs, columns=feature_names)

color = {"whiskers": "black", "medians": "black", "caps": "black"}
fig, ax = plt.subplots(figsize=(10, 10))
weights_linear_regression.plot.box(color=color, vert=False, ax=ax)
ax.set(title="Linear regression weights (linear scale)", xscale="symlog")
plt.show()

"""
Observe that some coefficients are extremely large while others are extremely small,
yet non-zero. Furthermore, the coefficient values can be very unstable across cross-validations

We can force the linear regression model to consider all features in a more homogeneous manner.
In fact, we could force large positive or negative weight to shrink toward zero. This is known as regularization
We use a ridge model which enforces such behavior 
"""

ridge = make_pipeline(
	PolynomialFeatures(degree=2, include_bias=False),
	Ridge(alpha=100, solver="cholesky")
)

cv_results = return_cv_results(ridge, X, y, 20)
train_error = -cv_results["train_score"]
test_error = -cv_results["test_score"]
print(
	"Mean squared error of linear regression model on the train and test sets:\n"
	f"Train set:{train_error.mean():.2e} ± {train_error.std():.2e}\n"
	f"Test set:{test_error.mean():.2e} ± {test_error.std():.2e}"
)
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_ridge = pd.DataFrame(coefs, columns=feature_names)

fig, ax = plt.subplots(figsize=(10, 10))
weights_ridge.plot.box(color=color, vert=False, ax=ax)
ax.set(title="Ridge regression weights")
plt.show()

scaled_ridge = make_pipeline(
	MinMaxScaler(),
	PolynomialFeatures(degree=2, include_bias=False),
	Ridge(alpha=10, solver="cholesky")  # You can test alpha=10000
)

cv_results = return_cv_results(scaled_ridge, X, y, 10)
train_error = -cv_results["train_score"]
test_error = -cv_results["test_score"]
print(
	"Mean squared error of linear regression model on the train and test sets:\n"
	f"Train set:{train_error.mean():.2e} ± {train_error.std():.2e}\n"
	f"Test set:{test_error.mean():.2e} ± {test_error.std():.2e}"
)

coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_ridge_scaled_data = pd.DataFrame(coefs, columns=feature_names)
fig, ax = plt.subplots(figsize=(8, 10))
weights_ridge_scaled_data.plot.box(color=color, vert=False, ax=ax)
ax.set(title="Ridge regression weights with data scaling")
plt.show()

alphas = np.logspace(-7, 5, num=100)
ridge = make_pipeline(
	MinMaxScaler(),
	PolynomialFeatures(degree=2, include_bias=False),
	RidgeCV(alphas=alphas, store_cv_results=True)
)

cv = ShuffleSplit(n_splits=50, random_state=0)
cv_results = return_cv_results(ridge, X, y, cv, n_jobs=2)
train_error = -cv_results["train_score"]
test_error = -cv_results["test_score"]
print(
	"Mean squared error of linear regression model on the train and test sets:\n"
	f"Train set:{train_error.mean():.2e} ± {train_error.std():.2e}\n"
	f"Test set:{test_error.mean():.2e} ± {test_error.std():.2e}"
)

mse_alphas = [
	est[-1].cv_results_.mean(axis=0) for est in cv_results["estimator"]
]
cv_alphas = pd.DataFrame(mse_alphas, columns=alphas)
cv_alphas = cv_alphas.aggregate(["mean", "std"]).T

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(cv_alphas.index, cv_alphas["mean"], yerr=cv_alphas["std"])
ax.set(
	xscale="log",
	xlabel="alpha",
	yscale="log",
	ylabel="Mean squared error\n (lower is better)",
	title="Testing error in RidgeCV's inner cross-validation",
)

plt.show()

best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
print(best_alphas)

print(
	f"Min optimal alpha: {np.min(best_alphas):.2f} and "
	f"Max optimal alpha: {np.max(best_alphas):.2f}"
)
