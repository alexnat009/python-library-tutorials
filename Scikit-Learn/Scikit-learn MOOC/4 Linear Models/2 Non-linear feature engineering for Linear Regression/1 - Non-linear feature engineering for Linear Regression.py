import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, SplineTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

"""
Even if linear models are not natively adapted to express a target that is not a linear function
of the data, it is still possible to make linear models more expressive by engineering additional
features

A machine learning pipeline that combines a non-linear feature engineering step followed by 
a linear regression step can therefore be considered a non-linear regression model as a whole
"""
rng = np.random.RandomState(0)
n_samples = 100
data_min, data_max = -1.4, 1.4
len_data = data_max - data_min
# sort the data to make plotting easier later
data = np.sort(rng.rand(n_samples) * len_data - len_data / 2)
noise = rng.randn(n_samples) * 0.3
target = data ** 3 - 0.5 * data ** 2 + noise

full_data = pd.DataFrame({
	"input_feature": data,
	"target": target
})

sns.scatterplot(
	data=full_data,
	x="input_feature",
	y="target",
	color="black",
	alpha=0.5
)


# X should be 2D for sklearn: (n_samples, n_features)
data = data.reshape((-1, 1))


def fit_score_plot_regression(model, title=None):
	plt.figure()
	model.fit(data, target)
	target_predicted = model.predict(data)
	mse = mean_squared_error(target, target_predicted)
	ax = sns.scatterplot(
		data=full_data,
		x="input_feature",
		y="target",
		color="black",
		alpha=0.5
	)

	ax.plot(data, target_predicted)
	if title is not None:
		ax.set_title(title + f" (MSE = {mse:.2f})")
	else:
		ax.set_title(f"Mean squared error = {mse:.2f}")



lr = LinearRegression()
fit_score_plot_regression(lr, title="Simple Linear Regression")
print(
	f"weight: {lr.coef_[0]:.2f}, "
	f"intercept: {lr.intercept_:.2f}"
)

"""
Notice that the learnt model can't handle the non-linear relationship
between data and target because linear models assume a linear relationship
there are 3 possible ways to solve this issue:
	1) choose a model that can natively deal with non-linearity
	2) engineer a richer set of features by including expert knowledge
	   which can be directly used by a simple linear model
	3) use a "kernel" to have a locally-based decision function instead
	   of a global linear decision function
"""

# 1) Model that supports non-linearity
dtr = DecisionTreeRegressor(max_depth=3).fit(data, target)
fit_score_plot_regression(dtr, title="Decision Tree Regression")

# 2) Engineer a richer set
"""
We know that we have a cubic and squared relationship between data and target.
We could create two new features (data**2 and data**3). This kind of transformation
is called polynomial feature expansion
"""
print(data.shape)
data_expanded = np.concatenate([data, data ** 2, data ** 3], axis=1)
print(data_expanded.shape)

"""
Instead of manually creating such polynomial features we could use:
"""
polynomial_expansion = PolynomialFeatures(degree=3, include_bias=False)

"""
We can verify that this procedure is equivalent to creating the features by hand:
compute the maximum of the absolute values of the differences between them
"""
print(np.abs(polynomial_expansion.fit_transform(data) - data_expanded).max())

polynomial_regression = make_pipeline(
	PolynomialFeatures(degree=3, include_bias=False),
	LinearRegression()
)
fit_score_plot_regression(polynomial_regression, title="Polynomial Regression")

# 3) Use of "Kernel"
svr = SVR(kernel="linear")
fit_score_plot_regression(svr, title="Linear support vector machine")
svr_3 = SVR(kernel="poly", degree=3)
fit_score_plot_regression(svr_3, title="Polynomial support vector machine")

"""
For larger datasets with n_samples >> 10000, it is often computationally 
more efficient to perform explicit feature expansion using PolynomialFeatures or other
non-linear transformers from scikit-learn such as KBinsDiscretizer or SplineTransformer
"""

binned_regression = make_pipeline(
	KBinsDiscretizer(n_bins=8),
	LinearRegression()
)
fit_score_plot_regression(binned_regression, title="Binned regression")

spline_regression = make_pipeline(
	SplineTransformer(degree=3, include_bias=False),
	LinearRegression()
)
fit_score_plot_regression(spline_regression, title="Spline regression")

nystroem_regression = make_pipeline(
	Nystroem(kernel="poly", degree=3, n_components=5, random_state=0),
	LinearRegression()
)
fit_score_plot_regression(nystroem_regression, title="Nystroem regression")
plt.show()