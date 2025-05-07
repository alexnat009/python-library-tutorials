import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_gaussian_quantiles
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, SplineTransformer, PolynomialFeatures

"""
Linear classification models are not suited to non-linear separable data. Nevertheless
one can still use feature engineering as previously done for regression models to overcome this issue
To do so, we use non-linear transformation that typically map the original feature space into a higher
dimension space, where the linear model can separate the data more easily
"""

feature_names = ["Feature #0", "Feature #1"]
target_name = "class"

X, y = make_moons(n_samples=100, noise=0.13, random_state=42)

# We store both the data and target in a dataframe to ease plotting
moons = pd.DataFrame(
	np.concatenate([X, y[:, np.newaxis]], axis=1),
	columns=feature_names + [target_name]
)
data_moons, target_moons = moons[feature_names], moons[target_name]

X, y, = make_gaussian_quantiles(n_samples=100, n_features=2, n_classes=2, random_state=42)
gauss = pd.DataFrame(
	np.concatenate([X, y[:, np.newaxis]], axis=1),
	columns=feature_names + [target_name]
)
data_gauss, target_gauss = gauss[feature_names], gauss[target_name]

xor = pd.DataFrame(
	np.random.RandomState(0).uniform(low=-1, high=1, size=(200, 2)),
	columns=feature_names,
)
target_xor = np.logical_xor(xor["Feature #0"] > 0, xor["Feature #1"] > 0)
target_xor = target_xor.astype(np.int32)
xor["class"] = target_xor
data_xor = xor[feature_names]

# Put your datasets and titles into a list of dicts or tuples
datasets = [
	{"data": data_moons, "target": target_moons, "title": "The moons dataset"},
	{"data": data_gauss, "target": target_gauss, "title": "The Gaussian quantiles dataset"},
	{"data": data_xor, "target": target_xor, "title": "The XOR dataset"},
]

# Create subplots dynamically based on the number of datasets
_, axs = plt.subplots(ncols=len(datasets), figsize=(5 * len(datasets), 4), constrained_layout=True)

common_scatter_plot_params = dict(
	cmap=ListedColormap(["tab:red", "tab:blue"]),
	edgecolor="white",
	linewidth=1,
)

for ax, dataset in zip(axs, datasets):
	data = dataset["data"]
	target = dataset["target"]
	title = dataset["title"]

	ax.scatter(
		data[feature_names[0]],
		data[feature_names[1]],
		c=target,
		**common_scatter_plot_params,
	)
	ax.set(
		title=title,
		xlabel=feature_names[0],
		ylabel=feature_names[1] if ax is axs[0] else None,  # Label y-axis only once
	)

plt.show()


def plot_decision_boundary(model, title=None):
	datasets = [
		(data_moons, target_moons),
		(data_gauss, target_gauss),
		(data_xor, target_xor),
	]

	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), layout='tight')  # type:figure.Figure, axes.Axes
	for i, ax, (data, target) in zip(range(len(datasets)), axs, datasets):
		model.fit(data, target)
		DecisionBoundaryDisplay.from_estimator(
			model,
			data,
			response_method="predict_proba",
			plot_method="pcolormesh",
			cmap="RdBu",
			alpha=0.8,
			# Setting vmin and vmax to the extreme values of the probability to
			# ensure that 0.5 is mapped to white (the middle) of the blue-red
			# colormap.
			vmin=0,
			vmax=1,
			ax=ax,
		)
		DecisionBoundaryDisplay.from_estimator(
			model,
			data,
			response_method="predict_proba",
			plot_method="contour",
			alpha=0.8,
			levels=[0.5],  # 0.5 probability contour line
			linestyles="--",
			linewidths=2,
			ax=ax,
		)
		ax.scatter(
			data[feature_names[0]],
			data[feature_names[1]],
			c=target,
			**common_scatter_plot_params,
		)
		if i > 0:
			ax.set_ylabel(None)
	if title is not None:
		fig.suptitle(title)
	plt.show()


lr = make_pipeline(StandardScaler(), LogisticRegression())
plot_decision_boundary(lr, title="Linear classifier")

# Engineering non-liner features

"""
As we did for the linear regression models, we now attempt to build a more
expressive ML pipeline by leveraging non-linear feature engineering, with techniques such as
binning, splines, polynomial features, and kernel approximation
"""

bin_classifier = make_pipeline(
	KBinsDiscretizer(n_bins=5, encode="onehot"),  # already the default params
	LogisticRegression()
)
plot_decision_boundary(bin_classifier, title="Binning classifier")

spline_classifier = make_pipeline(
	SplineTransformer(degree=3, n_knots=5),
	LogisticRegression()
)
plot_decision_boundary(spline_classifier, title="Spline classifier")

"""
Both KBinsDiscretizer and SplineTransformer are feature-wise transformations and thus
cannot capture interactions between features

Also KBinsDiscretizer(encode="onehot") and SplineTransformer do not require additional scaling
They can replace the scaling step for numerical features 
"""

poly_classifier = make_pipeline(
	StandardScaler(),
	PolynomialFeatures(degree=3, include_bias=False),
	LogisticRegression(C=10)
)
plot_decision_boundary(poly_classifier, title="Polynomial classifier")

nystroem_classifier = make_pipeline(
	StandardScaler(),
	Nystroem(kernel="poly", degree=3, coef0=1, n_components=100),
	LogisticRegression(C=10)
)
plot_decision_boundary(nystroem_classifier, title="nystroem classifier")

"""
The polynomial kernel approach would be interesting in cases where the original feature
space is already of high dimension: in these cases, computing the complete polynomial expansion with
PolynomialFeatures could be intractable, while the Nystroem method can control the output dimensionality
with the n_components parameter
"""
rbf_classifier = make_pipeline(
	StandardScaler(),
	Nystroem(kernel="rbf", gamma=1, n_components=100),
	LogisticRegression(C=5)
)
plot_decision_boundary(rbf_classifier, title=" RBF Nystroem classifier")

# Multi-step feature engineering

"""
It is possible to combine several feature engineering transformers in a single pipeline
to blend their respective biases
"""

bins_rbf_classifier = make_pipeline(
	KBinsDiscretizer(n_bins=5),
	Nystroem(kernel="rbf", gamma=1, n_components=100),
	LogisticRegression()
)
plot_decision_boundary(bins_rbf_classifier, title="Binning + Nystroem classifier")

spline_rbf_classifier = make_pipeline(
	SplineTransformer(n_knots=5),
	Nystroem(kernel="rbf", gamma=1.0, n_components=100),
	LogisticRegression(),
)
plot_decision_boundary(spline_rbf_classifier, title="Spline + RBF Nystroem classifier")

