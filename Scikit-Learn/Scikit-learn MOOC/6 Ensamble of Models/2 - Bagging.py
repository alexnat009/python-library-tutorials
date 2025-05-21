import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

"""
Bagging stand for Bootstrap AGGregatIng.It uses bootstrap resampling (random sampling with replacement) to
learn several models on random variations of the training set. At predict time, the predictions of each learner
are aggregated to give the final prediction
"""


def generate_data(n_samples=30):
	"""
	Generate synthetic dataset.
	"""
	x_min, x_max = -3, 3
	rng = np.random.default_rng(1)
	x = rng.uniform(x_min, x_max, size=n_samples)
	noise = 4.0 * rng.normal(size=(n_samples,))
	y = x ** 3 - 0.5 * (x + 1) ** 2 + noise
	y /= y.std()

	data_train = pd.DataFrame(x, columns=["Feature"])
	data_test = pd.DataFrame(
		np.linspace(x_max, x_min, num=300), columns=["Feature"]
	)
	target_train = pd.Series(y, name="Target")
	return data_train, data_test, target_train


data_train, data_test, target_train = generate_data(30)
sns.scatterplot(
	x=data_train["Feature"],
	y=target_train,
	color="black",
	alpha=0.5
)

plt.title("Synthetic regression dataset")
plt.show()

"""
The target to predict is a non-linear function of the only feature. However, a decision tree
is capable of approximating such a non-linear dependency
"""

dtr = DecisionTreeRegressor(max_depth=3, random_state=0)
dtr.fit(data_train, target_train)
y_pred = dtr.predict(data_test)

sns.scatterplot(
	x=data_train["Feature"],
	y=target_train,
	color="black",
	alpha=0.5
)
plt.plot(data_test["Feature"], y_pred, label="Fitted tree")
plt.legend()
plt.title("Predictions by a single decision tree")
plt.show()

"""
Bootstrap resampling

Bootstrapping involves uniformly resampling n data point from a dataset of n point, with 
replacement, ensuring each sample has an equal change of selection.

As a result, the output of the bootstrap sampling procedure is another dataset with n data point
likely containing duplicates. Consequently, some data point from the original dataset may not be selected
for a boostrap sample. These unselected data point are often referred to as the out-of-bag sample

"""


def bootstrap_sample(data, target, seed=0):
	rng = np.random.default_rng(seed)
	bootstrap_indices = rng.choice(
		np.arange(target.shape[0]),
		size=target.shape[0],
		replace=True
	)
	data_bootstrap = data.iloc[bootstrap_indices]
	target_bootstrap = target.iloc[bootstrap_indices]
	return data_bootstrap, target_bootstrap


n_bootstraps = 3
for bootstrap_idx in range(n_bootstraps):
	data_bootstarp, target_bootstrap = bootstrap_sample(
		data_train,
		target_train,
		seed=bootstrap_idx
	)
	plt.figure()
	plt.scatter(
		data_bootstarp["Feature"],
		target_bootstrap,
		color="tab:blue",
		facecolors="none",
		alpha=0.5,
		label="Resampled data",
		s=180,
		linewidth=5
	)
	plt.scatter(
		data_train["Feature"],
		target_train,
		color="black",
		s=60,
		alpha=1,
		label="Original data"
	)
	plt.title(f"Resampled data #{bootstrap_idx}")
plt.legend()
plt.show()

"""
Observe that the 3 variations all share common points with the original dataset. Some of the points
are randomly resampled several times and appear as darker blue circles
"""

"""
The 3 generated bootstrap samples are all different from the original datset and from each othet. To
confirm this intuition, we can check the number of unique samples in the bootstrap sample
"""
data_train_huge, data_test_huge, target_train_huge = generate_data(100000)

data_bootstrap_sample, target_bootstrap_sample = bootstrap_sample(data_train_huge, target_train_huge)

ratio_unique_sample = np.unique(data_bootstrap_sample).size / data_bootstrap_sample.size

print(
	"Percentage of samples present in the original dataset: "
	f"{ratio_unique_sample * 100:.1f}%"
)

"""
On average, roughly 63.2% of the original data points of the original dataset are present in a given bootstrap sample.
Since the bootstrap sample has the same size as the original dataset there are many samples
that are in the bootstrap sample multiple times

Using bootstrap we are able to generate many datasets, all slightly different. We can fit a decision tree for each of
these datasets and they all shall be slightly different as well
"""

bag_of_trees = []
for bootstrap_idx in range(n_bootstraps):
	dtr = DecisionTreeRegressor(max_depth=3, random_state=0)
	data_bootstrap_sample, target_bootstrap_sample = bootstrap_sample(data_train, target_train, seed=bootstrap_idx)
	dtr.fit(data_bootstrap_sample, target_bootstrap_sample)
	bag_of_trees.append(dtr)

"""
Now that we created a bag of different trees, we can use each of the trees to predict the samples
withing the range of data. They shall give slightly different predicitons
"""

sns.scatterplot(
	x=data_train["Feature"],
	y=target_train,
	color="black",
	alpha=0.5
)
for tree_idx, tree in enumerate(bag_of_trees):
	tree_predictions = tree.predict(data_test)
	plt.plot(
		data_test["Feature"],
		tree_predictions,
		linestyle="--",
		alpha=0.8,
		label=f"Tree #{tree_idx} predictions"
	)

plt.legend()
plt.title("Predictions of trees trained on different bootstraps")
plt.show()

"""
Aggregating

Once our trees are fitted, we are able to get predictions from each of them. In regression, themost straightforward way
to combine those predictions is just to average them: for a given test data point, we feed the input feature values
to each of the n trained models in the ensemble and as a result compute n predicted values for the target variable. The
final prediciton of the ensemble for th etest data point is the average of those n values.
"""

sns.scatterplot(
	x=data_train["Feature"], y=target_train, color="black", alpha=0.5
)

bag_predictions = []
for tree_idx, tree in enumerate(bag_of_trees):
	tree_predictions = tree.predict(data_test)
	plt.plot(
		data_test["Feature"],
		tree_predictions,
		linestyle="--",
		alpha=0.8,
		label=f"Tree #{tree_idx} predictions"
	)
	bag_predictions.append(tree_predictions)

bag_predictions = np.mean(bag_predictions, axis=0)

plt.plot(
	data_test["Feature"],
	bag_predictions,
	label="Averaged predictions",
	linestyle="-"
)
plt.legend(loc="upper left")
plt.title("Predictions of bagged trees")
plt.show()

"""
The continuous red line shows the averaged predictions, which would be the final predictions
given by our "bag" of decision tree regressors. Note that the predicitons of the ensemble is more
stable because of the averaging operation. As a result, the bag of trees as a whole is less likely to overfit
than the individual trees
"""

"""
Bagging in scikit-learn

scikit-learn implements the bagging procedure as a meta-estimator, that is, an estimator that wraps another estimator:
it takes a base model that is clones several times and trained independently on each bootstrap sample


"""
# We set n_estimators=100 instead of 3 in our manual implementation above to get a stronger
# smoothing effect
bagged_trees = BaggingRegressor(
	estimator=DecisionTreeRegressor(max_depth=3),
	n_estimators=100
)

bagged_trees.fit(data_train, target_train)
sns.scatterplot(
	x=data_train["Feature"],
	y=target_train,
	color="black",
	alpha=0.5
)
bagged_trees_predictions = bagged_trees.predict(data_test)
plt.plot(
	data_test["Feature"],
	bagged_trees_predictions,
)
plt.title("Predictions from a bagging regressor")
plt.show()

"""
Because we use 100 trees in the ensemble, the average prediction is indeed slightlty smoother but very
similar to our previous avergae plot

It is possible to access the internal models of the ensemble sotres as a python list in the 
bagged_trees.estimators_ attribute after fitting.
"""

"""
Lets compare the based model prediciton with their average
"""

for tree_idx, tree in enumerate(bagged_trees.estimators_):
	label = "Predictions of individual trees" if tree_idx == 0 else None

	tree_predictions = tree.predict(data_test.to_numpy())
	plt.plot(
		data_test["Feature"],
		tree_predictions,
		linestyle="--",
		alpha=0.1,
		color="tab:blue",
		label=label
	)
sns.scatterplot(
	x=data_train["Feature"],
	y=target_train,
	color="black",
	alpha=0.5
)

bagged_trees_predictions = bagged_trees.predict(data_test)

plt.plot(
	data_test["Feature"],
	bagged_trees_predictions,
	color="tab:orange",
	label="Predictions of ensemble"
)
plt.legend()
plt.show()

"""
Bagging complex pipelines

Even if here we used a decision tree as a base model, nothing prevents us from using any other
type of model

As we know that the original data generation function is noisy polynomial transformation of the input
variable, let us try to fit a bagged polynomial regression pipeline on this dataset
"""

polynomial_regressor = make_pipeline(
	MinMaxScaler(),
	PolynomialFeatures(degree=4, include_bias=False),
	Ridge(alpha=1e-10)
)

"""
The ensemble itself is simply built by passing the resulting pipeline as the estimator parameter
of the BaggingRegressor class
"""

baggin = BaggingRegressor(
	estimator=polynomial_regressor,
	n_estimators=100,
	random_state=0
)
baggin.fit(data_train, target_train)

for i, regressor in enumerate(baggin.estimators_):
	regressor_predictions = regressor.predict(data_test.to_numpy())
	plt.plot(
		data_test["Feature"],
		regressor_predictions,
		linestyle="--",
		alpha=0.2,
		label="Predictions of base models" if i == 0 else None,
		color="tab:blue"
	)
sns.scatterplot(
	x=data_train["Feature"],
	y=target_train,
	color="black",
	alpha=0.5
)
bag_predictions = baggin.predict(data_test)

plt.plot(
	data_test["Feature"],
	bag_predictions,
	color="tab:orange",
	label="predictions of ensemble"
)
plt.ylim(target_train.min(), target_train.max())
plt.legend()
plt.title("Bagged polynomial regression")
plt.show()

"""
The predictions of this bagged polynomial regression model look qualitatively better than
the bagged trees. This is somewhat expected since the base model better reflects our knowledge of the
true data generating process
"""