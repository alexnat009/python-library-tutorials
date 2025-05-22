import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, validation_curve
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("../datasets/penguins.csv")

feature_names = [
	"Culmen Length (mm)",
	"Culmen Depth (mm)",
	"Flipper Length (mm)",
]
target_name = "Body Mass (g)"

df = df[feature_names + [target_name]].dropna(axis="rows", how="any")
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
X, y = df[feature_names], df[target_name]

dtr = DecisionTreeRegressor(random_state=0)
rfr = RandomForestRegressor(random_state=0)

cv_results_dtr = cross_validate(dtr, X, y, cv=10, return_train_score=True, n_jobs=2)
cv_results_rfr = cross_validate(rfr, X, y, cv=10, return_train_score=True, n_jobs=2)

print(np.sum(cv_results_rfr["test_score"] > cv_results_dtr["test_score"]))
print(sum(f > t for f, t in zip(cv_results_rfr["test_score"], cv_results_dtr["test_score"])))

rfr_5 = RandomForestRegressor(n_estimators=5, random_state=0)
rfr_100 = RandomForestRegressor(n_estimators=100, random_state=0)
cv_results_rfr_5 = cross_validate(rfr_5, X, y, cv=10, return_train_score=True, n_jobs=2)
cv_results_rfr_100 = cross_validate(rfr_100, X, y, cv=10, return_train_score=True, n_jobs=2)

print(np.sum(cv_results_rfr_100["test_score"] > cv_results_rfr_5["test_score"]))
print(sum(f > t for f, t in zip(cv_results_rfr_100["test_score"], cv_results_rfr_5["test_score"])))

n_estimators = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])


def plot_validation_curve(model, ax, linestyle, label, param_name, param_range):
	train_scores, test_scores = validation_curve(
		model,
		X,
		y,
		param_name=param_name,
		param_range=param_range,
		scoring="r2",
		n_jobs=2,
		cv=5,
	)
	train_mean = train_scores.mean(axis=1)
	train_std = train_scores.std(axis=1)
	test_mean = test_scores.mean(axis=1)
	test_std = test_scores.std(axis=1)

	ax.plot(param_range, train_mean, label=f"Train {label}", linestyle=linestyle)
	ax.fill_between(
		param_range, train_mean - train_std, train_mean + train_std, alpha=0.2
	)
	ax.plot(param_range, test_mean, label=f"Test {label}", linestyle=linestyle, marker='o')
	ax.fill_between(
		param_range, test_mean - test_std, test_mean + test_std, alpha=0.2
	)
	ax.set_xscale("log")
	ax.set_xlabel(param_name)
	ax.set_ylabel("R2 Score")
	ax.legend()
	ax.grid(True)


fig, ax = plt.subplots(figsize=(10, 6))
plot_validation_curve(
	RandomForestRegressor(max_depth=None, random_state=0),
	ax=ax,
	linestyle='-',
	label="max_depth=None (full depth)",
	param_name="n_estimators",
	param_range=n_estimators
)
plot_validation_curve(
	RandomForestRegressor(max_depth=5, random_state=0),
	ax=ax,
	linestyle='--',
	label="max_depth=5 (limited depth)",
	param_name="n_estimators",
	param_range=n_estimators)

plt.title("Validation Curve of Random Forest: Varying Number of Trees")
plt.show()

rf_1_tree = RandomForestRegressor(n_estimators=1, random_state=0)
cv_results_tree = cross_validate(
	rf_1_tree, X, y, cv=10, return_train_score=True
)
print(cv_results_tree["train_score"])

tree = DecisionTreeRegressor(random_state=0)
cv_results_tree = cross_validate(
	tree, X, y, cv=10, return_train_score=True
)
print(cv_results_tree["train_score"])

max_iter = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
fig, ax = plt.subplots(figsize=(10, 6))
plot_validation_curve(
	HistGradientBoostingRegressor(random_state=0),
	ax=ax,
	linestyle='--',
	label="Histogram Boosting Regressor",
	param_name="max_iter",
	param_range=max_iter
)
plt.show()
