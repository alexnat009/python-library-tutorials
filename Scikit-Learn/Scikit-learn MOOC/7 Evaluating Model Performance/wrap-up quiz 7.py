import matplotlib.axes._axes as axes
import matplotlib.figure as figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit, cross_validate, LearningCurveDisplay, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cycling = pd.read_csv("../datasets/bike_rides.csv", index_col=0,
					  parse_dates=True)
cycling.index.name = ""
target_name = "power"
data, target = cycling.drop(columns=target_name), cycling[target_name]

print(data)
print(target)

speed_cubic = data["speed"] ** 3
speed_cubic.name = "speed_cubic"
speed = data["speed"]
speed.name = "speed"
speed_angle = data["speed"] * np.arctan(data["slope"])
speed_angle.name = "speed_angle"
speed_accel = data["speed"] * data["acceleration"]
speed_accel.name = "speed_accel"
df = pd.concat(
	[speed_cubic, speed, speed_angle, speed_accel],
	axis=1,
)
df["speed_accel"] = df["speed_accel"].clip(lower=0)
print(df)

model = make_pipeline(StandardScaler(), RidgeCV())
cv = ShuffleSplit(n_splits=4, random_state=0)
cv_results_lr = cross_validate(
	model,
	df,
	target,
	cv=cv,
	scoring="neg_mean_absolute_error",
	return_estimator=True,
	return_train_score=True
)
for estimator in cv_results_lr["estimator"]:
	ridge_model = estimator.named_steps["ridgecv"]
	print(ridge_model.coef_)
print(df["speed_angle"].mean())
print(-cv_results_lr["test_score"].mean())

print("Linear Model Train MAE:", -cv_results_lr["train_score"])
print("Linear Model Test MAE:", -cv_results_lr["test_score"])

hgbr = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True)
cv_results_hgbr = cross_validate(
	hgbr,
	data,
	target,
	cv=cv,
	scoring="neg_mean_absolute_error",
	return_estimator=True,
	return_train_score=True
)

print(-cv_results_hgbr["test_score"].mean())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))  # type:figure.Figure, axes.Axes

LearningCurveDisplay.from_estimator(
	model,
	data,
	target,
	cv=cv,
	scoring="neg_mean_absolute_error",
	n_jobs=2,
	ax=ax[0],
	score_type="both",
	negate_score=True,
	std_display_style="errorbar",
)

LearningCurveDisplay.from_estimator(
	hgbr,
	data,
	target,
	cv=cv,
	scoring="neg_mean_absolute_error",
	n_jobs=2,
	ax=ax[1],
	score_type="both",
	negate_score=True,
	std_display_style="errorbar",
)

# plt.show()

print("Boosting Model Train MAE:", -cv_results_hgbr["train_score"])
print("Boosting Model Test MAE:", -cv_results_hgbr["test_score"])

print(np.unique(data.index.date))

groups, _ = pd.factorize(data.index.date)

cv = LeaveOneGroupOut()

cv_results_hgbr_groups = cross_validate(
	hgbr,
	data,
	target,
	cv=cv,
	groups=groups,
	return_estimator=True,
	return_train_score=True,
	scoring="neg_mean_absolute_error"
)
cv_results_lr_groups = cross_validate(
	model,
	df,
	target,
	cv=cv,
	groups=groups,
	return_estimator=True,
	return_train_score=True,
	scoring="neg_mean_absolute_error"
)
for name, results in [
	("Linear model with LOGO", cv_results_lr_groups),
	("Histogram GBDT with LOGO", cv_results_hgbr_groups),
	("Linear model with ShuffleSplit", cv_results_lr),
	("Histogram GBDT with ShuffleSplit", cv_results_hgbr),
]:
	for split in ["train", "test"]:
		errors = -results[f"{split}_score"]
		print(f"{name} - MAE on {split} sets:\t"
			  f"{errors.mean():.3f} +/- {errors.std():.3f} Watts")

cv = LeaveOneGroupOut()
train_indices, test_indices = list(cv.split(data, target, groups=groups))[0]

data_linear_model_train = df.iloc[train_indices]
data_linear_model_test = df.iloc[test_indices]

data_train = data.iloc[train_indices]
data_test = data.iloc[test_indices]

target_train = target.iloc[train_indices]
target_test = target.iloc[test_indices]

model.fit(data_linear_model_train, target_train)
hgbr.fit(data_train, target_train)

linear_model_pred = model.predict(data_linear_model_test)
hgbr_pred = hgbr.predict(data_test)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 6), layout='constrained')  # type:figure.Figure, axes.Axes


def scatter_points(y_test, y_pred, title, axes):
	axes.scatter(y_test, y_pred, alpha=0.7)
	axes.plot([y_test.min(), y_test.max()],
			  [y_test.min(), y_test.max()], 'r--')
	axes.set_title(title)
	axes.set_xlabel("True Power")
	axes.set_ylabel("Predicted Power")


scatter_points(target_test, linear_model_pred, "Linear Model - Single Ride", ax[0])
scatter_points(target_test, hgbr_pred, "Boosting Model - Single Ride", ax[1])
plt.show()

time_slice = slice("2020-08-18 17:00:00", "2020-08-18 17:05:00")
print(data_linear_model_test)
data_test_linear_model_subset = data_linear_model_test[time_slice]
data_test_subset = data_test[time_slice]
target_test_subset = target_test[time_slice]

linear_model_pred = model.predict(data_test_linear_model_subset)
hgbr_pred = hgbr.predict(data_test_subset)
print(model.score(data_test_linear_model_subset, target_test_subset))
print(hgbr.score(data_test_subset, target_test_subset))
plt.figure(figsize=(12, 6))
plt.plot(target_test_subset.index, target_test_subset, label="True Power", color="black")
plt.plot(target_test_subset.index, linear_model_pred, label="Linear Model Prediction", color="blue")
plt.plot(target_test_subset.index, hgbr_pred, label="Boosting Model Prediction", color="green")

plt.xlabel("Time")
plt.ylabel("Power (Watts)")
plt.title("Power Prediction Between 17:00 and 17:05")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
