import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("../../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

X, y = df[culmen_columns], df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(X_train, y_train)

palette = ["tab:blue", "tab:green", "tab:orange"]
tab10_norm = mpl.colors.Normalize(vmin=-0.5, vmax=8.5)
disp = DecisionBoundaryDisplay.from_estimator(
	dtc,
	X_train,
	response_method="predict",
	cmap="tab10",
	norm=tab10_norm,
	alpha=0.5
)
sns.scatterplot(
	data=df,
	x=culmen_columns[0],
	y=culmen_columns[1],
	hue=target_column,
	palette=palette
)

_, ax = plt.subplots(figsize=(8, 6))
plot_tree(
	dtc,
	class_names=dtc.classes_.tolist(),
	feature_names=culmen_columns,
	impurity=True,
	ax=ax,
	filled=True,
	# node_ids=True,
	# proportion=True,
)
plt.show()

"""
The resulting tree has 7 nodes: 3 of them are "split nodes" and 4 are "leaf nodes", organized in 2 levels
We see that the second tree leve used the "Culmen Length" to make two new decisions. Qualitatively, we saw
that such a simple tree was enough to classify the penguins' species 
"""
test_score = dtc.score(X_test, y_test)
print(f" Accuracy of the DecisionTreeClassifier: {test_score:.2f}")

"""
Decision Tree is built by successively partitioning the feature space, considering one feature at a time

We predict an Adelie penguin if the feature value is below the threshold, which is not
surprising since this partition was almost pure. If the feature value is above the threshold, we predict
the Gentoo penguin, the class that is most probable
"""

# Predicted probabilites in multi-class problems (Estimated)

"""
One can further try to visualize, the output of predict+proba for a multiclass, problem using
DecisionBoundaryDisplay, except that for a K-class problem you have K probability outputs for each data points.
Visualizing all these on a single plot can quickly become tricky to interpret. It is then common to instead
produce K separate plots, one for each class, in a one-vs-rest fashion
"""

xx = np.linspace(30, 60, 100)
yy = np.linspace(10, 23, 100)
xx, yy = np.meshgrid(xx, yy)
Xfull = pd.DataFrame(
	{"Culmen Length (mm)": xx.ravel(), "Culmen Depth (mm)": yy.ravel()}
)

probas = dtc.predict_proba(Xfull)
n_classes = len(np.unique(dtc.classes_))
_, axs = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(12, 5))
plt.suptitle("Predicted probabilities for decision tree model", y=0.8)
imshow_handle = None
for class_of_interest in range(n_classes):
	axs[class_of_interest].set_title(
		f"Class {dtc.classes_[class_of_interest]}"
	)
	imshow_handle = axs[class_of_interest].imshow(
		probas[:, class_of_interest].reshape((100, 100)),
		extent=(30, 60, 10, 23),
		vmin=0.0,
		vmax=1.0,
		origin="lower",
		cmap="viridis"
	)
	axs[class_of_interest].set_xlabel("Culmen Length (mm)")
	if class_of_interest == 0:
		axs[class_of_interest].set_ylabel("Culmen Depth (mm)")
	idx = y_test == dtc.classes_[class_of_interest]
	axs[class_of_interest].scatter(
		X_test["Culmen Length (mm)"].loc[idx],
		X_test["Culmen Depth (mm)"].loc[idx],
		marker="o",
		c="w",
		edgecolor="k",
	)
ax = plt.axes((0.15, 0.04, 0.7, 0.05))
plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")
plt.title("Probability")
plt.show()

# Multi class of interest scikit-learn version +1.4
# n_classes = dtc.n_classes_
# classes = dtc.classes_
# print(n_classes, classes)
# fig, axs = plt.subplots(
# 	ncols=n_classes,
# 	figsize=(4 * n_classes, 5),
# 	sharey=True
# )  # type:figure.Figure, axes.Axes
# plt.suptitle("Predicted probabilities for decision tree model", y=0.92)
# for ax, cls in zip(axs, classes):
# 	display = DecisionBoundaryDisplay.from_estimator(
# 		dtc,
# 		Xfull,
# 		response_method="predict_proba",
# 		class_of_interest=cls,
# 		grid_resolution=100,
# 		xlabel="Culmen Length (mm)",
# 		ylabel="Culmen Depth (mm)",
# 		ax=ax,
# 		cmap="viridis",
# 		vmin=0.0,
# 		vmax=1.0,
# 	)
# 	imshow_handle = display.ax_.collections[0]
# 	ax.set_title(f"Class {cls}")
# 	mask = (y_test == cls)
# 	ax.scatter(
# 		X_test.loc[mask, "Culmen Length (mm)"],
# 		X_test.loc[mask, "Culmen Depth (mm)"],
# 		marker="o",
# 		c="w",
# 		edgecolor="k"
# 	)
#
# cbar_ax = fig.add_axes((0.15, 0.1, 0.7, 0.05))
# fig.colorbar(imshow_handle, cax=cbar_ax, orientation="horizontal")
# cbar_ax.set_title("Probability")
#
# fig.subplots_adjust(bottom=0.30, top=0.80),
# plt.show()

# Single class of interest
# DecisionBoundaryDisplay.from_estimator(
# 	dtc,
# 	X_train,
# 	response_method="predict_proba",
# 	class_of_interest="Gentoo",
# 	norm=tab10_norm,
# 	cmap="tab10"
# )
# sns.scatterplot(
# 	data=df,
# 	x=culmen_columns[0],
# 	y=culmen_columns[1],
# 	hue=target_column,
# 	palette=palette
# )
# plt.show()
