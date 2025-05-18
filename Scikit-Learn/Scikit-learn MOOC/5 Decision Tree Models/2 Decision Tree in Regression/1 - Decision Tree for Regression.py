import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree

df = pd.read_csv("../../datasets/penguins_regression.csv")

feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
X_train, y_train = df[[feature_name]], df[target_name]

"""
To illustrate how decision trees predict in a regression setting, we create a synthetic dataset
containing some of the possible flipper length values between the minimum and the maximum of the original data
"""
X_test = pd.DataFrame(
	np.arange(X_train[feature_name].min(), X_train[feature_name].max()),
	columns=[feature_name]
)

"""
Using the term "test" here refers to data that was not used for training. It should not be
confused with data coming from a train-test split, as it was generated in equally-spaced intervals for the 
visual evaluation of the predictions.

Note that this is methodologically valid here because our objective is to get some intuitive understanding on the shape 
of the decision function of the learned decision trees.

However, computing an evaluation metric on such a synthetic test set would be meaningless since the synthetic dataset
doesn't follow the same distribution as the real world data on which the model would be deployed
"""


def plot_points():
	sns.scatterplot(
		data=df,
		x=feature_name,
		y=target_name,
		color="black",
		alpha=0.5
	)


plot_points()
plt.title("Illustration of the regression dataset used")
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

plot_points()

plt.plot(X_test[feature_name], y_pred, label="Linear Regression")
plt.legend()
plt.title("Prediction function using a LinearRegression")
plt.show()

ax = sns.scatterplot(
	data=df,
	x=feature_name,
	y=target_name,
	color="black",
	alpha=0.5
)
plt.plot(
	X_test[feature_name],
	y_pred,
	label="Linear Regression",
	linestyle="--"
)
plt.scatter(
	X_test[::3],
	y_pred[::3],
	label="Predictions",
	color="tab:orange"
)
plt.legend()
plt.title("Prediction function using a LinearRegression")
plt.show()

dtr = DecisionTreeRegressor(max_depth=1)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

plot_points()
plt.plot(X_test[feature_name], y_pred, label="Decision tree")
plt.legend()
plt.title("Prediction function using a DecisionTreeRegressor")
plt.show()

"""
We see that the decision tree model doesn't have an a priori distribution for the data and we 
don't end up with a straight line to regress flipper length and body mass

Instead, we observe that the predictions of the tree are piecewise constant. Indeed, our feature
space was split into two partitions. Let's check the tree structure to see what was the threshold found
during the training
"""
_, ax = plt.subplots(figsize=(8, 6))
plot_tree(
	dtr,
	feature_names=[feature_name],
	ax=ax,
	filled=True,
)
plt.show()

"""
The threshold for our feature (flipper length) is 206.5mm. The predicted values on each side of the split
are two constants: 3698.71g amd 5032.36g. These values correspond to the mean values of the training samples in each
partition.

In classification, we saw that increasing the depth of the tree allowed us to get more complex decision boundaries.
"""
dtr = DecisionTreeRegressor(max_depth=3)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

plot_points()
plt.plot(X_test[feature_name], y_pred, color="black", alpha=0.5, label="Decision tree")
plt.legend()
plt.title("Prediction function using a DecisionTreeRegressor")
plt.show()

"""
Increasing the depth of the tree increases the number of partitions and thus the number of constant
values that the is capable of prediction.
"""
