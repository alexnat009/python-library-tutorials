import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("../../datasets/penguins_regression.csv")

feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
X_train, y_train = df[[feature_name]], df[target_name]

lr = LinearRegression()
lr.fit(X_train, y_train)

dtr = DecisionTreeRegressor(max_depth=3)
dtr.fit(X_train, y_train)

X_test = pd.DataFrame(
	np.arange(X_train[feature_name].min(), X_train[feature_name].max()),
	columns=[feature_name],
)

y_pred_lr = lr.predict(X_test)
y_pred_dtr = dtr.predict(X_test)

sns.scatterplot(
	data=df,
	x=feature_name,
	y=target_name,
	color="black",
	alpha=0.5
)
plt.plot(X_test[feature_name], y_pred_lr, label="Linear Regression")
plt.plot(X_test[feature_name], y_pred_dtr, label="Decision Tree Regression")
plt.legend()
plt.show()

X_test_augmented = pd.DataFrame(
	np.arange(X_test[feature_name].min() - 70, X_test[feature_name].max() + 70),
	columns=[feature_name]
)
y_pred_lr = lr.predict(X_test_augmented)
y_pred_dtr = dtr.predict(X_test_augmented)

sns.scatterplot(
	data=df,
	x=feature_name,
	y=target_name,
	color="black",
	alpha=0.5
)
plt.plot(X_test_augmented[feature_name], y_pred_lr, label="Linear Regression")
plt.plot(X_test_augmented[feature_name], y_pred_dtr, label="Decision Tree Regression")
plt.legend()
plt.show()

"""
The linear model extrapolates using the fitted model for flipper lengths < 175 and > 235mm. In fact we are using
the model parametrization to make these predictions.

As mentioned, decision trees are non-parametric models and we observe that they cannot extrapolate. For flipper lengths
below the minimum, the mass of the penguin in the training data with the shortest flipper length will always be 
predicted. Similarly, for flipper lengths above the maximum, the mass of the penguin in the training data with the
longest flipper will always be predicted.
"""
