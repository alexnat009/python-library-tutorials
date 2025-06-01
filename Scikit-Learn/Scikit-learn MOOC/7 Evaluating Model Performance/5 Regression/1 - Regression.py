import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, \
	mean_absolute_percentage_error, PredictionErrorDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("../../datasets/house_prices.csv")
X = df.drop(columns="SalePrice")
y = df["SalePrice"]
X = X.select_dtypes(np.number)
y /= 1000

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)

"""
Some ML models are designed to be solved as an optimization problem:
minimizing an error using a training set. A basic loss function used in regression is the mean
squared error (MSE). Thus, this metric is sometimes used to evaluate the model since it is optimized
by said model

"""

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
print(
	"Mean squared error on the training set: "
	f"{mean_squared_error(y_train, y_pred_train):.3f}"
)
"""
Our linear regression model is minimizing the mean squared error on the training set.
It means that there is no other set of coefficient which decreases the error

Then we can compute the mean squared error on the test set
"""

y_pred = lr.predict(X_test)
print(
	"Mean squared error on the training set: "
	f"{mean_squared_error(y_test, y_pred):.3f}"
)

"""
The raw MSE can be difficult to interpret. One way is to rescale the MSE by the variance of the target
This score is known as the R^2 also called the coefficient of determination. Indeed, this is the default
score used in scikit-learn by calling the method score
"""

lr.score(X_test, y_test)

"""
The R^2 score represents the proportion of variance of the target that is explained by the independent variables
in the model. The best score possible is 1 but there is no lower bound. However, a model that predicts the 
expected value of the target would ge a score of 0
"""

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
print(
	"R2 score for a regressor predicting the mean:"
	f"{dummy.score(X_test, y_test):.3f}"
)
"""
The R^2 score gies insight into the quality of the model's fit. However this score cannot be 
compared from one dataset to another and the value obtained doesn't have a meaningful interpretation relative
the original unit of the target. If we wanted to get an interpretable score, we would be interested in the median
or mean absolute error 
"""
print(
	"Mean absolute error: "
	f"{mean_absolute_error(y_test, y_pred):.3f} k$"
)

"""
By computing the mean absolute error, we can interpret that our model is predicting on average
22.6k$ away from the true house price. A disadvantage of this metric is that the mean can be impacted by large error
For some applications, we might not want these large error to have such a big influence on our metric. In this case we
can use the median absolute error 
"""

print(
	"Median absolute error: "
	f"{median_absolute_error(y_test, y_pred):.3f} k$"
)

"""
The mean absolute error (or median absolute error) still have a known limitation: committing 
an error of 50k$ for a house valued at 50$ has the same impact than committing an error of 50k$
for a house valued at 500k$. Indeed, the mean absolute error is not relative.

The mean absolute percentage error introduce this relative scaling 
"""

print(
	"Mean absolute percentage error: "
	f"{mean_absolute_percentage_error(y_test, y_pred) * 100:.3f} %"
)

"""
In addition to using metrics, we can visualize the results by plotting the predicted values
versus the true values.

In an ideal scenario where all variations in the target could be perfectly explained by the observed
features (without any unobserved factors of variations) and we have chosen an optimal model, we would expect
all prediction to fall along the diagonal line of the first plot below

In the real life, this is almost never the case: some unknown fraction of the variations in the
target cannot be explained by variations in data: they stem from external factors not represented by the
observed features

Therefore, the best we can hope for is that our model's predictions form a cloud points symmetrically
distributed around the diagonal line, ideally close enough to it for the model to be useful.

To gain more insight, it can be helpful to plot the residuals, which represent the difference between the actual
and predicted values. This is shown in the second plot

Residual plots make it easier to assess if the residuals exhibit a variance independent of the target values or if 
there is any systematic bias of the model associated with the lowest or highest predicted values
"""


def plot_residuals(y_test, y_pred):
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5), layout='constrained')  # type:figure.Figure, axes.Axes
	PredictionErrorDisplay.from_predictions(
		y_true=y_test,
		y_pred=y_pred,
		kind="actual_vs_predicted",
		scatter_kwargs={"alpha": 0.5},
		ax=ax[0]
	)

	ax[0].axis("square")
	ax[0].set_xlabel("Predicted values (k$)")
	ax[0].set_ylabel("True values (k$)")

	PredictionErrorDisplay.from_predictions(
		y_true=y_test,
		y_pred=y_pred,
		kind="residual_vs_predicted",
		scatter_kwargs={"alpha": 0.5},
		ax=ax[1],
	)
	ax[1].axis("square")
	ax[1].set_xlabel("Predicted values (k$)")
	ax[1].set_ylabel("Residual values (k$)")

	fig.suptitle(
		"Regression using a model\nwithout target transformation", y=1.1
	)
	plt.show()


plot_residuals(y_test, y_pred)
"""
On these plots, we see that our model tends to under-estimate the price of the house both for the lowest and large
True price values. This means that the residuals still hold some structure typically visible as the "banana" or "smile"
shape of the residual plot. This is often a clue that our model could be improved , either bu transforming the features,
the target or sometimes changing the model type or its parameters. In this case let's try to see 
if the model would benefit from a transformation that monotonically reshapes the target variable to follwo a 
normal distribution
"""

transformer = QuantileTransformer(n_quantiles=900, output_distribution="normal")

model_transformed_target = TransformedTargetRegressor(
	regressor=lr, transformer=transformer
)

model_transformed_target.fit(X_train, y_train)
y_pred = model_transformed_target.predict(X_test)
plot_residuals(y_test, y_pred)

"""
the model with the transformed target seems to exhibit fewer structure in its residuals: over-estimation and
under-estimation errors seem to be more balanced

We can confirm this by computing he previously mentioned metrics and observe that they all improved w.r.t the 
linear regression model without the target transformation
"""


print(
    "Mean absolute error: "
    f"{mean_absolute_error(y_test, y_pred):.3f} k$"
)
print(
    "Median absolute error: "
    f"{median_absolute_error(y_test, y_pred):.3f} k$"
)
print(
    "Mean absolute percentage error: "
    f"{mean_absolute_percentage_error(y_test, y_pred):.2%}"
)

"""
performing such transformation for linear regression is often disapproved by statisticians,
It is mathematically more justified to instead adapt the loss function of the regression model
itself, for instance by fitting a PoissonRegressor or a TweedieRegressor model instead of LinearRegression
"""