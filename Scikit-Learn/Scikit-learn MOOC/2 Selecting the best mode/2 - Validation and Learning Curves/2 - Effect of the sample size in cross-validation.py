from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import ShuffleSplit, LearningCurveDisplay, cross_validate
import matplotlib.pyplot as plt

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
y *= 100  # rescale the target in k$

"""
To understand the impact of the number of samples available for training on the 
generalization performance of a predictive model, it is possible to synthetically reduce 
the number of samples used to train the predictive model and check the training and testing errors

Therefore, we can vary the number of samples in the training set and repeat the experiment.
The training and testing scores can be plotted similarly to the validation curve, but instead of 
varying a hyperparameter, we vary the number of training samples. This curve is called the learning curve

It gives information regarding the benefit of adding new training samples to improve a model's generalization
performance

"""

train_size = np.linspace(0.1, 1.0, num=5, endpoint=True)
print(train_size)
cv = ShuffleSplit(n_splits=30, test_size=0.2)
regressor = DecisionTreeRegressor()
display = LearningCurveDisplay.from_estimator(
	regressor,
	X,
	y,
	train_sizes=train_size,
	cv=cv,
	score_type="both",
	scoring="neg_mean_absolute_error",
	negate_score=True,
	score_name="Mean Absolute Error (k$)",
	std_display_style="errorbar",
	n_jobs=3
)
display.ax_.set(xscale="log", title="Learning curve for decision tree")
plt.show()

"""
Looking at the training error alone:
we see that we get an error of 0 k$. It means that the trained model is clearly overfitting the training data

Looking at the testing error alone:
we observer that the more samples are added into the training set, the lower the testing error becomes.
Also we are searching for the plateau of the testing error for which there is no benefit to adding samples anymore or 
assessing the potential gain of adding more samples into the training set

It the testing error plateaus despite adding more training samples, it's possible that the model has achieved
its optimal performance. In this case, using a more expressive model might help reduce the error further.
Otherwise, the error may have reached the Bayes error rate, the theoretical minimum error due to 
inherent uncertainty not resolved by the available data. This minimum error is non-zero whenever some of the variation 
of the target variable y depends on external factors not fully observed in the features available in X, which is almost
always the case in practice
"""