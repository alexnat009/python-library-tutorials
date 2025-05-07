import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, ValidationCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("../../datasets/penguins.csv")
columns = ["Flipper Length (mm)", "Culmen Length (mm)", "Culmen Depth (mm)"]
target_name = "Body Mass (g)"

df_non_missing = df[columns + [target_name]].dropna()

data = df_non_missing[columns]
target = df_non_missing[target_name]

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42, test_size=0.2)

lr = LinearRegression()
# lr.fit(X_train, y_train)
# scores = lr.score(X_test, y_test)
# print(scores)
cv_results = cross_validate(lr, data, target, scoring="neg_mean_absolute_error", cv=10, n_jobs=2)
print(
	"Mean absolute error on testing set with original features: "
	f"{-cv_results['test_score'].mean():.3f} ± "
	f"{cv_results['test_score'].std():.3f} g"
)
model = make_pipeline(
	PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
	LinearRegression()
).set_output(transform="pandas")

model.fit(data, target)
few_data = model[0].transform(data[:5])
print(few_data['Flipper Length (mm)'] * few_data['Culmen Length (mm)'] == few_data[
	'Flipper Length (mm) Culmen Length (mm)'])

cv_results_with_features_engineering = cross_validate(model, data, target, cv=10, scoring="neg_mean_absolute_error")
print(
	"Mean absolute error on testing set with original features: "
	f"{-cv_results_with_features_engineering['test_score'].mean():.3f} ± "
	f"{cv_results_with_features_engineering['test_score'].std():.3f} g"
)

nystroem_regression = make_pipeline(
	Nystroem(kernel="poly", degree=2, random_state=0),
	lr
)
param_range = np.array([5, 10, 50, 100])
disp = ValidationCurveDisplay.from_estimator(
	nystroem_regression,
	data,
	target,
	param_name="nystroem__n_components",
	param_range=param_range,
	cv=10,
	scoring="neg_mean_absolute_error",
	negate_score=True,
	std_display_style="errorbar",
	n_jobs=2

)
disp.ax_.set(
	xlabel="Number of components",
	ylabel="Mean absolute error (g)",
	title="Validation curve for Nystroem regression",
)
plt.show()
params = {"nystroem__n_components": 10}
nystroem_regression.set_params(**params)
cv_results = cross_validate(
	nystroem_regression,
	data,
	target,
	cv=10,
	scoring="neg_mean_absolute_error",
	n_jobs=2,
)
print(
	"Mean absolute error on testing set with nystroem: "
	f"{-cv_results['test_score'].mean():.3f} ± "
	f"{cv_results['test_score'].std():.3f} g"
)

"""
If we had p = 10 original features (instead of 3), the PolynomialFeatures transformer
would have generated 100 (100 - 1) / 2 = 4950 additional interaction features
(so we would have 5050 features in total). The resulting pipeline would have been much slower to train
and predict. Furthermore, the large number of interaction features would probably have resulted in
an overfitting model


On the other hand, the Nystroem transformer generates a user-adjustable number of features (n_components).
Furthermore, the optimal number of components is usually much smaller that that. So the Nystroem transformer
can be more scalable when the number of original features is too large for PolynomialFeatures to be used

The main downside of the Nystroem transformer is that it's not possible to easily interpret the meaning
of the generated features and therefore the meaning of the learned coefficients for the downstream linear model  
"""
