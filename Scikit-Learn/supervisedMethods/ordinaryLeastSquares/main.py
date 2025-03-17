import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib

matplotlib.use('TkAgg')
np.random.seed(0)

X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.rand(100, 1)

plt.scatter(X, y, s=1)
plt.show()

X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b.dot(theta_best)
print(y_pred)

plt.plot(X_new, y_pred, 'r-')
plt.plot(X, y, 'b.')
plt.show()


def example():
	diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
	print(diabetes_X, diabetes_y)

	diabetes_X = diabetes_X[:, np.newaxis, 2]

	X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, random_state=42, test_size=0.2)

	reg = LinearRegression()

	reg.fit(X_train, y_train)

	y_pred = reg.predict(X_test)
	print(f'Coefficients: {reg.coef_}')
	print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
	print(f'R^2 error: {r2_score(y_test, y_pred)}')

	plt.scatter(X_test, y_test, color="black")
	plt.plot(X_test, y_pred, color="blue", linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.show()
