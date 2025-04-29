import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../../datasets/penguins_regression.csv")

feature_name = "Flipper Length (mm)"

target_name = "Body Mass (g)"

X, y = df[[feature_name]].to_numpy().flatten(), df[target_name].to_numpy()


def linear_model_flipper_mass(flipper_length, weight_flipper_length, intercept_body_mass):
	return flipper_length * weight_flipper_length + intercept_body_mass


def goodness_fit_measure_mse(true_values, predictions):
	return np.mean((true_values - predictions) ** 2)


def goodness_fit_measure_mae(true_values, predictions):
	return np.mean(np.abs(true_values - predictions))


weights = [-40, 45, 90]
intercepts = [15000, -5000, -14000]

param_grid = list(zip(weights, intercepts))
# same as list(product(weights, intercepts))
# param_grid = np.transpose([np.tile(weights, len(intercepts)), np.repeat(intercepts, len(weights))])


flipper_length_range = np.linspace(X.min(), X.max(), len(X))
ax = sns.scatterplot(data=df, x=feature_name, y=target_name, color="black", alpha=0.5)
mses = []
for weight, intercept in param_grid:
	predicted_body_mass = linear_model_flipper_mass(flipper_length_range, weight, intercept)
	error = goodness_fit_measure_mse(y, predicted_body_mass)
	mses.append(error)
	print(f'for weight={weight} and intercept={intercept}, error is {error}')
	ax.plot(flipper_length_range, predicted_body_mass, label=f"w={weight}, b={intercept}")
ax.legend()
plt.title("Penguins: Body Mass vs Flipper Length")
plt.show()

# Best result
best_idx = np.argmin(mses)
best_params = param_grid[best_idx]
print(f"Best MSE = {mses[best_idx]:.2f} with weight={best_params[0]}, intercept={best_params[1]}")
