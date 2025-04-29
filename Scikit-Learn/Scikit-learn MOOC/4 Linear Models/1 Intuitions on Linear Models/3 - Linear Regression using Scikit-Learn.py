from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../../datasets/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
X, y = df[[feature_name]], df[target_name]

lr = LinearRegression()
lr.fit(X, y)

weight_flipper_length, intercept_body_mass = lr.coef_[0], lr.intercept_
print(weight_flipper_length, intercept_body_mass)

flipper_weight_range = np.linspace(X.min(), X.max(), len(X))
predicted_body_mass = flipper_weight_range * weight_flipper_length + intercept_body_mass

ax = sns.scatterplot(data=df, x=feature_name, y=target_name, color="black", alpha=0.5)
ax.plot(flipper_weight_range, predicted_body_mass)
plt.show()

inferred_body_mass = lr.predict(X)
model_error_mse = mse(y, inferred_body_mass)
model_error_mae = mae(y, inferred_body_mass)
print(
	f"The mean squared error of the optimal model is {model_error_mse:.2f}\n"
	f"The mean squared error of the optimal model is {model_error_mae:.2f}")
