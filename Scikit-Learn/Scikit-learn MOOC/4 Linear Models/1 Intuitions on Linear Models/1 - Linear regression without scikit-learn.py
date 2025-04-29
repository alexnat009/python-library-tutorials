import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../../datasets/penguins_regression.csv")
print(df)

feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"

X, y = df[[feature_name]], df[target_name]

ax = sns.scatterplot(data=df, x=feature_name, y=target_name, color="black", alpha=0.5)
ax.set_title("Body Mass as a function of the Flipper Length")
plt.show()

"""
In this problem, penguin mass is out target. It is a continuous variable that roughly varies between 2700g
and 6300g. Thus, this is a regression problem. We also see that there is almost a linear relationship between the
body mass of the penguin and its flipper length.

Thus we could come up with a simple formula, where given a flipper length we could compute the 
body mass of a penguin using a linear relationship of the form 'y = a * x + b' where 'a' and 'b' 
are the 2 parameters of our model
"""


def linear_model_flipper_mass(flipper_length, weight_flipper_length, intercept_body_mass):
	return weight_flipper_length * flipper_length + intercept_body_mass


"""
Using this model, we can check the body mass values predicted for a range of flipper lengths.
We set 'weight_flipper_length' and 'intercept_body_mass' to arbitrary values of 45 and -5000, respectively.
"""

weight_flipper_length = 45
intercept_body_mass = -5000

flipper_length_range = np.linspace(X.min(), X.max(), 300)
predicted_body_mass = linear_model_flipper_mass(flipper_length_range, weight_flipper_length, intercept_body_mass)

"""
We can now plot all samples and the linear model prediciton
"""

label = "{0:.2f} (g / mm) * flipper length + {1:.2f} (g)"

ax = sns.scatterplot(data=df, x=feature_name, y=target_name, color="black", alpha=0.5)
ax.plot(flipper_length_range, predicted_body_mass)
ax.set_title(label.format(weight_flipper_length, intercept_body_mass))
plt.show()

body_mass_180 = linear_model_flipper_mass(
	flipper_length=180, weight_flipper_length=40, intercept_body_mass=0
)
body_mass_181 = linear_model_flipper_mass(
	flipper_length=181, weight_flipper_length=40, intercept_body_mass=0
)

print(
	"The body mass for a flipper length of 180 mm "
	f"is {body_mass_180} g and {body_mass_181} g "
	"for a flipper length of 181 mm"
)
