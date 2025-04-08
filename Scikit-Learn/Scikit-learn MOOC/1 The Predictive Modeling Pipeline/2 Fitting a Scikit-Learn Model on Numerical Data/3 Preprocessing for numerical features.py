import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

df = pd.read_csv("../../../datasets/adult-census.csv")

df = df.drop(columns="education-num")
X = df.drop(columns='class')
y = df["class"]
numerical_columns = X.select_dtypes("number")

X_train, X_test, y_train, y_test = train_test_split(numerical_columns, y, random_state=42, test_size=0.2)
print(X_train.describe())

scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(X_train)
print(scaler.mean_)
print(scaler.scale_)

"""
scikit-learn convention: if an attribute is learned from the data, its name ends with an underscore (i.e. _),
as in mean_ and scale_ for the StandardScaler.
"""
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.describe())

# Or equivalently one can use shorthand
# X_train_scaled = scaler.fit_transform(X_train)

"""
Notice that the mean of all the columns is close to 0 and the standard deviation in all cases is close to 1. We can
also visualize the effect of StandardSca1er using jointplot to show both the histograms of the distributions
and a scatterplot of any pair of numerical features at the same time. We can observe that StandardSca1er does
not change the structure of the data itself but the axes get shifted and scaled.
"""

num_points_to_plot = 300
sns.jointplot(
	data=X_train[:num_points_to_plot],
	x="age",
	y="hours-per-week",
	marginal_kws=dict(bins=15)
)
plt.suptitle(
	"Jointplot of 'age' vs 'hours-per-week' \nbefore StandardScaler", fontsize=16
)

sns.jointplot(
	data=X_train_scaled[:num_points_to_plot],
	x="age",
	y="hours-per-week",
	marginal_kws=dict(bins=15),
)

plt.suptitle(
	"Jointplot of 'age' vs 'hours-per-week' \nafter StandardScaler", fontsize=16
)
plt.show()

model = make_pipeline(StandardScaler(), LogisticRegression())
print(model)
start = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - start
score = model.score(X_test, y_test)
model_name = model.__class__.__name__
print(
	f"The accuracy using a {model_name} is {score:.3f} "
	f"with a fitting time of {elapsed_time:.3f} seconds "
	f"in {model[-1].n_iter_[0]} iterations"
)

"""
We could compare this predictive model with the predictive model used in the previous notebook which did not scale features.
"""
model = LogisticRegression()
start = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - start
model_name = model.__class__.__name__
score = model.score(X_test, y_test)
print(
	f"The accuracy using a {model_name} is {score:.3f} "
	f"with a fitting time of {elapsed_time:.3f} seconds "
	f"in {model.n_iter_[0]} iterations"
)

"""
We see that scaling the data before training the logistic regression was beneficial in terms of computational
performance. Indeed, the number of iterations decreased as well as the training time. The generalization
performance did not change since both models converged.
"""

