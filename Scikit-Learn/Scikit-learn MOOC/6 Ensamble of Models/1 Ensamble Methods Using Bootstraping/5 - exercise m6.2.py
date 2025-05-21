import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../datasets/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
X, y = df[[feature_name]], df[target_name]
X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=0
)

random_forest = RandomForestRegressor(
	n_estimators=3
)

random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest_scores = mean_absolute_error(y_test, y_pred)
print(
	f"Bagged decision tree classifier:{random_forest_scores}"
)

X_range = pd.DataFrame(np.linspace(170, 235, num=300), columns=X.columns)
trees = []
for tree_idx, tree in enumerate(random_forest.estimators_):
	single_tree_pred = tree.predict(X_range.to_numpy())
	plt.plot(
		X_range[feature_name],
		single_tree_pred,
		linestyle="--",
		color="tab:blue",
		alpha=0.2,
		label=f"Tree #{tree_idx}",
	)
	trees.append(single_tree_pred)
random_forest_pred = random_forest.predict(X_range)
sns.scatterplot(
	data=df,
	x=feature_name,
	y=target_name,
	color="black",
	alpha=0.8
)

plt.plot(
	X_range[feature_name],
	random_forest_pred,
	color="tab:orange",
	label="Random Forest"
)
plt.legend()
plt.show()
