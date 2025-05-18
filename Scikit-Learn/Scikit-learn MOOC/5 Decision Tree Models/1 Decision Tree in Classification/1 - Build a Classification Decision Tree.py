import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("../../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"
X = df[culmen_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)
test_score = lr.score(X_test, y_test)
print(f"Accuracy of the LogisticRegression: {test_score:.2f}")

tab10_norm = mpl.colors.Normalize(vmin=-0.5, vmax=8.5)
palette = ["tab:blue", "tab:green", "tab:orange"]

disp = DecisionBoundaryDisplay.from_estimator(
	lr,
	X_train,
	response_method="predict",
	cmap="tab10",
	norm=tab10_norm,
	alpha=0.5
)

sns.scatterplot(
	data=df,
	x=culmen_columns[0],
	y=culmen_columns[1],
	hue=target_column,
	palette=palette
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("Decision boundary using a logistic regression")
plt.subplots_adjust(right=0.76)
plt.show()

"""
We see that the lines are a combination of the input features since they are not
perpendicular to a specific axis, this is due to the model parametrization that we 
saw in some previous notebooks
"""

"""
Unlike linear models, the decision rule for the decision tree is not controlled by a simple
linear combination of weights and feature values

Instead, the decision rules of trees can be defined in terms of

○ The feature index used at each split node of the tree
○ The threshold value used at each split node
○ The value to predict at each leaf node

Decision trees partition the feature space by considering a single feature at a time
The number of splits depends on both the hyperparameters and the number of data points
in the training set; the more flexible the hyperparameters and the larget the training set,
the more splits can be considered by the model

As the number of adjustable components taking part in the decision rule changes with the training
size, we say that decision trees are non-parametric models


"""
dtc = DecisionTreeClassifier(max_depth=1)
dtc.fit(X_train, y_train)
DecisionBoundaryDisplay.from_estimator(
	dtc,
	X_train,
	response_method="predict",
	cmap="tab10",
	norm=tab10_norm,
	alpha=0.5
)

sns.scatterplot(
	data=df,
	x=culmen_columns[0],
	y=culmen_columns[1],
	hue=target_column,
	palette=palette
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.subplots_adjust(right=0.76)
plt.title("Decision boundary using a decision tree")
plt.show()

"""
The partitions found by the algorithm separates the data along the axis "Culmen Depth"
discarding the feature "Culmen Length". Thus, it highlights that a decision tree doesn't use a 
combination of features when making a single split. 
"""

_, ax = plt.subplots(figsize=(8, 6))
plot_tree(
	dtc,
	feature_names=culmen_columns,
	class_names=dtc.classes_.tolist(),
	impurity=False,
	ax=ax
)
plt.show()

"""
We see that the split was done on the culmen depth feature. The original dataset was subdivided
into 2 sets based on the culmen depth

This partition of the dataset minimizes the class diversity in each sub-partition. This measure 
is also known as a criterion, and is a settable parameter.

If we look more closely at the partition, we see that the sample superior to 16.45mm belongs 
mainly to the "Adelie" class. Looking at the values, we indeed observe 103 "Adelie" individuals 
in this space. We also count 52 "Chinstrap" samples and 6 "Gentoo" samples. We can make similar 
interpretation for the partition defined by a threshold inferior to 16.45mm. In this case, the most
represented class is the "Gentoo" species.
"""

test_penguin_1 = pd.DataFrame(
	{"Culmen Length (mm)": [0], "Culmen Depth (mm)": [15]}
)

print(dtc.predict(test_penguin_1))
test_penguin_2 = pd.DataFrame(
	{"Culmen Length (mm)": [0], "Culmen Depth (mm)": [20]}
)

print(dtc.predict(test_penguin_2))

"""
During the training, we have a count of samples in each partition, we can also compute the probability
of belonging to a specific class within the partition.
"""

y_pred_proba = dtc.predict_proba(test_penguin_2)
y_proba_class_0 = pd.Series(y_pred_proba[0], index=dtc.classes_)
y_proba_class_0.plot.bar()
plt.ylabel("Probability")
plt.title("Probability to belong to a penguin class")
plt.show()

"""
We can also compute the different probabilities manually directly from the tree structure
"""
adelie_proba = 103 / 161
chinstrap_proba = 52 / 161
gentoo_proba = 6 / 161
print(
	"Probabilities for the different classes:\n"
	f"Adelie: {adelie_proba:.3f}\n"
	f"Chinstrap: {chinstrap_proba:.3f}\n"
	f"Gentoo: {gentoo_proba:.3f}\n"
)

"""
Also note that as culmen length has been disregarded, it's not used during the prediction
"""
test_penguin_3 = pd.DataFrame(
	{"Culmen Length (mm)": [10_000], "Culmen Depth (mm)": [17]}
)
print(dtc.predict_proba(test_penguin_3))

test_score = dtc.score(X_test, y_test)
print(f" Accuracy of the DecisionTreeClassifier: {test_score:.2f}")

"""
We saw earlier that a single feature is not able to separate all three species: it Underfits.
However, from the previous analysis we saw that by using both features we should be able to get fairly
good results.
"""