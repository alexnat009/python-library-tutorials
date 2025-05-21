import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

X, y = df[culmen_columns], df[target_column]

"""
We will purposefully train a shallow decision tree. Since it is shallow, it is unlikely to overfit
and some of the training examples will even be misclassified
"""

palette = ["tab:red", "tab:blue", "black"]

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X, y)

"""
We can predict on the same dataset and check with samples are misclassified
"""

y_pred = tree.predict(X)
misclassified_samples_idx = np.flatnonzero(y != y_pred)
X_misclassified = X.iloc[misclassified_samples_idx]

DecisionBoundaryDisplay.from_estimator(
	tree,
	X,
	response_method="predict",
	cmap="RdBu",
	alpha=0.5
)

# Plot the original dataset
sns.scatterplot(
	data=df,
	x=culmen_columns[0],
	y=culmen_columns[1],
	hue=target_column,
	palette=palette
)
# Plot the misclassified samples
sns.scatterplot(
	data=X_misclassified,
	x=culmen_columns[0],
	y=culmen_columns[1],
	label="Misclassified samples",
	marker="+",
	s=150,
	color="k"
)

plt.legend(loc="lower left")
plt.title("Decision tree predictions \nwith misclassified samples highlighted")
plt.show()

"""
We observe that several samples have been misclassified by the classifier

We mentioned that boosting relies on creating a new classifier which tries to correct these 
misclassifications. In scikit-learn, learners have a parameter sample-weight which forces it to pay 
more attention to samples with higher weight during training

This parameter is set when calling classifier.fit(X,y,sample_weight=weight). We will use this trick to create 
a new classifier by 'discarding' all correctly classified samples and only considering the misclassified samples.
Thus, misclassified samples will be assigned a weight of 1 and well classified samples will be assigned a weight of 0
"""

sample_weight = np.zeros_like(y, dtype=int)
sample_weight[misclassified_samples_idx] = 1

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X, y, sample_weight=sample_weight)

DecisionBoundaryDisplay.from_estimator(
	estimator=tree,
	X=X,
	response_method="predict",
	cmap="RdBu",
	alpha=0.5
)

sns.scatterplot(
	data=df,
	x=culmen_columns[0],
	y=culmen_columns[1],
	hue=target_column,
	palette=palette
)

sns.scatterplot(
	data=X_misclassified,
	x=culmen_columns[0],
	y=culmen_columns[1],
	label="Previously misclassified samples",
	marker="+",
	s=150,
	color="k"
)

plt.legend(loc="lower left")
plt.title("Decision tree by changing sample weights")
plt.show()

"""
We see that the decision function drastically changed. Qualitatively, we see that the previously
misclassified samplesa re now correctly classified
"""

y_pred = tree.predict(X)
newly_misclassified_samples_idx = np.flatnonzero(y != y_pred)
remaining_misclassified_samples_idx = np.intersect1d(
	misclassified_samples_idx, newly_misclassified_samples_idx
)

print(
	"Number of samples previously misclassified and "
	f"still misclassified: {len(remaining_misclassified_samples_idx)}"
)

"""
However we are making mistakes on previously well classified samples. Thus, we get the intuition
that we should weight the predictions of each classifier differently, most probably by using the number of 
mistakes each classifier is making

So we could use that classification error to combine both trees
"""

ensemble_weight = [
	(y.shape[0] - len(misclassified_samples_idx)) / y.shape[0],
	(y.shape[0] - len(newly_misclassified_samples_idx)) / y.shape[0],
]
print(ensemble_weight)

"""
The first classifier was 94% accurate and the second one 69%. Therefore, when prediction a class, we should 
trust the first classifier slightly more than the second one. We could use these accuracy values to weight the
predictions of each learner

"""

"""
To summarize, boosting learns several classifiers, each of which will focus more or less on specific samples of the 
dataset. Boosting is thus different from bagging: here we never resample our dataset, we just assign different weight 
to the original dataset.

Boosting requires some strategy to combine the learners together:

○ one needs to define a way to compute the weights to be assigned to samples;
○ one needs to assign a weight to each learner when making predictions

Indeed, we defined a really simple scheme to assign sample weight and learner weight.
However, there are statistical theories (like in AdaBoost) for how these sample and learner weight can be optimally 
calculated

We will use the AdaBoost classifier implemented in scikit-learn and look at the underlying decision tree classifiers 
trained
"""
estimator = DecisionTreeClassifier(max_depth=3, random_state=0)

adaboost = AdaBoostClassifier(
	estimator=estimator,
	n_estimators=3,
	random_state=0
)
adaboost.fit(X, y)

for boosting_round, tree in enumerate(adaboost.estimators_):
	DecisionBoundaryDisplay.from_estimator(
		tree,
		X.to_numpy(),
		response_method="predict",
		cmap="RdBu",
		alpha=0.5
	)

	sns.scatterplot(
		data=df,
		x=culmen_columns[0],
		y=culmen_columns[1],
		hue=target_column,
		palette=palette,
	)
	plt.legend(loc="lower left")
	plt.title(f"Decision tree trained at round {boosting_round}")
plt.show()

print(f"Weight of each classifier: {adaboost.estimator_weights_}")
print(f"Error of each classifier: {adaboost.estimator_errors_}")


"""
We see that AdaBoost learned three different classifiers, each of which focuses on different
samples. Looking at the weight of each learner, we see that the ensemble gives the highest weight
to the first classifier. This indeed makes sense when we look at the errors of each classifier.
The first classifier also has the highest classification generalization performance

While AdaBoost is a nice algorith to demonstrate the internal machinery of boosting
algorithms, its' not the most efficient. This title is handed to the gradient-boosting decision
tree (GBDT) algorithm 
"""