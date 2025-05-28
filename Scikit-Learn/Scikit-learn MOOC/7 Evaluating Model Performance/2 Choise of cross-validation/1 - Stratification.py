import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate, ShuffleSplit, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(as_frame=True, return_X_y=True)

model = make_pipeline(StandardScaler(), LogisticRegression())

data_random = np.random.randn(9, 1)
cv = KFold(n_splits=3)
for train_index, test_index in cv.split(data_random):
	print("TRAIN:", train_index, "TEST:", test_index)

"""
By defining three splits we use three samples (1-fold) for testing and six (2-fold) for training
each time. KFold doesn't shuffle by default. It means that the tree first samples are selected for the testing
set at the first split, then the three next three samples for the second split, and the three next for the last split.
In the ned, all samples have been used in testing at least once among the different splits


"""

results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(
	f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)

y.plot()
plt.xlabel("Sample index")
plt.ylabel("Class")
plt.yticks(y.unique())
plt.title("Class value in target y")
plt.show()

"""
We see that the target vector y is ordered. This has some unexpected consequences when using the KFold cross-validation
To illustrate the consequences, we show the class count in each fold of the cross-validation in the train and test set
"""

n_splits = 3
cv = KFold(n_splits=n_splits)


def plot_cv_splits(cv, data, target, title=None):
	train_cv_counts = []
	test_cv_counts = []
	for fold_idx, (train_idx, test_idx) in enumerate(cv.split(data, target)):
		y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

		train_cv_counts.append(y_train.value_counts())
		test_cv_counts.append(y_test.value_counts())

	train_cv_counts = pd.concat(
		train_cv_counts, axis=1, keys=[f"Fold #{idx}" for idx in range(n_splits)]
	)
	test_cv_counts = pd.concat(
		test_cv_counts, axis=1, keys=[f"Fold #{idx}" for idx in range(n_splits)]
	)
	train_cv_counts.index.name = "Class label"
	test_cv_counts.index.name = "Class label"

	print(train_cv_counts)
	print(test_cv_counts)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	train_cv_counts.plot.bar(ax=axes[0])
	axes[0].legend(loc="upper left")
	axes[0].set_ylabel("Count")
	axes[0].set_title("Training set class counts")

	test_cv_counts.plot.bar(ax=axes[1])
	axes[1].legend(loc="upper left")
	axes[1].set_ylabel("Count")
	axes[1].set_title("Test set class counts")

	plt.tight_layout()
	fig.suptitle(title)
	plt.show()


plot_cv_splits(cv, X, y, "KFold")
"""
We can confirm that in each fold, only two of the three classes are present in the training set
and all samples of the remaining class is used as a test set. So our model is unable to predict this class
that was unseen during the training stage.


One possibility to solve the issue is to shuffle the data before splitting the it into three groups
"""

cv = KFold(n_splits=3, shuffle=True, random_state=0)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(
	f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)
"""
We get results that are closer to what we would expect with an accuracy above 90%. Now that we solved our
first issue, it would be interesting to check if the class frequency in the training and testing set is equal to our
original set's class frequency. It would ensure that we are training and testing our model with a class distribution
that we would encounter in production 
"""

plot_cv_splits(cv, X, y, "Shuffled KFold")

"""
We see that neither the training and testing sets have the same class frequencies as our original
dataset because the count for each class is varying a little

However, one might want to split our data by preserving the original class frequencies: we want to 
stratify our data by class. In scikit-learn, some cross-validation strategies implement the stratification
"""

cv = StratifiedKFold(n_splits=3)
result = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(
	f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)
plot_cv_splits(cv, X, y, "Stratified KFold")


"""
In this case, we observe that the class counts are very close both in the training and testing set. The difference
is due to the small number of samples in the iris dataset

In other words, stratifying is more effective than just shuffling when it comes to making sure that the distributions
of classes in all the folds are representative of the entire dataset. As training and testing folds have similar class 
distribution, stratifying lead to a more realistic measure of the model's ability to generalize. This is specially
important when the performance metrics depend on the proportion of the positive class 
"""