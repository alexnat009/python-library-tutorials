import numpy as np
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

np.random.seed(42)
data = np.random.randn(100, 100_000)
target = np.random.randint(0, 2, 100)

lr = LogisticRegression()

cv_results = cross_validate(lr, data, target)
score = cv_results["test_score"]
print(f"The mean accuracy is: {score.mean():.3f}")
"""
It is not surprising that the logistic regression model performs as bad as pure chance when we provide the
full dataset
"""

fs = SelectKBest(score_func=f_classif, k=10)
data_subset = fs.fit_transform(data, target)
test_score = cross_val_score(lr, data_subset, target)
print(f"The mean accuracy is: {test_score.mean():.3f}")

"""
Surprisingly, the logistic regression succeeded in having a fantastic accuracy using data that didn't have
any link with the target in the first place. We therefore know that these results are not legit

The reasons for obtaining these results are two folds: the pool of available features is a large compared to the number
of samples. It is possible to find a subset of features that will link the data and the target. By not splitting the 
data, we leak knowledge from the entire dataset and could use this knowledge while evaluating our model.

"""

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

fs = SelectKBest(score_func=f_classif, k=10)
fs.fit(X_train, y_train)
data_train_subset = fs.transform(X_train)
data_test_subset = fs.transform(X_test)
lr.fit(data_train_subset, y_train)
test_score = lr.score(data_test_subset, y_test)
print(f"The mean accuracy is: {test_score:.3f}")

"""
It's not a surprise that our model is not working. We see that selecting features only on the training
set will not help when testing our model. In this case, we obtained the expected results

Therefore, as with hyperparameters optimization or model selection, tuning the feature space should be done
solely on the training set, keeping a part of the data left-out

However, the previous case is not perfect. For instance, if we were asking to perform cross-validation,
the manual fit/transform of the datasets will make our life hard. Indeed, the solution here is to use a scikit-learn
pipeline in which the feature selection will be a pre processing stage before to train the model

"""

model = make_pipeline(
	SelectKBest(score_func=f_classif, k=10),
	LogisticRegression()
)
test_score = cross_val_score(model, data, target)
print(f"The mean accuracy is: {test_score.mean():.3f}")
