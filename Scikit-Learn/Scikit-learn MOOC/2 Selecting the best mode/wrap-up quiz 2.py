import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ShuffleSplit, cross_validate, ValidationCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../datasets/blood_transfusion.csv")

target_name = "Class"
X = df.drop(columns=target_name)
y = df[target_name]

print(y.value_counts())
cv = ShuffleSplit(n_splits=10, random_state=0, test_size=0.2)

dummy = DummyClassifier(strategy="most_frequent")
cv_results = cross_validate(dummy, X, y, cv=cv, scoring="balanced_accuracy")
print(cv_results["test_score"].mean())

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
pprint(model.get_params())

cv_results = cross_validate(model, X, y, cv=cv, scoring="balanced_accuracy", return_train_score=True)
print(cv_results["test_score"].mean(), cv_results["train_score"].mean())

param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

ValidationCurveDisplay.from_estimator(
	model,
	X,
	y,
	param_name="kneighborsclassifier__n_neighbors",
	param_range=param_range,
	scoring="balanced_accuracy",
	cv=5,
	score_name="Balanced Accuracy",
	std_display_style="errorbar",
	errorbar_kw={"alpha": 0.7},
	n_jobs=2
)
plt.show()
