import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, ShuffleSplit, ValidationCurveDisplay, LearningCurveDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import pandas as pd

df = pd.read_csv('../../../datasets/blood_transfusion.csv')

X = df.drop(columns="Class")
y = df["Class"]

model = make_pipeline(StandardScaler(), SVC())
cv = ShuffleSplit(random_state=0)
cv_results = cross_validate(model, X, y, cv=cv)
cv_results = pd.DataFrame(cv_results)
print(cv_results)

print(
	"Accuracy score of our model:\n"
	f"{cv_results['test_score'].mean():.3f} ± "
	f"{cv_results['test_score'].std():.3f}"
)
gamma_range = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
disp = ValidationCurveDisplay.from_estimator(
	model,
	X,
	y,
	param_name=param_name,
	param_range=gamma_range,
	cv=cv,
	scoring="accuracy",
	score_name="Accuracy",
	std_display_style="errorbar",
	errorbar_kw={"alpha": 0.7},
	n_jobs=2
)
disp.ax_.set(
	xlabel=r"Value of hyperparameter $\gamma$",
	title="Validation curve of support vector machine",
)
# plt.show()

train_sizes = np.linspace(0.1, 1, num=10)
LearningCurveDisplay.from_estimator(
	model,
	X,
	y,
	train_sizes=train_sizes, cv=cv,
	score_type="both",
	score_name="Accuracy",
	scoring="accuracy",
	std_display_style="errorbar",
	errorbar_kw={"alpha": 0.7},
	n_jobs=2
)
plt.show()