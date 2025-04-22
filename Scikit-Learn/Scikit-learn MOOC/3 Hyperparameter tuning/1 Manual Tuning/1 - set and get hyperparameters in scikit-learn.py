import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

df = pd.read_csv("../../../datasets/adult-census.csv")
target_name = "class"
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

X = df[numerical_columns]
y = df[target_name]

model = Pipeline(
	steps=[
		("preprocessor", StandardScaler()),
		("classifier", LogisticRegression())
	]

)
cv_results = cross_validate(model, X, y)
scores = cv_results["test_score"]
print(
	"Accuracy score via cross-validation:\n"
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)

params = {"classifier__C": 1e-3}
model.set_params(**params)
cv_results = cross_validate(model, X, y)
scores = cv_results["test_score"]
print(
	"Accuracy score via cross-validation:\n"
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)

for parameter in model.get_params():
	print(parameter)
for C in np.logspace(-3, 1, num=5):
	model.set_params(classifier__C=C)
	cv_results = cross_validate(model, X, y)
	scores = cv_results["test_score"]
	print(
		f"Accuracy score via cross-validation with C={C}:\n"
		f"{scores.mean():.3f} ± {scores.std():.3f}"
	)
