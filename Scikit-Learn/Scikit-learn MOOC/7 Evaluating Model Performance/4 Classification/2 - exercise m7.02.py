import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../../datasets/blood_transfusion.csv")
X = df.drop(columns="Class")
y = df["Class"]

dtr = DecisionTreeClassifier()

cv = StratifiedKFold(n_splits=10)

scores = cross_val_score(
	dtr,
	X,
	y,
	scoring="accuracy"
)
print(f"Accuracy score: {scores.mean():.3f} ± {scores.std():.3f}")

scores = cross_val_score(
	dtr,
	X,
	y,
	scoring="balanced_accuracy"
)
print(f"Balanced accuracy score: {scores.mean():.3f} ± {scores.std():.3f}")

tree = DecisionTreeClassifier()
try:
	scores = cross_val_score(
		tree, X, y, cv=10, scoring="precision", error_score="raise"
	)
except ValueError as exc:
	print(exc)

scorer = make_scorer(precision_score, pos_label="donated")
scores = cross_val_score(
	tree,
	X,
	y,
	cv=cv,
	scoring=scorer
)
print(f"Precision score: {scores.mean():.3f} ± {scores.std():.3f}")

scoring = ["accuracy", "balanced_accuracy"]
scores = cross_validate(
	tree,
	X,
	y,
	cv=cv,
	scoring=scoring
)
print(pd.DataFrame(scores))

color = {"whiskers": "black", "medians": "black", "caps": "black"}

metrics = pd.DataFrame(
	[scores["test_accuracy"], scores["test_balanced_accuracy"]],
	index=["Accuracy", "Balanced accuracy"],
).T

metrics.plot.box(vert=False, color=color)
plt.title("Computation of multiple scores using cross_validate")
plt.show()
