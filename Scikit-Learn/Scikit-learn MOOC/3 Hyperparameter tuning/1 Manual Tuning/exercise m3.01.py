from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv("../../datasets/adult-census.csv")

target_name = "class"

X = df.drop(columns=[target_name, "education-num"])
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer([
	("categorical_preprocessor", categorical_preprocessor, selector(dtype_include=object)),
], remainder="passthrough")

model = Pipeline([
	("preprocessor", preprocessor),
	("classifier", HistGradientBoostingClassifier(random_state=42))
])

learning_rate = [0.01, 0.1, 1, 10]
max_leaf_nodes = [3, 10, 30]
pprint(model.get_params())
best_score = 0
best_params = {}
for rate in learning_rate:
	for max_leaf in max_leaf_nodes:
		params = {"classifier__max_leaf_nodes": max_leaf, "classifier__learning_rate": rate}
		model.set_params(**params)

		scores = cross_val_score(model, X_train, y_train, cv=2)
		print(
			f"Accuracy score via cross-validation with learning_rate={rate} and max_leaf={max_leaf}:\n"
			f"{scores.mean():.3f} ± {scores.std():.3f}"
		)
		mean_score = scores.mean()
		if mean_score > best_score:
			best_score = mean_score
			best_params = {"classifier__learning_rate": rate, "classifier__max_leaf_nodes": max_leaf}

print(f"The best accuracy obtained is {best_score:.3f}")
print(f"The best parameters found are:\n {best_params}")

model.set_params(**best_params)
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Test score after the parameter tuning: {test_score:.3f}")
