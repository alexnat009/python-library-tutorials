import time

import pandas as pd
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

df = pd.read_csv('../../../datasets/adult-census.csv')

df = df.drop(columns="education-num")
target_name = "class"
X = df.drop(columns=target_name)
y = df[target_name]

numerical_column_selector = selector(dtype_exclude=object)
categorical_column_selector = selector(dtype_include=object)

numerical_columns = numerical_column_selector(X)
categorical_columns = categorical_column_selector(X)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
preprocessor = ColumnTransformer(
	[
		("categorical", categorical_preprocessor, categorical_columns)
	],
	remainder="passthrough"
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

start = time.time()
cv_result = cross_validate(model, X, y, cv=5)
elapsed_time = time.time() - start

scores = cv_result["test_score"]

print(
	"Only Ordinal Encoder\nThe mean cross-validation accuracy is: "
	f"{scores.mean():.3f} ± {scores.std():.3f} \n"
	f"with a fitting time of {elapsed_time:.3f}"
)

numerical_preprocessor = StandardScaler()
preprocessor = ColumnTransformer(
	[
		("categorical", categorical_preprocessor, categorical_columns),
		("numerical", numerical_preprocessor, numerical_columns)
	]
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

start = time.time()
cv_result = cross_validate(model, X, y, cv=5)
elapsed_time = time.time() - start
print(
	"Ordinal Encoder + Standard Scaler\nThe mean cross-validation accuracy is: "
	f"{scores.mean():.3f} ± {scores.std():.3f} \n"
	f"with a fitting time of {elapsed_time:.3f}"
)

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
	[
		("categorical", categorical_preprocessor, categorical_columns),
		("numerical", numerical_preprocessor, numerical_columns)
	]
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

start = time.time()
cv_result = cross_validate(model, X, y, cv=5)
elapsed_time = time.time() - start
print(
	"Ordinal Encoder + Standard Scaler\nThe mean cross-validation accuracy is: "
	f"{scores.mean():.3f} ± {scores.std():.3f} \n"
	f"with a fitting time of {elapsed_time:.3f}"
)
