import pandas as pd
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv('../../../datasets/adult-census.csv')

df = df.drop(columns="education-num")
target_name = "class"
X = df.drop(columns=target_name)
y = df[target_name]

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(X)
categorical_columns = categorical_columns_selector(X)

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer(
	[
		("one-hot-encoder", categorical_preprocessor, categorical_columns),
		("standard_scaler", numerical_preprocessor, numerical_columns),
	]
)

"""
A 'ColumnTransformer' does the following:
	1) 	It splits the columns of the original dataset based on the column names or indices
		provided. We obtain as many subsets as the number of transformers passed into the 
		ColumnTransformer
		
	2)	It transforms each subset. A specific transformer is applied to each subset: it
		internally calls 'fit_transform' or 'transform'. The output of this step is a set of 
		transformed datasets
		
	3)	It then concatenates the transformed datasets into a single dataset

The important thing is that it is like any other scikit-learn transformer. In particular
it can be combined with a classified in a 'pipeline'
"""

model = make_pipeline(
	preprocessor,
	LogisticRegression(max_iter=500)
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

cv_results = cross_validate(model, X, y, cv=5)
for i in cv_results.items():
	print(i)
scores = cv_results["test_score"]
print(
	"The mean cross-validation accuracy is: "
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)

categorical_preprocessor_ensemble = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
preprocessor_ensemble = ColumnTransformer(
	[
		("categorical", categorical_preprocessor_ensemble, categorical_columns)
	],
	remainder="passthrough"
)

model_ensemble = make_pipeline(preprocessor_ensemble, HistGradientBoostingClassifier())

model_ensemble.fit(X_train, y_train)
print(model_ensemble.score(X_test, y_test))

cv_results_ensemble = cross_validate(model_ensemble, X, y, cv=5)
scores = cv_results_ensemble["test_score"]
print(
	"The mean cross-validation accuracy is: "
	f"{scores.mean():.3f} ± {scores.std():.3f}"
)
