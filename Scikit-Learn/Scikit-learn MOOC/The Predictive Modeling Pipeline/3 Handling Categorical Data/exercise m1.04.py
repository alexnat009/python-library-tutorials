from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
import pandas as pd

df = pd.read_csv('../../../datasets/adult-census.csv')
target_name = "class"
X = df.drop(columns=[target_name, "education-num"])
y = df[target_name]

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(X)
print(categorical_columns)
X_categorical = X[categorical_columns]

modelOrdinalEncoder = make_pipeline(
	OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
	LogisticRegression(max_iter=500)
)
modelOneHotEncoder = make_pipeline(
	OneHotEncoder(handle_unknown="ignore"),
	LogisticRegression(max_iter=500)
)
cv_resultsOrdinal = cross_validate(modelOrdinalEncoder, X_categorical, y, cv=5, error_score="raise")
cv_resultsOneHot = cross_validate(modelOneHotEncoder, X_categorical, y, cv=5, error_score="raise")
scoreOrdinal = cv_resultsOrdinal["test_score"]
scoreOneHot = cv_resultsOneHot["test_score"]


def print_scores(scores):
	print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")


print_scores(scoreOrdinal)
print_scores(scoreOneHot)

cv_resultsDummy = cross_validate(DummyClassifier(strategy="most_frequent"), X_categorical, y)

scoreDummy = cv_resultsDummy["test_score"]
print_scores(scoreDummy)