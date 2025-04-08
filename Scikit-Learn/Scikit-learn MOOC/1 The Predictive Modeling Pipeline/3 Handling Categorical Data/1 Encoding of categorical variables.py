from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd

df = pd.read_csv("../../../datasets/adult-census.csv")

df = df.drop(columns="education-num")
X = df.drop(columns='class')
y = df["class"]

print(X["native-country"].value_counts().sort_values(ascending=False))
print(X.dtypes)

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(X)
print(categorical_columns)

X_categorical = X[categorical_columns]
print(X_categorical)

education_column = X_categorical[["education"]]
encoder: OrdinalEncoder = OrdinalEncoder().set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
print(education_encoded)
# print(encoder.categories_)

X_encoded = encoder.fit_transform(X_categorical)
# print(type(encoder))
print(encoder.categories_)
# print(X_encoded[:5])
"""
However, be careful when applying this encoding strategy: using this integer representation leads downstream
predictive models to assume that the values are ordered (0 < 1 < 2 < 3… for instance).
"""

encoder: OneHotEncoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
print(education_encoded)

X_encoded = encoder.fit_transform(X_categorical)
print(X_encoded[:5])

print(X["native-country"].value_counts())
"""
We see that the 'Holand-Netherlands' category is occurring rarely. This will be a problem
during cross-validation: if the sample ends up in the test set during splitting then classifier
wouldn't have seen that category during training and would not be able to encode it.


In scikit-learn, there are some possible solutions to bypass this issue:

1) List all the possible categories and provide them to the encoder via the keyword argument
'categories' instead of letting the estimator automatically determine them from the training data
when calling fit

2) set the parameter 'handle_unknown="ignore"', i.e. if an unknown category is encountered
during transform, the resulting one-hot encoded columns for this feature will be all zeros

3) adjust the 'min_frequency' parameter to collapse the rarest categories observed in the
training data into a single one-hot-encoded feature. If you enable this option, you can also
set 'handle_unknown"infrequent_if_exist"' to encode the unknown categories (categories only observed
at predict time) as ones in the last column
"""

model = make_pipeline(
	OneHotEncoder(handle_unknown="ignore"),
	LogisticRegression(max_iter=500)
)

cv_results = cross_validate(model, X_categorical, y)
for i in cv_results.items():
	print(i)

scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")
