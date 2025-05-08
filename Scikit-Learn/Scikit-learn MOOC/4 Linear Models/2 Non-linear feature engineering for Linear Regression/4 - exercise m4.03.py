import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

df = pd.read_csv("../../datasets/adult-census.csv")
target = df["class"]
data = df.select_dtypes(["integer", "floating"])
data = data.drop(columns=["education-num"])

model = make_pipeline(
	StandardScaler(),
	LogisticRegression()
)
cv_results_simple = cross_validate(model, data, target, cv=10, return_estimator=True, scoring="accuracy")
scores_simple = cv_results_simple["test_score"]
print(scores_simple)

coefs = [pipeline[-1].coef_[0] for pipeline in cv_results_simple["estimator"]]
coefs = pd.DataFrame(coefs, columns=data.columns)
print(coefs)
color = {"whiskers": "black", "medians": "black", "caps": "black"}
_, ax = plt.subplots()
coefs.abs().plot.box(color=color, vert=False, ax=ax)
plt.show()

df = pd.read_csv("../../datasets/adult-census.csv")
target = df["class"]
data = df.drop(columns=["class", "education-num"])

numerical_preprocessor = StandardScaler()
numerical_columns = selector(dtype_exclude=object)(data)

categorical_preprocessor = OneHotEncoder(
	handle_unknown="ignore",
	min_frequency=0.01
)
categorical_columns = selector(dtype_include=object)(data)

preprocessor = ColumnTransformer([
	("numerical_preprocessor", numerical_preprocessor, numerical_columns),
	("categorical_preprocessor", categorical_preprocessor, categorical_columns)
])

model = make_pipeline(
	preprocessor,
	LogisticRegression(max_iter=5000)
)
cv_results_complex = cross_validate(model, data, target, cv=10, return_estimator=True, scoring="accuracy")
scores_complex = cv_results_complex["test_score"]
print(scores_complex)

indices = np.arange(len(scores_simple))
plt.scatter(
	indices,
	scores_simple,
	color="tab:blue",
	label="numerical features only"
)
plt.scatter(
	indices,
	scores_complex,
	color="tab:red",
	label="all features"
)

plt.ylim((0, 1))
plt.xlabel("Cross-validation iteration")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.subplots_adjust(right=0.6)
plt.show()
print(
	"A model using both all features is better than a"
	" model using only numerical features for"
	f" {sum(scores_complex > scores_simple)} CV iterations out of 10."
)
preprocessor.fit(data)
feature_names = (
	preprocessor.named_transformers_["categorical_preprocessor"].get_feature_names_out(
		categorical_columns
	)
).tolist()
feature_names += numerical_columns

print(feature_names)

coefs = [pipeline[-1].coef_[0] for pipeline in cv_results_complex["estimator"]]
coefs = pd.DataFrame(coefs, columns=feature_names)
_, ax = plt.subplots(figsize=(14, 14))
coefs.abs().plot.box(color=color, vert=False, ax=ax)
plt.subplots_adjust(bottom=0.04, top=1, right=0.97, left=0.21)
plt.show()

model = make_pipeline(
	preprocessor,
	PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
	LogisticRegression(C=0.01, max_iter=5000)
)
cv_results_interactions = cross_validate(
	model,
	data,
	target,
	cv=10,
	n_jobs=2,
)
score_interactions = cv_results_interactions["test_score"]

plt.scatter(
	indices, scores_simple, color="tab:blue", label="numerical features only"
)
plt.scatter(
	indices,
	scores_complex,
	color="tab:red",
	label="all features",
)
plt.scatter(
	indices,
	score_interactions,
	color="black",
	label="all features and interactions",
)
plt.xlabel("Cross-validation iteration")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.subplots_adjust(right=0.6)
plt.show()
print(
	"A model using all features and interactions is better than a model"
	" without interactions for"
	f" {sum(score_interactions > scores_complex)} CV iterations"
	" out of 10."
)
