import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.kernel_approximation import Nystroem

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer

df = pd.read_csv("../datasets/ames_housing_no_missing.csv")
target_name = "SalePrice"
data = df.drop(columns=target_name)
target = df[target_name]

# Up to question 8 only numerical columns are used, after that all of them
numerical_features = [
	"LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
	"BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
	"GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
	"GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
	"3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
numerical_features.remove("GarageArea")  # Question 4,6
data_numerical = data[numerical_features]

# model = make_pipeline(StandardScaler(), Ridge(alpha=0)) # question 1
# model = make_pipeline(StandardScaler(), Ridge(alpha=1))  # question 2
alphas = np.logspace(-3, 3, num=101)
model = make_pipeline(StandardScaler(), RidgeCV(alphas))  # question 6
cv_results = cross_validate(model, data_numerical, target, cv=10, return_estimator=True, )
cv_results = pd.DataFrame(cv_results)
print(np.max(cv_results["estimator"].apply(lambda x: np.max(np.abs(x[-1].coef_)))))

coef_matrix = []
alpha_each_fold = []
for pipeline in cv_results['estimator']:
	ridge_model: Ridge = pipeline[-1]
	alpha_each_fold.append(ridge_model.alpha_)
	coefs = ridge_model.coef_
	coef_matrix.append(coefs)

print(f"alpha in folds: {np.mean(alpha_each_fold)}")

coef_matrix = np.array(coef_matrix)
coef_df = pd.DataFrame(coef_matrix, columns=numerical_features)
plt.figure(figsize=(12, 6))
coef_df.boxplot(rot=20)
plt.ylabel("Coefficient values")
plt.title("Distribution of Ridge coefficients across 10 folds")
plt.show()
median_abs_coef = coef_df.abs().median().sort_values(ascending=False)
top2_features = median_abs_coef.index[:2].tolist()

print("Two most important features based on median absolute coefficient:")
print(top2_features)

categorical_column_selector = make_column_selector(dtype_include=object)
categorical_columns = categorical_column_selector(data)

numerical_column_selector = make_column_selector(dtype_exclude=object)
numerical_columns = numerical_column_selector(data)

preprocessor = ColumnTransformer([
	("categorical_preprocessor", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
	("numerical_preprocessor", StandardScaler(), numerical_columns)
])
model_full = make_pipeline(preprocessor, RidgeCV(alphas=alphas))

simple_scores = cross_val_score(model, data_numerical, target, cv=10)
full_scores = cross_val_score(model_full, data, target, cv=10)

print(np.sum(simple_scores > full_scores))

preprocessor_nonlinear = ColumnTransformer([
	("categorical_preprocessor", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
	("numerical_preprocessor", SplineTransformer(), numerical_columns)
])
model_full_nonlinear = make_pipeline(
	preprocessor,
	Nystroem(kernel="poly", degree=2, n_components=300),
	RidgeCV(alphas=alphas)
)

full_scores_nonlinear = cross_val_score(model_full_nonlinear, data, target, cv=10)
print(np.sum(full_scores_nonlinear > full_scores))


"""
In this module, we saw that:

○ the predictions of a linear model depend on a weighted sum of the values of the input 
features added to an intercept parameter

○ fitting a linear model consists in adjusting both the weight coefficients and the intercept
to minimize the prediction errors on the training set

○ to train linear models successfully it is often required to scale the input features approximately
to the same dynamic range

○ regularization can be used to reduce over-fitting; weight coefficients are constrained to stay small
when fitting

○ the regularization hyperparameter needs to be fine-tuned by cross-validation for each new ML problem
and dataset

○ linear models can be used on problems where the target variable is not linearly related to the
input features but this requires extra feature engineering work to transform the data in order to avoid 
under-fitting 

"""