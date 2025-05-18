from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv(
	"../datasets/ames_housing_no_missing.csv",
	na_filter=False,  # required for pandas>2.0
)
target_name = "SalePrice"
data = df.drop(columns=target_name)
target = df[target_name]

numerical_features = [
	"LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
	"BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
	"GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
	"GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
	"3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]

lr = make_pipeline(StandardScaler(), LinearRegression())
dtr = make_pipeline(DecisionTreeRegressor(random_state=0))

cv_results_lr = cross_val_score(lr, data_numerical, target, cv=10)
cv_results_dtr = cross_val_score(dtr, data_numerical, target, cv=10)

print(np.sum(cv_results_lr > cv_results_dtr))

param_grid = {"max_depth": np.arange(1, 15, 1)}

inner_cv = KFold(n_splits=10, shuffle=True, random_state=0)
grid_search = GridSearchCV(
	estimator=DecisionTreeRegressor(random_state=0),
	param_grid=param_grid,
	cv=inner_cv,
)

outer_cv = KFold(n_splits=10, shuffle=True, random_state=1)
cv_results = cross_validate(
	grid_search,
	data_numerical,
	target,
	cv=outer_cv,
	return_estimator=True,
	n_jobs=3
)

best_depths = [est.best_params_["max_depth"] for est in cv_results["estimator"]]
# 5. Find the most common depth
depth_counts = Counter(best_depths)
optimal_depth, count = depth_counts.most_common(1)[0]

print(f"Optimal tree depth (most frequent): {optimal_depth} (selected in {count}/10 folds)")

print(
	f'Optimal Decision Tree is better in {np.sum(np.round(cv_results_lr, 3) < np.round(cv_results["test_score"], 3))}/10 folds')

categorical_columns_selector = make_column_selector(dtype_include=object)
categorical_columns = categorical_columns_selector(df)

preprocessor = ColumnTransformer([
	("numerical_preprocessor", StandardScaler(), numerical_features),
	("categorical_preprocessor", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
	 categorical_columns)
])

model = make_pipeline(preprocessor, DecisionTreeRegressor(max_depth=7))
cv = KFold(n_splits=10, shuffle=True, random_state=0)
cv_results_full = cross_validate(model, data, target, cv=cv)

print(
	f'Optimal Decision Tree With full features is better in '
	f'{np.sum(np.round(cv_results_full["test_score"], 3) > np.round(cv_results["test_score"], 3))}/10 folds')
