import pandas as pd
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

ames_housing = pd.read_csv("../datasets/ames_housing_no_missing.csv")

target_name = "SalePrice"
X = ames_housing.drop(columns=target_name)
y = ames_housing[target_name]
y = (y > 200_000).astype(int)
print(y)

print(X.info())
print(X.head())

numerical_column_selector = selector(dtype_exclude=object)
numerical_columns = numerical_column_selector(X)
print(len(numerical_columns))

numerical_features = [
	"LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
	"BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
	"GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
	"GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
	"3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
X_numerical = X[numerical_features]

model = make_pipeline(StandardScaler(), LogisticRegression())

cv_results_nums = cross_validate(model, X_numerical, y, cv=10)
scores_num = cv_results_nums["test_score"]

print(
	"The mean cross-validation accuracy is: "
	f"{scores_num.mean():.3f} ± {scores_num.std():.3f}"
)
categorical_features = [col for col in X.columns if col not in numerical_features]
print(categorical_features)
preprocessor = ColumnTransformer(
	[
		("numerical", StandardScaler(), numerical_features),
		("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
	]
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

cv_results_full = cross_validate(model, X, y, cv=10)
scores_full = cv_results_full["test_score"]

print(
	"The mean cross-validation accuracy is: "
	f"{scores_full.mean():.3f} ± {scores_full.std():.3f}"
)

plt.figure(figsize=(8, 5))
plt.scatter(range(1, 11), scores_num, marker='o', linestyle='--', label="Numerical Only")
plt.scatter(range(1, 11), scores_full, marker='s', linestyle='-', label="Numerical + Categorical")

# Formatting
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Accuracy Score")
plt.title("Fold-to-Fold Comparison of Model Accuracy")
plt.legend()
plt.grid(True)
plt.show()
