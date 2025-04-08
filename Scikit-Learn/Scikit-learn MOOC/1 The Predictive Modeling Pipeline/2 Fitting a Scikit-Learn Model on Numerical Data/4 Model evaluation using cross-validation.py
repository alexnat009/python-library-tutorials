from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd

df = pd.read_csv("../../../datasets/adult-census.csv")

df = df.drop(columns="education-num")
X = df.drop(columns='class')
y = df["class"]
numerical_columns = X.select_dtypes("number")

model = make_pipeline(StandardScaler(), LogisticRegression())
cv_result = cross_validate(model, numerical_columns, y, cv=5, verbose=1)
for i in cv_result.items():
	print(i)

scores = cv_result["test_score"]
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)