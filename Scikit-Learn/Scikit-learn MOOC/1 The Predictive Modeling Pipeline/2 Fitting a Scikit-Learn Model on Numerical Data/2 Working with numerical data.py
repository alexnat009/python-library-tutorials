import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
df = df.drop(columns="education-num")
print(df.head())
X = df.drop(columns='class')
y = df["class"]

print(y.value_counts(normalize=True) * 100)

numerical_columns = X.select_dtypes("number")
print(numerical_columns.head())
print(numerical_columns["age"].describe())

X_train, X_test, y_train, y_test = train_test_split(
	numerical_columns,
	y,
	random_state=53,
	test_size=0.2
)
lr = LogisticRegression()
lr.fit(X_train, y_train)
accuracy = lr.score(X_test, y_test)
print(f"accuracy is {accuracy}")
