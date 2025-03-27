import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

df = pd.read_csv("../../../datasets/adult-census.csv")

df = df.drop(columns="education-num")
print(df.head())
X = df.drop(columns='class')
y = df["class"]
numerical_columns = X.select_dtypes("number")

X_train, X_test, y_train, y_test = train_test_split(numerical_columns, y, random_state=42, test_size=0.2)
print(y_train)


def dummy(kwargs):
	dc = DummyClassifier(**kwargs)
	dc.fit(X_train, y_train)
	accuracy = dc.score(X_test, y_test)
	print(f'Accuracy - {accuracy}')


dummy({"strategy": "constant", "constant": " <=50K"})
dummy({"strategy": "constant", "constant": " >50K"})
dummy({"strategy": "most_frequent"})
