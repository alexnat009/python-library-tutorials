import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../../datasets/adult-census-numeric.csv")

# print(df.info())
X = df.select_dtypes("number").to_numpy()
y = df.select_dtypes("object").to_numpy().ravel()

model = KNeighborsClassifier()
model.fit(X, y)

df_test = pd.read_csv("../../datasets/adult-census-numeric-test.csv")
X_test = df_test.select_dtypes("number").to_numpy()
y_test = df_test.select_dtypes("object").to_numpy().ravel()

accuracy = model.score(X_test, y_test)
print(f'accuracy = {accuracy}')
