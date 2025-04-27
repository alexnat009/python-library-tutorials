import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../datasets/penguins_classification.csv")

print(df.head())

print(df.info())
# print(df.select_dtypes("number").columns)
# print(df.select_dtypes("object").columns)
X = df.select_dtypes("number").columns.values
y = df.select_dtypes("object").columns.values
print(X)
print(y)
print(df[y].value_counts())
df.hist()
plt.show()
sns.pairplot(
	df,
	vars=X,
	hue=y[0],
	plot_kws={"alpha": 0.5},
	height=3,
	# diag_kind="hist",
	# diag_kws={"bins": 30}
)
plt.show()
