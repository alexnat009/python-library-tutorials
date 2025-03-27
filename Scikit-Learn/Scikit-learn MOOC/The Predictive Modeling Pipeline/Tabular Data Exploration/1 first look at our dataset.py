import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../../datasets/adult-census.csv')

target_column = "class"
print(df[target_column].value_counts())

numerical_columns = df.select_dtypes('number').columns.values
categorical_columns = df.select_dtypes('object').columns.values
sorted_columns = np.concat([numerical_columns, categorical_columns])
df = df[sorted_columns]

print(f"The dataset contains {df.shape[0]} samples and {df.shape[1]} columns, hence 13 features")

# df.hist()
# plt.show()

print(df['sex'].value_counts())
print(df["education"].value_counts())

tmp = pd.crosstab(index=df["education"], columns=df["education-num"])
"""
For every entry in 'education', there is only one single corresponding value in 'education-num'. 
This shows that 'education' and 'education-num' give you the same information. 
For example, 'education-num'=2 is equivalent to 'education'='1st-4th'.
In practice that means we can remove 'education-num' without losing information.
Note that having redundant (or highly correlated) columns can be a problem for machine learning algorithms.
"""

n_sample_to_plot = 500
columns = ["age", "education-num", "hours-per-week"]
sns.pairplot(
	data=df[:n_sample_to_plot],
	vars=columns,
	hue=target_column,
	plot_kws={"alpha": 0.2},
	height=3,
	diag_kind="hist", diag_kws={"bins": 30}
)
plt.show()

"""
By looking at the previous plots, we could create some hand-written rules that predict whether someone has a high- or low-income.
For instance, we could focus on the combination of the "hours-per-week" and "age" features.
"""

sns.scatterplot(
	data=df[:n_sample_to_plot],
	x="age",
	y="hours-per-week",
	hue=target_column,
	alpha=0.5,
	s=100
)
plt.show()

"""
The data points show the distribution of 'hours-per-week' and 'age' in the dataset. Blue points mean low-income and
orange points mean high-income. This part of the plot is the same as the bottom-left plot in the pairplot above
"""

"""
Now we try to find regions that mainly contains a single class such that we can esaily decide what class one should predict
We could come up with hand-written rules as show in this plot
"""

sns.scatterplot(
	data=df[:n_sample_to_plot],
	x="age",
	y="hours-per-week",
	hue=target_column,
	alpha=.5,
	s=60
)
age_limit, hours_per_week_limit = 27, 40
plt.axvline(x=age_limit, ymin=0, ymax=1, color="black", linestyle="--")
plt.axhline(y=hours_per_week_limit, xmin=0.18, xmax=1, color="black", linestyle="--")

plt.annotate("<=50K", (17, 25), rotation=90, fontsize=35)
plt.annotate("<=50K", (35, 20), fontsize=35)
plt.annotate("???", (45, 60), fontsize=35)
plt.show()

"""
In the region age < 27 (left region) the prediction is low-income.
Indeed, there are many blue points and we cannot see any orange points.

In the region age > 27 AND hours-per-week < 40 (bottom-right region),
the prediction is low-income. Indeed, there are many blue points and only a few orange points.

In the region age > 27 AND hours-per-week > 40 (top-right region),
we see a mix of blue points and orange points.
It seems complicated to choose which class we should predict in this region.
"""
