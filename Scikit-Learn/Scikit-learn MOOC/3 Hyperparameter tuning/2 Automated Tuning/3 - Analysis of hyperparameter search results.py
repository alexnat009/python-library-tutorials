import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

cv_results = pd.read_csv("../../figures/randomized_search_results.csv")


def shorten_param(param_name):
	if "__" in param_name:
		return param_name.rsplit("__", 1)[1]
	return param_name


cv_results = cv_results.rename(shorten_param, axis=1)

df = pd.DataFrame(
	{
		"max_leaf_nodes": cv_results["max_leaf_nodes"],
		"learning_rate": cv_results["learning_rate"],
		"score_bin": pd.cut(cv_results["mean_test_score"], bins=np.linspace(0.5, 1.0, 6)),
	}
)

sns.set_palette("YlGnBu_r")
ax = sns.scatterplot(
	data=df,
	x="max_leaf_nodes",
	y="learning_rate",
	hue="score_bin",
	s=50,
	color="k",
	edgecolor=None,
)
ax.set_xscale("log")
ax.set_yscale("log")

ax.legend(title="mean_test_score", loc="center left", bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(right=0.7)
plt.show()

"""
We see that top performing values are located in a band of learning rate [0.01 : 1.0],
but we have no control in how the other hyperparameters interact with such values for the
learning rate. Instead we can visualize all the hyperparameters at the same time using a parallel coordinates
plot
"""
cv_results_shortend = cv_results.rename(shorten_param, axis=1)
transformed = cv_results_shortend.apply(
	# Transform axis values by taking a log10 or log2 to spread the active
	# ranges and improve the readability of the plot
	{
		"learning_rate": np.log10,
		"max_leaf_nodes": np.log2,
		"max_bins": np.log2,
		"min_samples_leaf": np.log10,
		"l2_regularization": np.log10,
		"mean_test_score": lambda x: x,
	}
)
fig = px.parallel_coordinates(
	data_frame=transformed,
	color="mean_test_score",
	color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()

"""
Learning Rate:
It is interesting to confirm that the yellow lines (top performing models) all reach intermediate values
of the learning rate, that is, between -2 and 0, which corresponds to learning rate values of 0.01 to 1.0 
"""

"""
Max Bins:
Now we can also observe that it's not possible to select the highest performing models by selecting lines
of on the max_bins axis between 1 and 3
"""

"""
Other hyperparameters:
The other hyperparameters are not very sensitive. We can check if we select the
learning_rate: [0.01 : 1.0] and max_bins: [2^5 : 2^8], we always select top performing models,
whatever the values of the other hyperparameters
"""
