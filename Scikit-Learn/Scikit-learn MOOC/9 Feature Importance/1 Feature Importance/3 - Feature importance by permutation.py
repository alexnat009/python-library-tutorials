import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import X_test, X_train, y_test, y_train, X, y, X_with_rnd_feat, train_dataset

# Any model could be used here
model = RandomForestRegressor()
# model = make_pipeline(StandardScaler(), RidgeCV())
model.fit(X_train, y_train)
print(f"model score on training data: {model.score(X_train, y_train)}")
print(f"model score on testing data: {model.score(X_test, y_test)}")

"""
Feature importance

Lets compute the feature importance for a given feature, sat the MedInc feature

For that, we will shuffle this specific feature, keeping the other feature as is, and run over
same model (already fitted) to predict the outcome. The decrease of the score shall indicate how the model had used
this feature to predict the target. The permutation feature importance is defined to be the decrease in a model score
when a single feature value is randomly shuffled

For instance, if the feature is crucial for the model, the outcome would also be permuted (just as the feature), thus
the score would be cloze to zero. Afterward, the feature importance is the decrease in score. So in that case, the
feature importance would be close to the score.

On the contrary, if the feature is not used by the model, the score shall remain the same, thus the feature importance
will be close to 0 
"""


def get_score_after_permutation(model, X, y, curr_feat):
	"""
	return the score of model when curr_feat is permuted
	"""

	X_permuted = X.copy()
	col_idx = list(X.columns).index(curr_feat)
	# permute one column
	X_permuted.iloc[:, col_idx] = np.random.permutation(
		X_permuted[curr_feat].values
	)

	permuted_score = model.score(X_permuted, y)
	return permuted_score


def get_feature_importance(model, X, y, curr_feat):
	baseline_score_train = model.score(X, y)
	permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)

	feature_importance = baseline_score_train - permuted_score_train
	return feature_importance


curr_feat = "MedInc"

feature_importance = get_feature_importance(model, X_train, y_train, curr_feat)
print(
	f'feature importance of "{curr_feat}" on train set is '
	f"{feature_importance:.3}"
)

"""
Since there is some randomness, it is advisable to run it multiple times and inspect the mean and 
the standard deviation of the feature importance
"""

n_repeats = 10

list_feature_importance = []
for n in range(n_repeats):
	list_feature_importance.append(get_feature_importance(model, X_train, y_train, curr_feat))

print(
	f'feature importance of "{curr_feat}" on train set is '
	f"{np.mean(list_feature_importance):.3} "
	f"± {np.std(list_feature_importance):.3}"
)

"""
0.76 over 0.98 is very relevant. So we can imagine our model relies heavily on this feature to predict
the class. We can now compute the feature permutation importance for all the features
"""


def permutation_importance_(model, X, y, n_repeats=10):
	importances = []
	for curr_feat in X.columns:
		list_feature_importance = []
		for n in range(n_repeats):
			list_feature_importance.append(
				get_feature_importance(model, X, y, curr_feat)
			)
		importances.append(list_feature_importance)
	return {
		"importances_mean": np.mean(importances, axis=1),
		"importances_std": np.std(importances, axis=1),
		"importances": importances,
	}


def plot_feature_importance(perm_importance_result, feat_name):
	indices = perm_importance_result["importances_mean"].argsort()
	plt.barh(
		range(len(indices)),
		perm_importance_result["importances_mean"][indices],
		xerr=perm_importance_result["importances_std"][indices],
	)
	plt.rcParams['figure.constrained_layout.use'] = True

	plt.yticks(range(len(indices)), labels=feat_name[indices])
	plt.show()


perm_importance_result_train = permutation_importance(model, X_train, y_train, n_repeats=10)
plot_feature_importance(perm_importance_result_train, X_train.columns)

"""
We see again that the feature MedInc, Latitude and Longitude are very important for th emodel.

We note that our random variable rnd_num is now very less important that latitude. Indeed, the feature importance built-in
RandomForest has bias for continuos data, such as AceOccup and rnd_num 

However, the model still uses these rnd_num feature to compute the output. It is in line with the overfitting we had 
noticed between the train and test score
"""
