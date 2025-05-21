import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

"""
Random forests are a popular model in ML. They are modification of the bagging algorithm. In bagging, any 
classifier or regressor can be used. In random forests, the base classifier or regressor is always a decision tree


Random forests have another particularity: When training a tree, the search for the best split is done only on a subset
of the original features taken at random. The random subsets are different for each split node. The goal is to inject
additional randomization into the learning procedure to try to decorrelate the prediction errors of the individual trees


Therefore, random forests sare using randomization on both axes of the data matrix:
○ by bootstrapping samples for each tree in the forest;
○ randomly selecting a subset of features at each node of the tree 
"""

df = pd.read_csv("../../datasets/adult-census.csv")
target_name = "class"
X = df.drop(columns=[target_name, "education-num"])
y = df[target_name]

categorical_encoder = OrdinalEncoder(
	handle_unknown="use_encoded_value", unknown_value=-1
)

preprocessor = ColumnTransformer([
	("categorical_preprocessor", categorical_encoder, make_column_selector(dtype_include=object))
], remainder="passthrough")

tree = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=0))
tree_scores = cross_val_score(tree, X, y)
print(
	"Decision tree classifier: "
	f"{tree_scores.mean():.3f} ± {tree_scores.std():.3f}"
)

bagged_trees = make_pipeline(
	preprocessor,
	BaggingClassifier(
		estimator=DecisionTreeClassifier(random_state=0),
		n_estimators=20,
		n_jobs=2,
		random_state=0
	)
)
bagged_trees_scores = cross_val_score(bagged_trees, X, y)
print(
	"Bagged decision tree classifier: "
	f"{bagged_trees_scores.mean():.3f} ± {bagged_trees_scores.std():.3f}"
)

random_forest = make_pipeline(
	preprocessor,
	RandomForestClassifier(n_estimators=50, n_jobs=2, random_state=0)
)
random_forest_scores = cross_val_score(random_forest, X, y)
print(
	"Random forest classifier: "
	f"{random_forest_scores.mean():.3f} ± "
	f"{random_forest_scores.std():.3f}"
)



"""
Details about default hyperparameters

For random forests, it it possible to control the amount of randomness for each split by setting the value
of max_features hyperparameter

○ max_features=0.5 means that 50% of the features are considered at each split
○ max_features=1.0 means that all features are considered at each split which effectively disable feature subsampling


by default, RandomForestRegressor disables feature subsampling while RandomForestClassifier uses
max_features=np.sqrt(n_features). These default values reflect good practices given in the scientific literature

However, max_features is one of the hyperparameters to consider when tuning a random forest:

○ too moch randomness in the trees can lead to underfitting base models and can be detrimental for the ensemble as a
whole

○ too few randomness in the trees lead to more correlation of the prediction errors and as a result reduce the benefits
of the averaging step in terms of overfitting control

We summarize these details in the following table:

Ensemble model class		Base model class			Default value for max_features		Features subsampling strategy
-------------------------------------------------------------------------------------------------------------------------
BaggingClassifier			User specified (flexible)	n_features (no subsampling)			Model Level

RandomForestClassifier		DecisionTreeClassifier		sqrt(n_features)					Tree Node Level

BaggingRegressor			User specified (flexible)	n_features (no subsampling)			Model Level

RandomForestRegressor		DecisionTreeRegressor		n_features (no_subsampling)			Tree Node Level
"""