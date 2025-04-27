import pandas as pd
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("../../datasets/adult-census.csv")

target_class = "class"
X = df.drop(columns=["education-num", target_class])
y = df[target_class]

categorical_columns_selector = selector(dtype_include=object)

categorical_columns = categorical_columns_selector(X)
categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer([
	("categorical_preprocessor", categorical_preprocessor, categorical_columns)
], remainder="passthrough")

model = Pipeline([
	("preprocessor", preprocessor),
	("classifier", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))
])

cv_results = cross_validate(model, X, y, cv=5)
cv_results = pd.DataFrame(cv_results)

print(
	"Generalization score without hyperparameters"
	f" tuning:\n{cv_results['test_score'].mean():.3f} ±"
	f" {cv_results['test_score'].std():.3f}"
)

param_grid = {
	"classifier__learning_rate": (0.05, 0.5),
	"classifier__max_leaf_nodes": (10, 30)
}
model_grid_search = GridSearchCV(model, param_grid, n_jobs=2, cv=2)
model_grid_search.fit(X, y)

print(model_grid_search.best_params_)

"""
CAVEAT:

The mean and std of the scores computed by the cross-validation in the grid-search are
potentially not good estimates of the generalization performance we would obtain by refitting
a model with the best combination of hyper-parameter values on the full dataset

We therefore used knowledge from the full dataset to both decide our model's hyperparameters and
to train the refitted model

Because of this one must keep an external, held-out set for the final evaluation of the refitted model
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_grid_search.fit(X_train, y_train)

accuracy = model_grid_search.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.3f}")

"""
The score measure on the final test set is almost within the range of the internal CV score for the
best hyperparameter combination. This is reassuring as it means that the tuning procedure didn't cause significant
overfitting in itself. That is expected because our grid search explored very few hyperparameter combinations for the
sake of speed.

The test score of the final model is actually a bit higher than what we could have expected from the internal 
cross-validation. This is expected too since the refitted model is trained on a larger dataset than the models 
evaluated in the internal CV loop of the grid-search procedure

This is often the case that models trained on a larger number of samples tend to generalize better  
"""

"""
However, this evaluation only provides us a single point estimate of the generalization performance. It is beneficial 
to have a rough idea of the uncertainty of our estimated generalization performance. Therefore, we should instead use an 
additional cross-validation for this evaluation.
 
This pattern in called nested cross-validation. We use an inner cross-validation for the selection of the hyperparameters
and an outer cross-validation for the evaluation of generalization performance of the refitted tuned model
"""
cv_results = cross_validate(
	model_grid_search, X, y, cv=5, n_jobs=2, return_estimator=True
)

cv_results = pd.DataFrame(cv_results)
cv_test_scores = cv_results["test_score"]
print(
	"Generalization score with hyperparameters tuning:\n"
	f"{cv_test_scores.mean():.3f} ± {cv_test_scores.std():.3f}"
)

for cv_fold, estimator_in_fold in enumerate(cv_results["estimator"]):
	print(
		f"Best hyperparameters for fold #{cv_fold + 1}:\n"
		f"{estimator_in_fold.best_params_}"
	)

"""
It is interesting to see whether the hyper-parameter tuning procedure always
selects similar values for the hyperparameters. If it's the case, then all is fine. 
It means that we can deploy a model fit with those h-parameters and expect that it will have an actual
predictive performance close to what we measured in the outer cross-validation.

But is is also possible that some hyperparameters do not matter at all, and as a result in different tuning
sessions give different results. In this case, any value will do. This can typically be confirmed by doing
a parallel coordinate plot of the results of a large hyperparameter search
"""