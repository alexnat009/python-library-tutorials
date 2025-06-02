import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import X_test, X_train, y_test, y_train, y, X_with_rnd_feat, train_dataset

"""
We will detail methods to investigate the importance of features used by a given model.

	○ Interpreting the coefficients on a linear model;
	○ The attribute feature_importance_ in RandomForest;
	○ Permutation feature importance, which is an inspection technique that can be used for any fitted model
"""

sns.pairplot(
	train_dataset[
		["MedHouseVal", "Latitude", "AveRooms", "AveBedrms", "MedInc"]
	],
	kind="reg",
	diag_kind="kde",
	plot_kws={"scatter_kws": {"alpha": 0.1}},
)
plt.show()

"""
We see in the upper right plot the median income seems to be positively correlated to the median house price (target)

We can also see that the average number of rooms AveRooms is very correlated to the average number of bedrooms AveBedrms
"""

model = RidgeCV()
model.fit(X_train, y_train)

print(f"model score on training data: {model.score(X_train, y_train)}")
print(f"model score on testing data: {model.score(X_test, y_test)}")

"""
Our linear model obtains a R^2 score of 0.6, so it explains a significant part of the target. Its 
coefficient should be somehow relevant.
"""
coefs = pd.DataFrame(
	model.coef_, columns=["Coefficients"], index=X_train.columns
)

coefs.plot(kind="barh", figsize=(9, 7))
plt.title("Ridge model")
plt.rcParams['figure.constrained_layout.use'] = True
plt.axvline(x=0, color=".5")
plt.show()

"""
1) Sign of coefficients

The coefficient of a linear model are a conditional association: they quantify the variation of 
the output (the price) when the given feature is varied, keeping all other features constant. We shouldn't
interpret them as a marginal association, characterizing the link between the two quantities ignoring all the rest

The coefficients associated to AveRooms is negative because the number of rooms is strongly correlated with the
number of bedrooms, AveBedrms. What we are seeing here is that for districts where the house have the same number
of bedrooms on average, when there are more rooms (hence non-bedroom rooms_, the houses are worth comparatively less 
"""

"""
2) Scale of coefficients

The AveBedrms have the higher coefficient. However, we can't compare the magnitude of these coefficients directly,
since they are not scaled. Indeed, Population is an integer which can be thousands, while AveBedrms is around 4 and
Latitude is in degree

So the Popular coefficient is expressed in "100k$/ habitant" while the AveBedrms is expressed in "100k$/nb of bedrooms"
and the Latitude coefficient "100k$/degree"

We see that changing population by one doesn't change the outcome, while as we go south (latitude increase) the price
becomes cheaper. Also, adding a bedroom (keeping all other feature constant) shall rise the price of the house by 80k$

So looking at the coefficient plot to gauge feature importance can be misleading as some of them vary on a small scale,
while other vary a lot more, several decades.

This becomes visible if we compare the standard deviations of our different features
"""

X_train.std(axis=0).plot(kind="barh", figsize=(9, 7))
plt.title("Features std. dev.")
plt.rcParams['figure.constrained_layout.use'] = True
plt.xlim((0, 100))
plt.show()

"""
So before any interpretation, we need to scale each column (removing the mean and scaling the variance to 1)
"""

model = make_pipeline(StandardScaler(), RidgeCV())
model.fit(X_train, y_train)
print(f"model score on training data: {model.score(X_train, y_train)}")
print(f"model score on testing data: {model.score(X_test, y_test)}")

coefs = pd.DataFrame(
	model[1].coef_, columns=["Coefficients"], index=X_train.columns
)

coefs.plot(kind="barh", figsize=(9, 7))
plt.title("Ridge model")
plt.axvline(x=0, color=".5")
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

"""
Now that the coefficients have been scaled, we can safely compare them

The median income feature, with longitude and latitude are the three variables that most influence the model

The plot above tells us about dependencies between a specific feature and the target when all other features remain
constant, i.e. conditional dependencies. An increase of the HouseAge will induce an increase of the price when 
all other features remain constant. On the contrary, an increase of the average rooms will induce an decrease of the
price when all other features remain constant
"""

"""
3) Checking the variability of the coefficients
"""
cv = RepeatedKFold(n_splits=5, n_repeats=5)
cv_model = cross_validate(
	model,
	X_with_rnd_feat,
	y,
	cv=cv,
	return_estimator=True,
	n_jobs=3
)

coefs = pd.DataFrame(
	[model[1].coef_ for model in cv_model["estimator"]],
	columns=X_with_rnd_feat.columns,
)
sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
plt.axvline(x=0, color=".5")
plt.xlabel("Coefficient importance")
plt.title("Coefficient importance and its variability")
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

"""
Every coefficient looks pretty stable, which mean that different Ridge model put almost the same
weight to the same feature
"""

"""
4) Linear models with sparse coefficients (LASSO)

It is important to keep in mind that the associations extracted depend on the model. To illustrate this
point we consider a Lasso model, that performs feature selection with L1 penalty.
"""

model = make_pipeline(StandardScaler(), Lasso(alpha=0.015))

model.fit(X_train, y_train)
print(f"model score on training data: {model.score(X_train, y_train)}")
print(f"model score on testing data: {model.score(X_test, y_test)}")

coefs = pd.DataFrame(
	model[1].coef_, columns=["Coefficients"], index=X_train.columns
)

coefs.plot(kind="barh", figsize=(9, 7))
plt.title("Lasso model, strong regularization")
plt.axvline(x=0, color=".5")
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

"""
Here the model score is a bit lower, because of the strong regularization. However, it has
zeroed out 3 coefficients, selecting a small number of variables to make its predictions.

We can see that out of the two correlated features AveRooms and AveBedrms, the model has selected one.
Note that this choice is partly arbitrary: choosing one doesn't mean that the other is not important for
prediction. Avoid over-interpreting models, as they are imperfect
"""

cv_model = cross_validate(
	model,
	X_with_rnd_feat,
	y,
	cv=cv,
	return_estimator=True,
	n_jobs=3
)

coefs = pd.DataFrame(
	[model[1].coef_ for model in cv_model["estimator"]],
	columns=X_with_rnd_feat.columns,
)

sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
plt.axvline(x=0, color=".5")
plt.xlabel("Coefficient importance")
plt.title("Coefficient importance and its variability")
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

"""
We can see that both the coefficients associated to AveRooms and AveBedrms have a strong variability and that
they can both be non zero. Given that they are strongly correlated, the model can pick one or the
other to predict well. This choice is a bit arbitrary, and must not be over-interpreted 
"""

"""
TAKE-AWAY

○ Coefficients must be scaled to the same unit of measurements to retrieve feature importance, or comparing them

○ Coefficients in multivariate linear models represent the dependency between a given feature and the target,
conditional on the other features

○ Correlated features might induce instabilities in the coefficients of linear models and their effects
cannot be well teased apart

○ Inspecting coefficients across the folds of a cross validation loop gives an idea of their stability
"""
