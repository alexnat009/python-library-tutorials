import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("../../datasets/penguins_classification.csv")

# only keep the Adelie and Chinstrap classes
df = df.set_index("Species").loc[["Adelie", "Chinstrap"]].reset_index()
feature_names = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_name = "Species"

X = df[feature_names]
y = df[target_name]
for feature_name in feature_names:
	plt.figure()
	# plot the histogram for each specie
	df.groupby("Species")[feature_name].plot.hist(alpha=0.5, legend=True)

	plt.xlabel(feature_name)

	plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.3f}")

DecisionBoundaryDisplay.from_estimator(
	model,
	X_test,
	response_method="predict",
	cmap="RdBu_r",
	alpha=0.5
)

sns.scatterplot(
	data=pd.concat([X_test, y_test], axis=1),
	x=feature_names[0],
	y=feature_names[1],
	hue=target_name,
	palette=["tab:red", "tab:blue"],
)
plt.title("Decision boundary of the trained\n LogisticRegression")
plt.show()

coefs = model[-1].coef_[0]
weights = pd.Series(coefs, index=[f"Weight for '{c}'" for c in feature_names])
print(weights)
weights.plot.barh()
plt.show()

test_case = pd.DataFrame(
	{"Culmen Length (mm)": [45], "Culmen Depth (mm)": [17]}
)
test_penguin = (model.predict(test_case))
test_penguin_proba = model.predict_proba(test_case)
print(test_penguin_proba)

"""
Similarly to the hard decision boundary shown above, we can set the response_method to "predict_proba"
in the DecisionBoundaryDisplay to rather show the confidence on individual classifications. In such case the boundaries
encode the estimated probabilities by color.

In particular, when using matplotlib diverging colormaps such as "RdBu_r", the softer the color, the more unsure aboute
which class to choose  
"""

DecisionBoundaryDisplay.from_estimator(
	model,
	X_test,
	response_method="predict_proba",
	cmap="RdBu_r",
	alpha=0.5
)

sns.scatterplot(
	data=pd.concat([X_test, y_test], axis=1),
	x=feature_names[0],
	y=feature_names[1],
	hue=target_name,
	palette=["tab:red", "tab:blue"]
)
plt.title("Predicted probability of the trained\n LogisticRegression")
plt.show()
