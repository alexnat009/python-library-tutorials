import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../datasets/penguins_classification.csv")

df = df.set_index("Species").loc[["Adelie", "Chinstrap"]].reset_index()

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"
X = df[culmen_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.4)


def plot_decision_boundary(model):
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)
	C = model.get_params()["logisticregression__C"]

	disp = DecisionBoundaryDisplay.from_estimator(
		model,
		X_train,
		response_method="predict_proba",
		plot_method="pcolormesh",
		cmap="RdBu_r",
		alpha=0.9,
		vmin=0.0,
		vmax=1.0
	)
	DecisionBoundaryDisplay.from_estimator(
		model,
		X_train,
		response_method="predict_proba",
		plot_method="contour",
		linestyles="--",
		linewidths=1,
		alpha=0.8,
		levels=[0.5],
		ax=disp.ax_
	)
	sns.scatterplot(
		data=pd.concat([X_train, y_train]),
		x=culmen_columns[0],
		y=culmen_columns[1],
		hue=target_column,
		palette=["tab:blue", "tab:red"],
		ax=disp.ax_
	)
	plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
	plt.title(f"C: {C} \n Accuracy on the test set: {accuracy:.2f}")


lr = make_pipeline(StandardScaler(), LogisticRegression())

Cs = [1e-6, 0.01, 0.1, 1, 10, 100, 1e6, 1e10]

# for C in Cs:
# 	lr.set_params(logisticregression__C=C)
# 	# plot_decision_boundary(lr)
# 	# plt.show()
# 	print(f'C={C}, coefficients={lr["logisticregression"].coef_}')

"""
OBSERVATIONS:

○ For low values of C (strong regularization). the classifier is less confident in its
predictions. We are enforcing a spread sigmoid

○ For high values of C (weak regularization). The classifier is more confident: the areas with dark
blue (very confident in predicting "Adelie") and dark red (very confident in predicting "Chinstrap") nearly cover
the entire feature space. We are enforcing a steep sigmoid

○ The smaller the C  (stronger regularization), the lower the cost of a misclassification. As more data points lay 
in the low-confidence zone, the more the decision rules are influenced almost uniformly by all the data points. This
leads to a less expressive model, which may underfit

○ The higher the C (weaker regularization), the more the decision is influenced by a few training points very close 
to the boundary, where decisions are costly. Remember that the model may overfit if the number of samples in the
training set is too small, as at least a minimum of samples is needed to average the noise out.


The orientation is the result of two factors: minimizing the number of misclassified training points with high confidence
and their distance to the decision boundary.

Finally, for small values of C the position of the decision boundary is affected by the class imbalance: when C is near 
zero, the model predicts the majority class everywhere in the feature space. In our case, there are approximately two 
times more "Adelie" than "Chinstrap" penguins. This explains why the decision boundary is shifted to the right when C 
gets smaller. Indeed, the most regularized model predicts light blue almost everywhere in the feature space  

"""

lr_weights = []
for C in Cs:
	lr.set_params(logisticregression__C=C)
	lr.fit(X_train, y_train)
	coefs = lr[-1].coef_[0]
	lr_weights.append(pd.Series(coefs, index=culmen_columns))

lr_weights = pd.concat(lr_weights, axis=1, keys=[f"C: {C}" for C in Cs])
lr_weights.plot.barh()
plt.title("Logistic regression weights depending of C")
plt.show()

"""
As small C provides a more regularized model, it shrinks the weight values toward zero, as in the Ridge model

In particular, with a strong penalty, the weight of the features named "Culmen Depth (mm)" is almost zero. it explains
why the decision separation in the plot is almost perpendicular to the "Culmen Length (mm)" feature

For even stronger penalty strengths (e.g. C=1e6) the weights of both features are almost zero. It explains
why the decision separation in the plot is almost constant in the feature space: the prediction probability is only
based on the intercept parameter of the model
"""

lr_with_nystroem = make_pipeline(
	StandardScaler(),
	Nystroem(kernel="rbf", gamma=1, n_components=100),
	LogisticRegression()
)


for C in Cs:
	lr_with_nystroem.set_params(logisticregression__C=C)
	plot_decision_boundary(lr_with_nystroem)
	plt.show()
	print(f'C={C}, coefficients={lr["logisticregression"].coef_}')


"""
OBSERVATIONS:

○ For the lowest values of C, the overall pipeline underfits: it predicts the majority class everywhere, as previously

○ When C increases, the model starts to predict some datapoints from the "Chinstrap" class but the model is not very
confident anywhere in the feature space

○ The decision boundary is no longer a straight line: the linear model is now classifying in 100-dimensional feature
space created by the Nystroem transformer. As a result, the decision boundary induced by the overall pipeline is now 
expressive enough to wrap around the minority class

○ For C = 1 in particular, it finds a smooth red blob around most of the "Chinstrap" data points. When moving away from
the data points, the model is less confident in its predictions and again tends to predict the majority class according
to the proportion in the training set

○ For higher values of C, the models starts to overfit: it is very confident in its predictions almost everywhere, but 
it should not be trusted. the model also makes a larger number of mistakes on the test set while adopting a very curvy
decision boundary to attempt fitting all the training points, including the noisy ones at the frontier between the two
classes. This makes the decision boundary very sensitive to the sampling of the training set and as a result, it doesn't
generalize well in the region. This is confirmed by the lower accuracy on the test set


Finally we can also note that the linear model on the raw features was as good or better than the best model using
non-linear feature engineering. So in this case, we didn't really need this extra complexity in out pipeline.
Simples is better
 
"""