import matplotlib.axes._axes as axes
import matplotlib.figure as figure

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, \
	balanced_accuracy_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

"""
ML models rely on optimizing an objective function, by seeking its minimum or maximum. It is important
to understand that this objective function is usually decoupled from the evaluation metric that we want
to optimize in practice. The objective function serves as a proxy for the evaluation metric.
"""

df = pd.read_csv("../../datasets/blood_transfusion.csv")
X = df.drop(columns="Class")
y = df["Class"]

plt.subplots(layout="constrained")
y.value_counts().plot.barh()
plt.xlabel("Number of samples")
plt.title("Number of samples per classes present\n in the target")
plt.show()

"""
We can see that the vector y contains two classes corresponding to whether a subject gave blood.
We will use a logistic regression classifier to predict this outcome

To focus on the metrics presentation, we will only use a single split instead of cross-validation
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0, test_size=0.5)

lr = LogisticRegression()
lr.fit(X_train, y_train)

"""
Classifier predictions

before we go into details regarding the metrics, we will recall what type of predictions a classifier can provide

For this reason, we will create a synthetic sample for a new potential donor 
"""

new_donor = pd.DataFrame(
	{
		"Recency": [6],
		"Frequency": [2],
		"Monetary": [1000],
		"Time": [20],
	}
)

print(lr.predict(new_donor))

"""
With this information, our classifier predicts that this synthetic subject is more likely to not 
donate blood again

However, we cannot check whether the prediction is correct (we don't know the true target value). That's
the purpose of the testing set. First, we predict whether a subject will give blood with the help of the trained 
classifier
"""

y_pred = lr.predict(X_test)
print(y_pred[:5])

"""
Now that we have these predictions, we can compare them with the true predictions (sometimes called ground-truth)
which we didn't use until now
"""

print(y_test == y_pred)
print(np.mean(y_test == y_pred))
"""
This measure is called the accuracy. Here. our classifier is 78% accurate at classifying if a subject will
give blood. scikit-learn provides a function that computes this metric in the module sklearn.metrics
"""

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

"""
Logistic regression also has a method named score which computes the accuracy score
"""

print(lr.score(X_test, y_test))

"""
Confusion matrix and derived metrics

The comparison that we did above and the accuracy that we calculated didn't take into account the type of error
our classifier was making. Accuracy is a n aggregate of the errors made by the classifier. We may be interested in 
fined granularity - to know independently what the error is for each of the two following cases:

	○ We predicted that a person will give blood but they didn't
	○ We predicted that a person won't give blood but they did
"""
fig, ax = plt.subplots(layout="constrained")
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=ax)
plt.show()

"""
The in-diagonal numbers are related to prediction that were correct while off-diagonal numbers are related
to incorrect predictions (misclassifications). We now know the four types of correct and erroneous predictions

	○ TOP-LEFT: True Positives (TP), correspond to people who gave blood and were predicted as such by the classifier
	○ BOTTOM-RIGHT: True Negatives (TN), correspond to people who didn't gave blood and were predicted as such by the
					classifier
	
	
	○ TOP-RIGHT: False Negatives (FN), correspond to people who gave blood but were predicted to not have given blood 
	○ BOTTOM-LEFT: False Positives (FP), correspond to people who didn't gave blood but were predicted to have given 
					blood



Once we have split this information, we can compute metrics to highlight the generalization performance of our 
classifier in a particular setting. For instance, we could be interested in the fraction of people who really gave
blood when the classifier predicted so or the fraction of people predicted to have given blood out of the total 
population that actually did so

The former metric, known as the precision, is defined as TP / (TP + FP) and represents how likely the person actually
gave blood when the classifier predicted that they did. The latter, known as the recall, defined as TP / (TP + FN) and
assesses how well the classifier is able to correctly identify people who did give blood. We could, similarly to 
accuracy, manually compute these values, however scikit-learn provides function to compute these statistics.

Precision : TP / (TP + FP)

Recall: TP / (TP + FN)
"""

precision = precision_score(y_test, y_pred, pos_label="donated")
recall = recall_score(y_test, y_pred, pos_label="donated")
print(f"Precision score: {precision:.3f}")
print(f"Recall score: {recall:.3f}")

"""
The issue of class imbalance

At this stage, we could ask ourself a reasonable question. While the accuracy didn't look bad, the recall score
is relatively low

As we mentioned, precision and recall only focuses on samples predicted to be positive, while accuracy takes both into
account. In addition, we didn't look at the ration of classes (labels). We could check this ration in the training set
"""

plt.subplots(layout="constrained")
y_train.value_counts(normalize=True).plot.barh()
plt.xlabel("Class frequency")
plt.title("Class frequency in the training set")
plt.show()

"""
We observe that the positive class, donated comprises only 24% of the samples. The good accuracy of our 
classifier is then linked to its ability to correctly predict the negative class "not donated" which may or may not 
be relevant, depending on the application. We can illustrate the issue using a dummy classifier as a baseline
"""

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print(
	"Accuracy of the dummy classifier: "
	f"{dummy.score(X_test, y_test):.3f}"
)

"""
With the dummy classifier, which always predict the negative class "not donated" we obtain an accuracy score of 76%.
Therefore, it mean that this classifier, without learning anything from the data, is capable of predicting as 
accurately as our logistic regression model


The problem illustrated above is also know as the class imbalance problem. When the classes are imbalanced, accuracy
shouldn't be used. In this case, one should either use the precision and recall as presented above or the balanced
accuracy score instead of accuracy
"""

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced accuracy: {balanced_accuracy:.3f}")
"""
The balanced accuracy is equivalent to accuracy in the context of balanced classes. It is defined as the average recall
obtained on each class
"""

"""
Evaluation and different probability threshold

All statistics that we presented up to now rely on classifier.predict which outputs the most likely label.
W haven't made use of the probability associated with this prediction, which gives the confidence of the
classifier in this prediction. By default, the prediction of a classifier corresponds to a threshold of 0.5 probability
in a binary classification problem. We can quickly check this relationship with the classifier that we trained
"""

target_proba_predicted = pd.DataFrame(
	lr.predict_proba(X_test), columns=lr.classes_
)

print(target_proba_predicted[:5])
print(y_pred[:5])

equivalence_pred_proba = (
		target_proba_predicted.idxmax(axis=1).to_numpy() == y_pred
)
print(np.all(equivalence_pred_proba))
print(pd.concat(
	[target_proba_predicted[:5], pd.Series(y_pred[:5], name="predicted_label", index=target_proba_predicted.index[:5])],
	axis=1
))

"""
The default decision threshold (0.5) might not be the best threshold that leads to optimal generalization
performance of our classifier. In this case, one can vary the decision threshold, and therefore the underlying
prediction, and compute the same statistics presented earlier. Usually, the two metrics recall and precision are
computed and plotted on a graph. Each metric plotted on a graph axis and each point on the graph corresponds to a 
specific decision threshold.
"""
fig, ax = plt.subplots(layout="constrained")
PrecisionRecallDisplay.from_estimator(
	lr, X_test, y_test, pos_label="donated", marker="+", ax=ax
)
PrecisionRecallDisplay.from_estimator(
	dummy,
	X_test,
	y_test,
	pos_label="donated",
	color="tab:orange",
	linestyle="--",
	ax=ax
)

plt.xlabel("Recall (also known as TPR or sensitivity)")
plt.ylabel("Precision (also known as PPV)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
ax.set_title("Precision-recall curve")
plt.show()

"""
On this graph 
X-axis: Recall (TPR) = TP / (TP + FN)
Y-axis: Precision = TP / (TP + FP)
Why it's useful:
	Better for imbalanced datasets than ROC
	Focuses only on the positive class performance
"""

"""
On this curve, each blue cross corresponds to a level of probability which we used as a decision threshold.
We can see that, by varying this decision threshold, we get different precision vs recall values

A perfect classifier would have a precision of 1 for all recall values. A metric characterizing the curve is 
linked to the area under the curve (AUC) and is named average precision (AP). With an ideal classifier, the average
precision would be 1.

Notice that the AP if a DummyClassifier, used as baseline to define the chance level, coincides with the number 
of samples in the positive class divided by the total number of samples
"""

prevalence = (
		y_test.value_counts()["donated"] / y_test.value_counts().sum()
)
print(f"Prevalence of the class 'donated': {prevalence:.2f}")

"""
The precision and recall metric focuses on the positive class, however, one might be interested in 
the compromise between accurately discriminating the positive class and accurately discriminating the
negative classes. The statistics used for this are sensitivity and specificity. Sensitivity is just another name for
recall. However, specificity measures the proportion of correctly classified samples in the negative class defined
as: TN / (TN + FP). Similar to the precision-recall curve, sensitivity and specificity are generally plotted
as a curve called the Receiver Operating Characteristic (ROC) curve. Below is such a curve  
"""

fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')  # type:figure.Figure, axes.Axes
RocCurveDisplay.from_estimator(
	lr, X_test, y_test, pos_label="donated", marker="+", ax=ax
)
RocCurveDisplay.from_estimator(
	dummy,
	X_test,
	y_test,
	pos_label="donated",
	color="tab:orange",
	linestyle="--",
	ax=ax
)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate\n(also known as sensitivity or recall)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
ax.set_title("Receiver Operating Characteristic curve")
plt.show()

"""
This curve was built using the same principle as the precision-recall curve:
we vary the probability threshold for determining "hard" prediction and compute the
metrics. As with the precision-recall curve, we can compute the area under the ROC (ROC-AUC)
to characterize the generalization performance of our classifier. However, it is important to observe
that the lower bound of the ROC-AUC is 0.5. Indeed, we shod the generalization performance of a dummy
classifier to show that even the worst generalization performance obtained will be above this line 
"""

"""
Instead of using a dummy classifier, we can use the parameter plot_chance_level available in the ROC and PR 
displays
"""

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), layout='constrained')  # type:figure.Figure, axes.Axes
PrecisionRecallDisplay.from_estimator(
	lr,
	X_test,
	y_test,
	pos_label="donated",
	marker="+",
	plot_chance_level=True,
	chance_level_kw={"color": "tab:orange", "linestyle": "--"},
	ax=ax[0]
)
RocCurveDisplay.from_estimator(
	lr,
	X_test,
	y_test,
	pos_label="donated",
	marker="+",
	plot_chance_level=True,
	chance_level_kw={"color": "tab:orange", "linestyle": "--"},
	ax=ax[1],
)
fig.suptitle("PR and ROC curves")
plt.show()
