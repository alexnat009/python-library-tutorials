import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, LeaveOneGroupOut, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

"""
In ML, it is quite common to assume that the data are i.i.d meaning that the generative provess
doesn't have any memory of past samples to generate new samples

i.i.d stand for "independent and identically distributed"

This assumption is usually violated in time series, where each sample can be influenced by previous samples in an
inherently ordered sequence.

In this notebook we demonstrate the issues that arise when using the cross-validation strategies we 
have presented so far, along with non-i.i.d. data. For such purpose we load financial quotations from some 
energy companies  
"""

symbols = {
	"TOT": "Total",
	"XOM": "Exxon",
	"CVX": "Chevron",
	"COP": "ConocoPhillips",
	"VLO": "Valero Energy"
}

template_name = "../../datasets/financial-data/{}.csv"
quotes = {}
for symbol in symbols:
	data = pd.read_csv(
		template_name.format(symbol),
		index_col=0,
		parse_dates=True
	)
	quotes[symbols[symbol]] = data["open"]

quotes = pd.DataFrame(quotes)

quotes.plot()
plt.ylabel("Quote value")
plt.legend(loc="upper left")
plt.title("Stock values over time")
plt.show()

"""
Here, we want to predict the quotation of Chevron using all other energy companies' quotes.
To make explanatory plot, we first use a train-test split and then we evaluate other cross-validation
methods 
"""
X, y = quotes.drop(columns=["Chevron"]), quotes["Chevron"]

"""
We will use a decision tree regressor that we expect to overfit and thus not generalize to
unseen data. We use a ShuffleSplit cross-validation to check the generalization performance
of our model
"""

dtr = DecisionTreeRegressor()
cv = ShuffleSplit(random_state=0)

test_score = cross_val_score(dtr, X, y, cv=cv, n_jobs=2)
print(f"The mean R2 is: {test_score.mean():.2f} ± {test_score.std():.2f}")

"""
Surprisingly, we get outstanding generalization performance. We will investigate and find
reason for such good result with a model that is expected to fail. We previously mentioned 
that ShuffleSplit is a cross-validation method that iteratively shuffles and splits the data

We can simplify the procedure with a single split and plot the prediction. We can use 
train_test_split for this purpose
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)

# Shuffling breaks the index order, but we still want it to be time-ordered
X_train.sort_index(ascending=True, inplace=True)
X_test.sort_index(ascending=True, inplace=True)
y_train.sort_index(ascending=True, inplace=True)
y_test.sort_index(ascending=True, inplace=True)

dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

# Recover the `DatetimeIndex` from `target_test` for correct plotting
y_pred = pd.Series(y_pred, index=y_test.index)

test_score = r2_score(y_test, y_pred)
print(f"The R2 on this single split is: {test_score:.2f}")

y_train.plot(label="training")
y_test.plot(label="testing")
y_pred.plot(label="prediction")

plt.ylabel("Quote value")
plt.legend(loc="upper left")
plt.title("Model predictions using a ShuffleSplit strategy")
plt.show()

"""
From the plot above, we can see that the training and testing samples are alternating. This structure
effectively evaluates the model's ability to interpolate between neighboring data points, rather than its
true generalization ability. As a result, the model's predictions are close to the actual values, even if it
has not learned anything meaningful from the data/ This is a form of DATA LEAKAGE, where the model gains
access to future information (testing data) while training, leading to an over-optimistic estimate of the
generalization performance.

An easy way to verify this is to not shuffle the data during the split. In this case, we will use the first 75%
of the data to train and the remaining data to test. This way we preserve the time order of the data, and unsure 
training on past data and evaluating on future data
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)

test_score = r2_score(y_test, y_pred)
print(f"The R2 on this single split is: {test_score:.2f}")

"""
In this case, we see that our model is not magical anymore. Remember that a negative R^2 means
that the regressor performs worse than always prediction the mean of the target. We can visually check what we are
predicting as follows
"""

y_train.plot(label="training")
y_test.plot(label="testing")
y_pred.plot(label="prediction")

plt.ylabel("Quote value")
plt.legend(loc="upper left")
plt.title("Model predictions using a split without shuffling")
plt.show()

"""
We see that our model cannot predict anything because it doesn't have samples around the 
testing sample. Let's check how we could have made a proper cross-validation scheme to get a reasonable
generalization performance estimate

One solution would be to group the samples into time block, by quarter, and predict each group's information by using 
information from the other groups. We can use the LeaveOneGroupOut cross-validation for this purpose. 
"""

groups = quotes.index.to_period("Q")
cv = LeaveOneGroupOut()
test_score = cross_val_score(
	dtr,
	X,
	y,
	cv=cv,
	groups=groups,
	n_jobs=2
)
print(f"The mean R2 is: {test_score.mean():.2f} ± {test_score.std():.2f}")

"""
In this case, we see that we cannot make good predictions, which is less surprising than our original results.

Another thing o consider is the actual application of our solution. If our model is aimed at forecasting, we should
not use training data that are ulterior to the testing data. In this case, we can use the TimeSeriesSplit cross-validation
to enforce this behaviour.
"""

cv = TimeSeriesSplit(n_splits=groups.nunique())
test_score = cross_val_score(
	dtr,
	X,
	y,
	cv=cv,
	n_jobs=2
)
print(f"The mean R2 is: {test_score.mean():.2f} ± {test_score.std():.2f}")
"""
In conclusion, it is really important not to carelessly use a cross-validation strategy which do not respect some
assumptions such as having i.i.d. data. It might lead to misleading outcomes, creating the false impression that a
predictive model performs well when it may not be the case in the intended real-world scenario

"""