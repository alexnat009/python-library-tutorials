from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

df = load_digits()
X, y = df.data, df.target

"""
Here we use a MinMaxScaler as we know that each pixel's gray-scale is strictly bounded between [0:16]
This makes MinMaxScaler more suited in this case than StandardScaler, as some pixels consistently have 
low variance (pixels at the borders might almost always be zero of most digits are centered in the image.
Then using, StandardScaler can result in a very high scaled value due to division by a small number

"""

model = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000))

cv = KFold(shuffle=False)
test_score_no_shuffling = cross_val_score(model, X, y, cv=cv, n_jobs=2)
print(
	"The average accuracy without shuffling is "
	f"{test_score_no_shuffling.mean():.3f} ± "
	f"{test_score_no_shuffling.std():.3f}"
)

cv = KFold(shuffle=True)
test_score_with_shuffling = cross_val_score(model, X, y, cv=cv, n_jobs=2)
print(
	"The average accuracy with shuffling is "
	f"{test_score_with_shuffling.mean():.3f} ± "
	f"{test_score_with_shuffling.std():.3f}"
)

all_scores = pd.DataFrame(
	[test_score_no_shuffling, test_score_with_shuffling],
	index=["KFold without shuffling", "KFold with shuffling"]
).T

all_scores.plot.hist(bins=16, edgecolor="black", alpha=0.7)
plt.xlim([0.8, 1.0])
plt.xlabel("Accuracy score")
plt.legend(loc="upper left")
plt.title("Distribution of the test scores")
plt.show()

"""
Shuffling the data results in a higher cross-validated test accuracy with less variance
compared to when the data is not shuffled. It means that some specific fold lead to a low
score in this case.
"""

print(test_score_no_shuffling)

"""
Thus, shuffling the data breaks the underlying structure and thus makes the classification task
easier to our model. To get a better understanding, we can read the dataset description in more detail
"""

print(df.DESCR)

"""
If we read carefully, load_digits loads a copy of the test set of the UCI ML hand-written digits datatest,
which consists of 1797 images by 13 different writers. Thus, each writer wrote several times the same numbers.
Let's suppose the dataset is ordered by writer. Subsequently, not shuffling the data will keep all writer samples
together either in the training or the testing sets. Mixing the data will break this structure, and therefore digits
written by the same writer will be available in both the training and testing sets

Besides, a writer will usually tend to write digits in the same manner. Thus, our model will learn to identify
a writer's pattern for each digit instead of recognizing the digit itself.

We can solve this problem by ensuring that the data associated with a writer should either 
belong to the training or the training set. Thus, we want to group samples for each writer.

Indeed, we can recover the groups by looking at the target variable.
"""

print(y[:200])

"""
It might not be obvious at first, but there is a structure in the target:
there is a repetitive pattern that always starts by some series of ordered digits from 0 to 9
followed by random digits at a certain point. If we look in detail, we see that there are 14 such
patterns, always with around 130 samples each.

Even if it's not exactly corresponding to the 13 writers in the documentation. We can make the
hypothesis that each of these patterns corresponds to a different writer and thus a different group
"""

# defines the lower and upper bounds of sample indices
# for each writer
writer_boundaries = [
	0,
	130,
	256,
	386,
	516,
	646,
	776,
	915,
	1029,
	1157,
	1287,
	1415,
	1545,
	1667,
	1797,
]
groups = np.zeros_like(y)
lower_bounds = writer_boundaries[:-1]
upper_bounds = writer_boundaries[1:]

for group_id, lb, up in zip(count(), lower_bounds, upper_bounds):
	groups[lb:up] = group_id

print(groups)
print(lower_bounds)
print(upper_bounds)

plt.plot(groups)
plt.yticks(np.unique(groups))
plt.xticks(writer_boundaries, rotation=90)
plt.xlabel("Target index")
plt.ylabel("Writer index")
plt.title("Underlying writer groups existing in the target")
plt.show()

cv = GroupKFold()
test_score = cross_val_score(
	model,
	X,
	y,
	groups=groups,
	cv=cv,
	n_jobs=2
)
print(
	f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)

"""
We see that this strategy leads to a lower generalization performance than the other two
techniques. However, this is the most reliable estimate if our goal is to evaluate the capabilities of the
model to generalize to new unseen writers. In this sense, shuffling the dataset would lead the model
to memorize the different writer's particular handwriting
"""

all_scores = pd.DataFrame(
	[test_score_no_shuffling, test_score_with_shuffling, test_score],
	index=[
		"KFold without shuffling",
		"KFold with shuffling",
		"KFold with groups",
	],
).T

all_scores.plot.hist(bins=16, edgecolor="black", alpha=0.7)
plt.xlim([0.8, 1.0])
plt.xlabel("Accuracy score")
plt.legend(loc="upper left")
plt.title("Distribution of the test scores")
plt.show()


