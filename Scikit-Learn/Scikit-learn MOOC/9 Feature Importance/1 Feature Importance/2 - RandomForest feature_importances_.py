import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from data import X_test, X_train, y_test, y_train

model = RandomForestRegressor()
model.fit(X_train, y_train)

print(f"model score on training data: {model.score(X_train, y_train)}")
print(f"model score on testing data: {model.score(X_test, y_test)}")

"""
Contrary to the testing set, the score on the training set is almost perfect, which means that our model
is overfitting here 
"""

importances = model.feature_importances_

"""
The importance of a feature is basically: how much this feature is used in each tree of the forest.
Formally, it is computed as the (normalized) total reduction of the criterion brought by that features
"""

indices = np.argsort(importances)
plt.barh(range(len(importances)), importances[indices])
plt.yticks(range(len(importances)), labels=np.array(X_train.columns)[indices])
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

"""
Median income is still the most important feature

It also has a small bias toward high cardinality features, such as the noisy feature rnd_num, which are predicted having
0.07 importance, more than HouseAge (which has low cardinality) 

"""
