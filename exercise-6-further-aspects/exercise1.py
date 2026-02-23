# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:13:53 2024

@author: turunenj
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split
from scipy.stats import randint
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

clf = RandomForestClassifier(random_state=0)
np.random.seed(0)

param_distributions = {
    "max_depth": [None] + list(range(3, 20)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ['sqrt', 'log2', None],
    "bootstrap": [True, False]
}

search = HalvingRandomSearchCV(clf,
                               param_distributions,
                               resource='n_estimators',
                               max_resources=300,
                               random_state=0).fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Training score:", search.score(X_train, y_train))
print("Testing score:", search.score(X_test, y_test))
