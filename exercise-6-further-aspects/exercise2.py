# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:11:32 2024

@author: turunenj
"""

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

opt = BayesSearchCV(AdaBoostClassifier(), {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.01, 2.0, prior='log-uniform'),
    'algorithm': ['SAMME']
},
                    n_iter=32,
                    random_state=0)

_ = opt.fit(X_train, y_train)

print("Best parameters:", opt.best_params_)
print("Training score:", opt.score(X_train, y_train))
print("Testing score:", opt.score(X_test, y_test))
