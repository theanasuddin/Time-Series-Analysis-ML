# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:58:40 2024

@author: turunenj
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Train model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)
