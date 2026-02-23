# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:14:14 2024

@author: turunenj
"""
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# import some data to play with Iris plants dataset
iris = datasets.load_iris()

# Take the first two features (sepal length and sepal width in cm).
X = iris.data[:, :2]
y = iris.target

# split data into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# choose one model: RBF kernel SVC
C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel="rbf", gamma=0.7, C=C)
clf.fit(X_train, y_train)

# predictions
y_pred = clf.predict(X_test)

# accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (RBF SVC): {acc*100:.2f}%")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# visualize confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (RBF SVC)")
plt.show()
