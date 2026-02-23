import numpy as np
from aeon.datasets import load_arrow_head
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train, y_train = load_arrow_head(split="train", return_type="numpy3d")
X_test, y_test = load_arrow_head(split="test", return_type="numpy3d")

X_train2d = X_train.squeeze()
X_test2d = X_test.squeeze()

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train2d, y_train)

# Predict
y_pred = clf.predict(X_test2d)

# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy on ArrowHead (RF): {acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
