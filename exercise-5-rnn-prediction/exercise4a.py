import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X_simple = X[["MedInc"]].copy()

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_simple,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=25)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Compute MSE
mse = mean_squared_error(y_test, y_pred)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(X_test["MedInc"], y_test, color="blue", label="Actual")
plt.scatter(X_test["MedInc"], y_pred, color="red", label="Predicted")
# Draw regression line
x_line = np.linspace(X_simple["MedInc"].min(), X_simple["MedInc"].max(), 100)
y_line = model.intercept_ + model.coef_[0] * x_line
plt.plot(x_line, y_line, color="black", linewidth=2, label="Fit line")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value")
plt.legend()
plt.title(f"Linear Regression: MSE = {mse:.4f}")
plt.tight_layout()
plt.show()
