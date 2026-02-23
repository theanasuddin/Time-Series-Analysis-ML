import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

LOOKBACK = 5
TEST_RATIO = 0.20
EPOCHS_LSTM = 60
LR = 0.001
DEVICE = torch.device("cpu")

# Linear regression (Statology example)
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_simple = X[["MedInc"]].copy()
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_simple, y, test_size=TEST_RATIO, random_state=25)
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)
mse_lr = mean_squared_error(y_test_lr, y_pred_lr)

# Save linear regression plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test_lr["MedInc"], y_test_lr, color='blue', label='Actual', s=10)
plt.scatter(X_test_lr["MedInc"],
            y_pred_lr,
            color='red',
            label='Predicted',
            s=10)
x_line = np.linspace(X_simple["MedInc"].min(), X_simple["MedInc"].max(), 200)
y_line = lr_model.intercept_ + lr_model.coef_[0] * x_line
plt.plot(x_line, y_line, color='black', linewidth=2, label='Fit line')
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.title(f"Linear Regression (MSE={mse_lr:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_statology.png", dpi=200)
plt.close()

# Adapt multivariate LSTM model to the Statology dataset
features = X.values.astype(float)
target = y.reshape(-1, 1)
n = len(X)
test_size = int(np.ceil(TEST_RATIO * n))
train_size = n - test_size

feat_scaler = StandardScaler()
tgt_scaler = StandardScaler()
feat_scaler.fit(features[:train_size])
tgt_scaler.fit(target[:train_size])
features_scaled = feat_scaler.transform(features)
target_scaled = tgt_scaler.transform(target).reshape(-1)


def make_sequences(Xa, ya, lookback):
    Xs, ys = [], []
    for i in range(len(Xa) - lookback):
        Xs.append(Xa[i:i + lookback])
        ys.append(ya[i + lookback])
    return np.array(Xs), np.array(ys)


X_all, Y_all = make_sequences(features_scaled, target_scaled, LOOKBACK)
split_index = train_size - LOOKBACK
X_train, Y_train = X_all[:split_index], Y_all[:split_index]
X_test, Y_test = X_all[split_index:], Y_all[split_index:]

X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
Y_train_t = torch.from_numpy(Y_train).float().to(DEVICE)
X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
Y_test_t = torch.from_numpy(Y_test).float().to(DEVICE)


class LSTMModel(nn.Module):

    def __init__(self, n_features, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(-1)


n_features = X_train.shape[2]
model = LSTMModel(n_features=n_features, hidden_size=64)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
for _ in range(EPOCHS_LSTM):
    model.train()
    opt.zero_grad()
    y_hat = model(X_train_t)
    loss = loss_fn(y_hat, Y_train_t)
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(X_test_t).cpu().numpy()
y_pred_test = tgt_scaler.inverse_transform(y_pred_test_scaled.reshape(
    -1, 1)).reshape(-1)
y_test_actual = tgt_scaler.inverse_transform(Y_test.reshape(-1, 1)).reshape(-1)
mse_lstm = mean_squared_error(y_test_actual, y_pred_test)

# Save LSTM plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='True', color='black')
plt.plot(y_pred_test, label='LSTM Pred', color='red', alpha=0.7)
plt.xlabel("Test sample index")
plt.ylabel("MedHouseVal")
plt.title(f"LSTM Predictions (MSE={mse_lstm:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_predictions_statology.png", dpi=200)
plt.close()

# Combined
img1 = Image.open("linear_regression_statology.png")
img2 = Image.open("lstm_predictions_statology.png")
w = max(img1.width, img2.width)
h = img1.height + img2.height
combined = Image.new('RGB', (w, h), (255, 255, 255))
combined.paste(img1, (0, 0))
combined.paste(img2, (0, img1.height))
combined.save("comparison_statology_lr_vs_lstm.png")

# Results
print(f"{mse_lr:.4f}")
print(f"{mse_lstm:.4f}")
