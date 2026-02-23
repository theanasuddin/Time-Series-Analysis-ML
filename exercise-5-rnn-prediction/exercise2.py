import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

TICKER = "BTC-USD"
START = "2014-09-17"
END = "2022-01-14"
LOOKBACK = 30
HORIZON = 1
TEST_RATIO = 0.20
EPOCHS = 60
LR = 0.001
DEVICE = torch.device("cpu")

df = yf.download(TICKER,
                 start=START,
                 end=END,
                 progress=False,
                 auto_adjust=False)
csv_name = f"{TICKER}.csv"
df.to_csv(csv_name)

target_col = "Close"
if "Adj Close" in df.columns:
    features_cols = ["Open", "High", "Low", "Volume", "Adj Close"]
else:
    features_cols = ["Open", "High", "Low", "Volume", "Close"]

df = df.dropna()
features = df[features_cols].values.astype(float)
target = df[[target_col]].values.astype(float)

n = len(df)
test_size = int(np.ceil(TEST_RATIO * n))
train_size = n - test_size

feat_scaler = StandardScaler()
tgt_scaler = StandardScaler()
feat_scaler.fit(features[:train_size])
tgt_scaler.fit(target[:train_size])
features_scaled = feat_scaler.transform(features)
target_scaled = tgt_scaler.transform(target)


def make_sequences(X, y, lookback, horizon=1):
    Xs, ys = [], []
    for i in range(len(X) - lookback - (horizon - 1)):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback:i + lookback + horizon])
    return np.array(Xs), np.array(ys)


X_all, Y_all = make_sequences(features_scaled, target_scaled, LOOKBACK,
                              HORIZON)
split_index = train_size - LOOKBACK - (HORIZON - 1)
X_train, Y_train = X_all[:split_index], Y_all[:split_index]
X_test, Y_test = X_all[split_index:], Y_all[split_index:]

X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
Y_train_t = torch.from_numpy(Y_train).float().to(DEVICE)
X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
Y_test_t = torch.from_numpy(Y_test).float().to(DEVICE)


def compute_mse_scaled(y_true_scaled, y_pred_scaled, scaler):
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    return mean_squared_error(y_true, y_pred), y_true, y_pred


class BaselineLSTM(nn.Module):

    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(x.size(0), 1, 1)


class ImprovedLSTM(nn.Module):

    def __init__(self,
                 n_features,
                 preproc_dim=32,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.pre = nn.Linear(n_features, preproc_dim)
        self.lstm = nn.LSTM(preproc_dim,
                            hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        b, seq, nf = x.size()
        x = torch.relu(self.pre(x.view(b * seq, nf))).view(b, seq, -1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(b, 1, 1)


def train_model(model, X_tr, Y_tr, X_val, Y_val, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        y_hat = model(X_tr)
        loss = loss_fn(y_hat.view(-1), Y_tr.view(-1))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val).cpu().numpy()
    return y_pred_val


n_features = X_train.shape[2]

baseline = BaselineLSTM(n_features)
y_pred_baseline = train_model(baseline, X_train_t, Y_train_t, X_test_t,
                              Y_test_t, EPOCHS, LR)
mse_baseline, y_true_base, y_pred_base = compute_mse_scaled(
    Y_test.reshape(-1, 1), y_pred_baseline.reshape(-1, 1), tgt_scaler)

improved = ImprovedLSTM(n_features)
y_pred_improved = train_model(improved, X_train_t, Y_train_t, X_test_t,
                              Y_test_t, EPOCHS, LR)
mse_improved, y_true_imp, y_pred_imp = compute_mse_scaled(
    Y_test.reshape(-1, 1), y_pred_improved.reshape(-1, 1), tgt_scaler)

print(f"Baseline MSE: {mse_baseline:.6f}")
print(f"Improved MSE: {mse_improved:.6f}")

plt.figure(figsize=(12, 5))
plt.plot(y_true_base, label="True", color="black")
plt.plot(y_pred_base, label="Baseline", color="red", alpha=0.7)
plt.title(f"Baseline Predictions (MSE = {mse_baseline:.2e})")
plt.xlabel("Time Step")
plt.ylabel("BTC Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("baseline_predictions.png", dpi=200)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_true_imp, label="True", color="black")
plt.plot(y_pred_imp, label="Improved", color="blue", alpha=0.7)
plt.title(f"Improved Predictions (MSE = {mse_improved:.2e})")
plt.xlabel("Time Step")
plt.ylabel("BTC Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("improved_predictions.png", dpi=200)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_true_imp, label="True", color="black")
plt.plot(y_pred_base, label="Baseline", color="red", alpha=0.5)
plt.plot(y_pred_imp, label="Improved", color="blue", alpha=0.8)
plt.title("Baseline vs Improved Predictions")
plt.xlabel("Time Step")
plt.ylabel("BTC Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_predictions.png", dpi=200)
plt.show()

out_df = pd.DataFrame({
    "date":
    df.index[LOOKBACK + split_index:LOOKBACK + split_index + len(y_pred_base)],
    "y_true":
    y_true_base,
    "y_pred_baseline":
    y_pred_base,
    "y_pred_improved":
    y_pred_imp,
})
out_df.to_csv("btc_predictions_comparison.csv", index=False)
