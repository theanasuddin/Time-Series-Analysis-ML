import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load and prepare data
data = sns.load_dataset("flights")['passengers'].values.astype(float)
n = len(data)
# Split 80 / 20
test_size = int(np.ceil(0.20 * n))
train_size = n - test_size

train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
train_scaled = torch.FloatTensor(train_scaled).view(-1)

train_window = 12
batch_sequence_list = []


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_scaled, train_window)


# Improved LSTM
class LSTMForecast(nn.Module):

    def __init__(self,
                 input_size=1,
                 preproc_size=8,
                 lstm_hidden=64,
                 lstm_layers=2,
                 dropout=0.2,
                 output_size=1):
        super().__init__()
        self.preproc = nn.Linear(input_size, preproc_size)
        self.lstm = nn.LSTM(input_size=preproc_size,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=dropout)
        self.decoder = nn.Linear(lstm_hidden, output_size)

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

    def init_hidden(self, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm_layers, 1, self.lstm_hidden, device=device)
        c0 = torch.zeros(self.lstm_layers, 1, self.lstm_hidden, device=device)
        return (h0, c0)

    def forward(self, input_seq):
        seq = input_seq.view(len(input_seq), 1, -1)
        seq_flat = seq.view(-1, seq.size(-1))
        pre = self.preproc(seq_flat)
        pre = torch.relu(pre)
        pre = pre.view(len(input_seq), 1, -1)

        lstm_out, self.hidden = self.lstm(pre, self.hidden)
        out = self.decoder(lstm_out.view(len(input_seq), -1))
        return out[-1]


# Train / evaluate
device = torch.device("cpu")
model = LSTMForecast(input_size=1,
                     preproc_size=16,
                     lstm_hidden=128,
                     lstm_layers=2,
                     dropout=0.15,
                     output_size=1)
model.to(device)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200

# Moved the hidden_cell initialization before the training loop
model.hidden = model.init_hidden(device=device)

for epoch in range(epochs):
    model.train()
    random.shuffle(train_inout_seq)
    epoch_loss = 0.0

    model.hidden = model.init_hidden(device=device)

    for seq, label in train_inout_seq:
        optimizer.zero_grad()
        model.hidden = tuple([h.detach() for h in model.hidden])
        y_pred = model(seq.to(device))
        loss = loss_function(y_pred.view(-1), label.to(device).view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 25 == 0 or epoch == 0:
        avg_loss = epoch_loss / len(train_inout_seq)
        print(f"Epoch {epoch+1}/{epochs} — avg train loss: {avg_loss:.6f}")

# Evaluation / forecasting
model.eval()
model.hidden = model.init_hidden(device=device)

test_inputs = train_scaled[-train_window:].tolist()

fut_pred = len(test_data)
for _ in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = model.init_hidden(device=device)
        pred = model(seq.to(device)).item()
        test_inputs.append(pred)

preds_scaled = np.array(test_inputs[train_window:]).reshape(-1, 1)
preds = scaler.inverse_transform(preds_scaled)

actual = test_data.reshape(-1, 1)
mse = np.mean((preds.flatten() - actual.flatten())**2)
print(f"\nTest MSE on {fut_pred} points: {mse:.4f}")

# Plot results
plt.figure(figsize=(12, 4))
x_all = np.arange(n)
plt.plot(x_all, data, label='Actual (full series)')
x_pred = np.arange(train_size, train_size + fut_pred)
plt.plot(x_pred,
         preds.flatten(),
         marker='o',
         label='Predictions (test horizon)')
plt.axvline(x=train_size - 0.5, color='gray', linestyle='--', alpha=0.5)
plt.legend()
plt.title('LSTM Forecast (improved model)')
plt.show()
