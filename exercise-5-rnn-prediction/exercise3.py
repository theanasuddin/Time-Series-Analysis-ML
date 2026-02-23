import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Data loading and preparation
data = sns.load_dataset("flights")["passengers"].values.astype(float)
n = len(data)
test_size = int(np.ceil(0.20 * n))
train_size = n - test_size

train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = torch.FloatTensor(
    scaler.fit_transform(train_data.reshape(-1, 1))).view(-1)

train_window = 12


def create_inout_sequences(input_data, tw):
    seqs = []
    L = len(input_data)
    for i in range(L - tw):
        seq = input_data[i:i + tw]
        label = input_data[i + tw:i + tw + 1]
        seqs.append((seq, label))
    return seqs


train_seq = create_inout_sequences(train_scaled, train_window)


class SeqModel(nn.Module):

    def __init__(self, rnn_type="LSTM", input_size=1, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size)
        else:
            raise ValueError("rnn_type must be LSTM, RNN, or GRU")
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        seq = seq.view(len(seq), 1, -1)
        out, hn = self.rnn(seq)
        out2 = self.linear(out.view(len(seq), -1))
        return out2[-1]

    def init_hidden(self):
        if isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(1, 1, self.hidden_size),
                    torch.zeros(1, 1, self.hidden_size))
        else:
            return torch.zeros(1, 1, self.hidden_size)


# Train and test routine
def train_model(model, train_seq, epochs=100, lr=0.001):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for seq, label in train_seq:
            model.zero_grad()
            model.rnn.flatten_parameters()
            hidden = model.init_hidden()
            if isinstance(hidden, tuple):
                hidden = tuple([h.detach() for h in hidden])
            else:
                hidden = hidden.detach()
            y_pred = model(seq)
            loss = loss_fn(y_pred.view(-1), label.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model


def predict_full(model, train_scaled, test_len):
    test_inputs = train_scaled[-train_window:].tolist()
    model.eval()
    with torch.no_grad():
        for _ in range(test_len):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            model.rnn.flatten_parameters()
            pred = model(seq).item()
            test_inputs.append(pred)
    preds = scaler.inverse_transform(
        np.array(test_inputs[train_window:]).reshape(-1, 1))
    return preds.flatten()


def summed_abs_error(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)))


# Train three variants
models = {}
preds = {}
errors = {}

for rtype in ["LSTM", "RNN", "GRU"]:
    m = SeqModel(rnn_type=rtype, input_size=1, hidden_size=64)
    m = train_model(m, train_seq, epochs=150, lr=0.005)
    models[rtype] = m
    p = predict_full(m, train_scaled, len(test_data))
    preds[rtype] = p
    errors[rtype] = summed_abs_error(test_data, p)

# Plotting
plt.figure(figsize=(10, 5))
xaxis = np.arange(n)
plt.plot(xaxis, data, label="Actual (full series)")
for rtype, p in preds.items():
    xx = np.arange(train_size, train_size + len(p))
    plt.plot(xx, p, label=f"Pred {rtype}")
plt.axvline(x=train_size - 0.5, color="black", linestyle="--")
plt.legend()
plt.title("Actual vs Predictions: LSTM, RNN, GRU")
plt.show()

# Results summary
for r in ["LSTM", "RNN", "GRU"]:
    print(f"{r} summed abs error on test set: {errors[r]:.4f}")
