# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from aeon.datasets import load_classification

X, y = load_classification("JapaneseVowels")  # selected aeon dataset
classes, y = np.unique(y, return_inverse=True)

# train/test split of data 80/20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

def prepare_data(X_train, X_test, normalize=True):
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test

# neural net definition
class classifier_selfmade_network(nn.Module):

    def __init__(
        self,
        inp_units,
        num_units=700,
        num_units1=1000,
        out_units=3,
        nonlin=F.relu,
        nonlin1=F.relu,
        nonlin2=F.relu,
    ):
        super(classifier_selfmade_network, self).__init__()
        self.num_units = num_units
        self.num_units1 = num_units1
        self.nonlin = nonlin
        self.nonlin1 = nonlin1
        self.nonlin2 = nonlin2
        self.dense0 = nn.Linear(inp_units, num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.dense2 = nn.Linear(num_units, num_units1)
        self.output = nn.Linear(num_units1, out_units)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin1(self.dense1(X))
        X = self.nonlin2(self.dense2(X))
        X = self.output(X)
        return X

# training and evaluation
def run_experiment(normalize=False):
    Xtr, Xte = prepare_data(X_train, X_test, normalize)
    inputx = torch.tensor(Xtr).float()
    outputy = torch.tensor(y_train).long()
    train_data = data_utils.TensorDataset(inputx, outputy)
    train_loader = data_utils.DataLoader(dataset=train_data,
                                         batch_size=16,
                                         shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = classifier_selfmade_network(inp_units=Xtr.shape[1],
                                      out_units=len(classes))
    net = nn.DataParallel(net)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.00125)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.9,
                                                     patience=10)

    for epoch in range(50):  # fewer epochs for quick demonstration
        train_loss = 0.0
        if epoch > 0:
            scheduler.step(loss)
        for (xd, yd) in train_loader:
            xd, yd = xd.to(device), yd.to(device)
            optimizer.zero_grad()
            outputti = net(xd)
            loss = criterion(outputti, yd)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print("Epoch:", epoch, "Loss:", train_loss / len(train_loader))

    # testing
    tensori = torch.tensor(Xte).float().to(device)
    test_values = net(tensori).detach().cpu().numpy()
    y_pred = np.argmax(test_values, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", acc * 100, "%")
    print("Confusion matrix:\n", cm)

    # plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (normalize={normalize})")
    plt.show()

    return acc, cm

# run with and without normalization
acc_no_norm, cm_no_norm = run_experiment(normalize=False)
acc_norm, cm_norm = run_experiment(normalize=True)

print("\nAccuracy without normalization:", acc_no_norm * 100, "%")
print("Accuracy with normalization:", acc_norm * 100, "%")
