# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:13:59 2024

@author: turunenj
"""

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load training data
name = './Exercise_Train_data.xlsx'
df = pd.read_excel(name, sheet_name='Sheet1')
df.drop([0], axis=0, inplace=True)

y = df["Column208"]
X1 = df.drop("Column208", axis=1)

# transform to tensors
inputx = torch.tensor(X1.values).float()
outputy = torch.tensor(y.values).float()

train_data = data_utils.TensorDataset(inputx, outputy)
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=10,
                                     shuffle=True)  # batch size changed to 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# define neural network
class classifier_selfmade_network(nn.Module):

    def __init__(
            self,
            inp_units=3 * 69,
            num_units=800,  # increased hidden units
            num_units1=1200,  # increased hidden units
            out_units=3,
            nonlin=F.relu,
            nonlin1=F.relu,
            nonlin2=F.relu):
        super(classifier_selfmade_network, self).__init__()
        self.dense0 = nn.Linear(inp_units, num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.dense2 = nn.Linear(num_units, num_units1)
        self.output = nn.Linear(num_units1, out_units)
        self.nonlin, self.nonlin1, self.nonlin2 = nonlin, nonlin1, nonlin2

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin1(self.dense1(X))
        X = self.nonlin2(self.dense2(X))
        X = self.output(X)
        return X


net = classifier_selfmade_network()
net = nn.DataParallel(net)
net.to(device)

# optimizer and loss
optimizer = optim.Adam(net.parameters(), lr=0.00125)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.9,
                                                 patience=10)

# training
for epoch in range(400):
    train_loss = 0.0
    if epoch > 0:
        scheduler.step(loss)
    for (xd, yd) in train_loader:
        yd = yd.type(torch.LongTensor)
        xd, yd = xd.to(device), yd.to(device)

        outputti = net(xd)
        optimizer.zero_grad()
        loss = criterion(outputti, yd)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print("Epoch:", epoch, "\tLR:", optimizer.param_groups[0]['lr'],
          "\tTraining Loss:", (train_loss / len(train_loader)))

# testing
name = './Exercise_Test_data.xlsx'
df_test = pd.read_excel(name, sheet_name='Sheet1')
df_test.drop([0], axis=0, inplace=True)

yx = df_test["Column208"]
X1_test = df_test.drop("Column208", axis=1)
y_test = yx.to_numpy().astype(int)

tensori = torch.tensor(X1_test.values).float()
input_tensor = tensori.to(device)
test_values = net(input_tensor)
test_values = test_values.detach().cpu().numpy()

y_pred = np.argmax(test_values, axis=1)

correct = np.sum(y_pred == y_test)
total = len(y_test)
print("Correct:", correct / total * 100, "%")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["Biking", "Running", "Other"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Scale'})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
