import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)
X = mnist_X.astype(np.float32) / 255
y = mnist_y.astype(np.int32)

#plt.imshow(X[0].reshape(28, 28), cmap='gray')
#print(y[0])
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))
print(model)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for data, targets in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)
    print('{}/{} ({:.0f}%)'.format(correct, data_num, 100 * correct / data_num))

test()

for epoch in range(3):
    train(epoch)
test()
