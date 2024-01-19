import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from extra.datasets import fetch_mnist
import time

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, initialization='kaiming_uniform'):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(getattr(torch.nn.init, initialization)(torch.empty(out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.l2(x)
        return x

model = TinyNet()
optimizer = optim.SGD(model.parameters(), lr=3e-4)
def sparse_categorical_crossentropy(y_pred, y_true):
    loss = nn.CrossEntropyLoss()
    return loss(y_pred, y_true)

X_train, Y_train, X_test, Y_test = fetch_mnist()

for step in range(1000):
    samp = np.random.randint(0, X_train.shape[0], size=(64))
    batch = torch.Tensor(X_train[samp]).requires_grad_(False)
    labels = torch.LongTensor(Y_train[samp])

    out = model(batch)

    loss = sparse_categorical_crossentropy(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=-1)
    acc = (pred == labels).float().mean()

    if step % 100 == 0:
        print(f"Step {step+1} | Loss: {loss.item()} | Accuracy: {acc.item()}")

start_time = time.time()
with torch.no_grad():
    avg_acc = 0
    for step in range(1000):
        samp = np.random.randint(0, X_test.shape[0], size=(64))
        batch = torch.Tensor(X_test[samp]).requires_grad_(False)
        labels = torch.LongTensor(Y_test[samp]).long()

        out = model(batch)

        pred = out.argmax(dim=-1)
        avg_acc += (pred == labels).float().mean()

        if step % 100 == 0:
            print(f"Step {step+1} | Loss: {loss.item()} | Accuracy: {acc.item()}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for test set run: {elapsed_time} seconds")