import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from bijou.datasets import mnist
from bijou.data import DataProcess as dp, Dataset, DataLoader, DataBunch
from bijou.modules import Lambda
from bijou.metrics import accuracy
from bijou.learner import Learner
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


def mnist_resize(x):
    return x.view(-1, 1, 28, 28)


def cnn_model(out_dim):
    return nn.Sequential(
        Lambda(mnist_resize),
        nn.Conv2d(1, 8, 5, padding=2, stride=2), nn.ReLU(),  # 14
        nn.Conv2d(8, 16, 3, padding=1, stride=2), nn.ReLU(),  # 7
        nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.ReLU(),  # 4
        nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),  # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(dp.flatten),
        nn.Linear(32, out_dim)
    )


x_train, y_train, x_valid, y_valid, x_test, y_test = mnist()
x_train, x_valid, x_test = dp.normalize_to(x_train, x_valid, x_test)
train_ds, valid_ds, test_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid), Dataset(x_test, y_test)
bs = 512
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=bs)
test_dl = DataLoader(test_ds, batch_size=bs)
data = DataBunch(train_dl, valid_dl)

model = cnn_model(10)
opt = optim.Adam(model.parameters(), lr=0.005)

loss_func = F.cross_entropy
learner = Learner(model, opt, loss_func, data, metrics=accuracy)

learner.fit(5)

plt.show()
