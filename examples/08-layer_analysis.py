import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from bijou.datasets import mnist
from bijou.callbacks import LayerAnalysisCallback
from bijou.metrics import accuracy
from bijou.data import Dataset, DataLoader, DataBunch
from bijou.learner import Learner
from torch import optim
import torch.nn.functional as F
import torch.nn as nn


x_train, y_train, x_valid, y_valid, x_test, y_test = mnist()
train_ds, valid_ds, test_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid), Dataset(x_test, y_test)
bs = 128
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=bs)
test_dl = DataLoader(test_ds, batch_size=bs)
data = DataBunch(train_dl, valid_dl)


in_dim = data.train_ds.x.shape[1]
h_dim = 128
model = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, 10))
opt = optim.SGD(model.parameters(), lr=0.35)


cbs = [
        LayerAnalysisCallback(forward=True),  # show outputs analysis of each sigle layer
        LayerAnalysisCallback(forward=True)   # show gradients analysis of each sigle layer
      ]
loss_func = F.cross_entropy
learner = Learner(model, opt, loss_func, data, metrics=[accuracy], callbacks=cbs)

learner.fit(3)

plt.show()
