import sys
sys.path.append('..')

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from bijou.learner import Learner
from bijou.data import Dataset, DataLoader, DataBunch
from bijou.metrics import accuracy
from bijou.callbacks import Interpreter
from bijou.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


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


loss_func = F.cross_entropy
learner = Learner(model, opt, loss_func, data, metrics=[accuracy], callbacks=Interpreter())

learner.fit(3)
learner.test(test_dl)

def loss_noreduction(pred, target):
    return F.cross_entropy(pred, target, reduction='none')

scores, xs, ys, preds, indecies = learner.interpreter.top_data(metric=loss_noreduction,
                                                    k=10, phase='train', largest=True)
print(scores)
print(indecies)

plt.figure(figsize=[12, 6])
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(xs[i][0].view([28, -1]))
    plt.title(f'{ys[i]} --> {np.argmax(preds[i])}')

# m = learner.interpreter.confusion_matrix()
learner.interpreter.plot_confusion(phase='train', class_names=range(10))
learner.interpreter.plot_confusion(phase='val', class_names=range(10))
learner.interpreter.plot_confusion(phase='test', class_names=range(10))

mcfs = learner.interpreter.most_confused()
print([[c[0], len(c[1])]for c in mcfs])

plt.show()
