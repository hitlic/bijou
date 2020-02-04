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
                                                    k=10, target='train', largest=True)
print(scores)
print(indecies)

plt.figure(figsize=[12, 6])
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(xs[i].view([28, -1]))
    plt.title(f'{ys[i]} --> {np.argmax(preds[i])}')

# m = learner.interpreter.confusion_matrix()
learner.interpreter.plot_confusion(target='train', class_names=range(10))
learner.interpreter.plot_confusion(target='val', class_names=range(10))
learner.interpreter.plot_confusion(target='test', class_names=range(10))

mcfs = learner.interpreter.most_confused()
print([[c[0], len(c[1])]for c in mcfs])

plt.show()




# import sys
# sys.path.append('..')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from bijou.learner import Learner
# from bijou.data import Dataset, DataLoader, DataBunch
# from bijou.metrics import accuracy
# from bijou.callbacks import Interpreter
# from datasets import mnist_data
# import matplotlib.pyplot as plt
# import numpy as np

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(1)
# else:
#     torch.manual_seed(1)

# # 1. ------ 数据
# x_train, y_train, x_valid, y_valid = mnist_data()
# x_test = x_valid[:500]
# y_test = y_valid[:500]

# train_ds, valid_ds, test_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid), Dataset(x_test, y_test)
# bs = 128
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
# valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True)
# test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True)
# data = DataBunch(train_dl, valid_dl)

# # 2. ------ 模型和优化器
# in_dim = data.train_ds.x.shape[1]
# out_dim = y_train.max().item()+1
# h_dim = 50
# model = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))
# opt = optim.SGD(model.parameters(), lr=0.35)


# # 3. ------ learner
# loss_func = F.cross_entropy
# learner = Learner(model, opt, loss_func, data, metrics=[accuracy], callbacks=Interpreter())

# # 4. ------ fit
# learner.fit(1)

# # 5. ------ test
# learner.test(test_dl)


# def loss(pred, target):
#     return F.cross_entropy(pred, target, reduction='none')

# scores, xs, ys, preds, indecies = learner.interpreter.top_data(loss, k=10, target='train', largest=True)
# print(scores)
# print(indecies)
# # print(xs)
# plt.figure(figsize=[12, 6])
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(xs[i].view([28, -1]))
#     plt.title(f'{ys[i]} --> {np.argmax(preds[i])}')


# # m = learner.interpreter.confusion_matrix()
# learner.interpreter.plot_confusion(target='train', class_names=range(10))
# learner.interpreter.plot_confusion(target='val', class_names=range(10))
# learner.interpreter.plot_confusion(target='test', class_names=range(10))

# mcfs = learner.interpreter.most_confused()
# print([[c[0], len(c[1])]for c in mcfs])

# plt.show()
