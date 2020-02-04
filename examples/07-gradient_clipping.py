import sys
sys.path.append('..')

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from bijou.learner import Learner
from bijou.data import Dataset, DataLoader, DataBunch
from bijou.metrics import accuracy
from bijou.callbacks import GradientClipping
from bijou.datasets import mnist
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
learner = Learner(model, opt, loss_func, data, metrics=[accuracy], callbacks=GradientClipping(0.001))

learner.fit(3)
learner.test(test_dl)
pred = learner.predict(x_valid)

print(pred.size())

learner.recorder.plot_loss()
learner.recorder.plot_metrics()
plt.show()



# import sys
# sys.path.append('..')

# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from bijou.learner import Learner
# from bijou.data import Dataset, DataLoader, DataBunch
# from bijou.metrics import accuracy
# from bijou.callbacks import GradientClipping
# from datasets import mnist_data
# import matplotlib.pyplot as plt

# # 1. ------ 数据
# x_train, y_train, x_valid, y_valid = mnist_data()
# train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
# bs = 128
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
# valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True)
# data = DataBunch(train_dl, valid_dl)

# # 2. ------ 模型和优化器
# in_dim = data.train_ds.x.shape[1]
# out_dim = y_train.max().item()+1
# h_dim = 50
# model = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))
# opt = optim.SGD(model.parameters(), lr=0.35)


# # 3. ------ learner
# loss_func = F.cross_entropy
# learner = Learner(model, opt, loss_func, data, metrics=[accuracy], callbacks=GradientClipping(0.01))
# # 4. ------ fit
# learner.fit(3)

# # 5. ------ test
# learner.test(valid_dl)

# # 6. ------ predict
# pred = learner.predict(x_valid)
# print(pred.size())

# # 7. ------ plot
# learner.recorder.plot_loss()
# plt.show()
