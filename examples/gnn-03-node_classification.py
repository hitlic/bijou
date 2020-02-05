import sys
sys.path.append('..')

import torch
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from bijou.data import PyGGraphLoader, DataBunch
from bijou.learner import Learner
from bijou.metrics import masked_cross_entropy, masked_accuracy
from bijou.datasets import cora
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
else:
    torch.manual_seed(1)

# 1. dataset
dataset = Planetoid(root=cora(), name='Cora')
train_dl = PyGGraphLoader(dataset, 'train')
val_dl = PyGGraphLoader(dataset, 'val')
test_dl = PyGGraphLoader(dataset, 'test')
# train_dl, val_dl, test_dl = PyGGraphLoader.loaders(dataset)
data = DataBunch(train_dl, train_dl)


# 2. model and optimizer
class Model(nn.Module):
    def __init__(self, feature_num, class_num):
        super().__init__()
        self.conv1 = GCNConv(feature_num, 16)
        self.conv2 = GCNConv(16, class_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        outputs = F.relu(x)
        return outputs

model = Model(dataset.num_node_features, dataset.num_classes)
opt = optim.SGD(model.parameters(), lr=0.5, weight_decay=0.01)


# 3. learner
learner = Learner(model, opt, masked_cross_entropy, data, metrics=[masked_accuracy])

# 4. fit
learner.fit(100)

# 5. test
learner.test(test_dl)

# 6. predict
pred = learner.predict(dataset[0])
print(pred.size())

# 7. plot
learner.recorder.plot_metrics()
plt.show()
