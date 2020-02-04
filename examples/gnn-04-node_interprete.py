import sys
sys.path.append('..')

import torch
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from bijou.data import PyGDataWrapper, DataBunch
from bijou.learner import Learner
from bijou.metrics import masked_cross_entropy, masked_accuracy
from bijou.datasets import cora
from bijou.callbacks import PyGNodeInterpreter
import networkx as nx
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
else:
    torch.manual_seed(1)

dataset = Planetoid(root=cora(), name='Cora')
train_data = PyGDataWrapper(dataset[0], 'train')
val_data = PyGDataWrapper(dataset[0], 'val')
test_data = PyGDataWrapper(dataset[0], 'test')
data = DataBunch(train_data, val_data)


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

learner = Learner(model, opt, masked_cross_entropy, data, metrics=[masked_accuracy], callbacks=PyGNodeInterpreter)

learner.fit(100)
learner.test(test_data)

def loss_noreduction(pred, target):
    return F.cross_entropy(pred[target.mask], target.data[target.mask], reduction='none')

scores, xs, ys, preds, indecies = learner.interpreter.top_data(loss_noreduction, k=10, target='train', largest=True)
learner.interpreter.plot_confusion(target='train')
learner.interpreter.plot_confusion(target='val')
learner.interpreter.plot_confusion(target='test')

scores, xs, ys, preds, indecies = learner.interpreter.most_confused()
print(scores)
print(ys)
print(preds)
print(indecies)

# learner.interpreter.plot_graph(loss)
learner.interpreter.plot_graph(loss_noreduction, layout=nx.spring_layout, max_node_size=1000, min_node_size=300,
                               label_score=True, label_id=True, k=15, font_color='r', font_size=6)
plt.show()
