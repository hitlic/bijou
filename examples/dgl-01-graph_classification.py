import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.data import MiniGCDataset
import torch.optim as optim
from bijou.data import DGLDataLoader
from bijou.callbacks import DGLInterpreter
from bijou.metrics import accuracy
import matplotlib.pyplot as plt

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


# Create training and test sets.
train_ds = MiniGCDataset(320, 10, 20)
val_ds = MiniGCDataset(100, 10, 20)
test_ds = MiniGCDataset(80, 10, 20)

train_dl = DGLDataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DGLDataLoader(val_ds, batch_size=32, shuffle=False)
test_dl = DGLDataLoader(test_ds, batch_size=32, shuffle=False)

from bijou.data import DataBunch
data = DataBunch(train_dl, val_dl)


# Create model
model = Classifier(1, 256, train_ds.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from bijou.learner import Learner

learner = Learner(model, optimizer, loss_func, data, metrics=accuracy, callbacks=DGLInterpreter)
learner.fit(80)
learner.test(test_dl)
learner.predict(test_dl)

learner.recorder.plot_metrics()
plt.show()
