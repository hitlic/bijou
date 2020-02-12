import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.function as fn
from dgl.data import MiniGCDataset
from bijou.data import DGLDataLoader, DataBunch
from bijou.metrics import accuracy
from bijou.learner import Learner
import matplotlib.pyplot as plt


# 1. dataset
train_ds = MiniGCDataset(320, 10, 20)
val_ds = MiniGCDataset(100, 10, 20)
test_ds = MiniGCDataset(80, 10, 20)

train_dl = DGLDataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DGLDataLoader(val_ds, batch_size=32, shuffle=False)
test_dl = DGLDataLoader(test_ds, batch_size=32, shuffle=False)

data = DataBunch(train_dl, val_dl)

# 2. mode and optimizer

msg = fn.copy_src(src='h', out='m')  # Sends a message of node feature h.

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

model = Classifier(1, 256, train_ds.num_classes) 
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. learne
loss_func = nn.CrossEntropyLoss()
learner = Learner(model, optimizer, loss_func, data, metrics=accuracy)

# 4. fit
learner.fit(80)

# 5. test
learner.test(test_dl)

# 6. predict
learner.predict(test_dl)

# 7. plot
learner.recorder.plot_metrics()
plt.show()
