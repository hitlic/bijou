import sys
sys.path.append('..')

import networkx as nx
from dgl.data import citation_graph as citegrh
from dgl import DGLGraph
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from bijou.data import DGLGraphLoader, DataBunch
from bijou.learner import Learner
import matplotlib.pyplot as plt


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    val_mask = th.BoolTensor(data.val_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, val_mask, test_mask


g, features, labels, train_mask, val_mask, test_mask = load_cora_data()

train_dl = DGLGraphLoader([g], features=features, labels=labels, mask=train_mask)
val_dl = DGLGraphLoader([g], features=features, labels=labels, mask=val_mask)
test_dl = DGLGraphLoader([g], features=features, labels=labels, mask=test_mask)
data = DataBunch(train_dl, val_dl)

net = Net()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)


def acc(pred, target):
    label = target.data
    mask = target.mask
    pred = th.argmax(pred[mask], 1)
    target = label[mask]
    return (pred == target).int().sum()*1.0/len(target)


def loss(pred, target):
    return F.cross_entropy(pred[target.mask], target.data[target.mask])

learner = Learner(net, optimizer, loss, data, metrics=acc)

learner.fit(50)
learner.test(test_dl)
learner.predict(test_dl)

learner.recorder.plot_metrics()
plt.show()
