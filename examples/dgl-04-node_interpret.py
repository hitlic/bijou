import sys
sys.path.append('..')

import torch.nn.functional as F
import torch.nn as nn
import torch as th
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from bijou.learner import Learner
from bijou.data import GraphLoader, DataBunch
from bijou.metrics import masked_accuracy, masked_cross_entropy
from bijou.callbacks import DGLGraphInterpreter
import matplotlib.pyplot as plt
import networkx as nx


# 1. dataset
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
train_dl = GraphLoader(g, features=features, labels=labels, mask=train_mask)
val_dl = GraphLoader(g, features=features, labels=labels, mask=val_mask)
test_dl = GraphLoader(g, features=features, labels=labels, mask=test_mask)
# train_dl, val_dl, test_dl = GraphLoader.loaders(g, features=features, labels=labels,
#                                                 masks=[train_mask, val_mask, test_mask])
data = DataBunch(train_dl, val_dl)


# 2. model and optimizer
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

net = Net()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)


# 3. learner
learner = Learner(net, optimizer, masked_cross_entropy, data, metrics=masked_accuracy, callbacks=DGLGraphInterpreter)

# 4. fit
learner.fit(50)

# 5. test
learner.test(test_dl)


def loss_noreduction(pred, target):
    return F.cross_entropy(pred[target.mask], target.data[target.mask], reduction='none')


scores, xs, ys, preds, indecies = learner.interpreter.top_data(loss_noreduction, k=10, phase='train', largest=True)
learner.interpreter.plot_confusion(phase='train')
learner.interpreter.plot_confusion(phase='val')
learner.interpreter.plot_confusion(phase='test')

print('scores:\n', scores)
print('ys:\n', ys)
print('preds:\n', preds)
print('indecies:\n', indecies)

confuses = learner.interpreter.most_confused()

# layout = nx.kamada_kawai_layout
learner.interpreter.plot_graph(loss_noreduction, max_node_size=1000, min_node_size=300,
                               label_score=True, label_id=True, k=15, font_color='r', font_size=6)
plt.show()
