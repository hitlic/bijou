import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool, TopKPooling, GCNConv
from bijou.learner import Learner
from bijou.datasets import yoochoose_10k
from bijou.data import PyGDataLoaderWrapper, DataBunch
from bijou.metrics import accuracy
from bijou.callbacks import PyGGraphInterpreter
from pyg_dataset import YooChooseBinaryDataset
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
else:
    torch.manual_seed(1)

# 1. dataset
dataset = YooChooseBinaryDataset(root=yoochoose_10k()).shuffle()
train_ds, val_ds, test_ds = dataset[:8000], dataset[8000:9000], dataset[9000:]
train_dl = PyGDataLoaderWrapper(DataLoader(train_ds, batch_size=64, shuffle=True))
val_dl = PyGDataLoaderWrapper(DataLoader(val_ds, batch_size=64))
test_dl = PyGDataLoaderWrapper(DataLoader(test_ds, batch_size=64))
train_db = DataBunch(train_dl, val_dl)


# 2. mode and optimizer
class Model(nn.Module):
    def __init__(self, feature_dim, class_num, embed_dim=64, gcn_dims=(32, 32), dense_dim=64):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=feature_dim, embedding_dim=embed_dim)
        self.gcns = nn.ModuleList()
        in_dim = embed_dim
        for dim in gcn_dims:
            self.gcns.append(GCNConv(in_dim, dim))
            in_dim = dim
        self.graph_pooling = TopKPooling(gcn_dims[-1], ratio=0.8)
        self.dense = nn.Linear(gcn_dims[-1], dense_dim)
        self.out = nn.Linear(dense_dim, class_num)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        x = x.squeeze(1)
        for gcn in self.gcns:
            x = gcn(x, edge_index)
            x = F.relu(x)
        x, _, _, batch, _, _ = self.graph_pooling(x, edge_index, None, batch)
        x = global_max_pool(x, batch)
        outputs = self.dense(x)
        outputs = F.relu(outputs)
        outputs = self.out(outputs)
        return outputs

model = Model(dataset.item_num, 2)
opt = optim.SGD(model.parameters(), lr=0.5)


# 3. learner
learner = Learner(model, opt, F.cross_entropy, train_db, metrics=[accuracy], callbacks=PyGGraphInterpreter())

# 4. fit
learner.fit(3)

# 5. test
learner.test(test_dl)


loss = nn.CrossEntropyLoss(reduction='none')
scores, xs, ys, preds, indecies = learner.interpreter.top_data(loss, k=10, target='train', largest=True)

plt.figure(figsize=[10, 5])
for i in range(10):
    plt.subplot(2, 5, i + 1)
    learner.interpreter.plot_graph(xs[i], node_size=50)
    plt.title(f'loss-{scores[i]:0.6f}')

learner.interpreter.plot_confusion()
learner.interpreter.plot_confusion('val')
learner.interpreter.plot_confusion('test')
mcfs = learner.interpreter.most_confused()
print([[c[0], len(c[1])]for c in mcfs])
plt.show()
