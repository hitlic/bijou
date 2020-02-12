from .data import DataLoaderBase
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import dgl

def collate(samples):
    # The input `samples` is a list of pairs  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class DGLDataLoader(TorchDataLoader, DataLoaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, collate_fn=collate, **kwargs)
