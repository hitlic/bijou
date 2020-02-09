from .data import DataLoaderBase, MaskedTensor, GraphLoader
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import dgl

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class DGLDataLoader(TorchDataLoader, DataLoaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, collate_fn=collate, **kwargs)


class DGLGraphLoader(GraphLoader):
    def __init__(self, dgl_dataset, phase=None, feature_name='feat', label_name='label', features=None, labels=None, mask=None):
        """
        Args:
            pyg_data: torch_geometric.data.data.Data对象
            phase: 'train', 'val' or 'test'
        """
        dgl_data = dgl_dataset[0]
        if mask is None:
            assert phase in ['train', 'val', 'test'], 'param "phase" must be in [train, val, test]'
            mask = dgl_data.ndata.get(f'{phase}_mask')
            assert mask is not None, f'Data must has a attribute "{phase}_mask"'
            mask = torch.tensor(mask, dtype=torch.book)
        if labels is None:
            labels = dgl_data.ndata.get(label_name)
            assert labels is not None, f'Data must has a attribute "{label_name}"'
            labels = torch.tensor(labels, torch.long)
        if features is None:
            features = dgl_data.ndata.get(feature_name)
            assert features is not None, f'Data must has a attribute "{feature_name}"'
            features = torch.tensor(features, dtype=torch.float)

        # GraphLoader must have following three attributes
        self.data = (dgl_data, features)
        self.label = MaskedTensor(labels, mask)
        self.mask = mask
