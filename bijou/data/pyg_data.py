import torch
from torch_geometric.data import DataLoader as DataLoader_pyg


class PyGDataLoader(DataLoader_pyg):

    def __iter__(self):
        for data in super().__iter__():
            yield data, data.y

    @classmethod
    def loaders(cls, train_ds, val_ds, test_ds=None, batch_size=1):
        train_dl = cls(train_ds, batch_size=batch_size)
        val_dl = cls(val_ds, batch_size=batch_size)
        if test_ds is None:
            return train_dl, val_dl
        else:
            return train_dl, val_dl, cls(test_ds, batch_size=batch_size)


class MaskedTensor:
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def to(self, device):
        self.data = self.data.to(device)
        self.mask = self.mask.to(device)
        return self

    def __len__(self):
        return len(self.data)


class PyGGraphLoader:
    """
    用于节点计算，仅包含单一的torch_geometric.data.data.Data的pyg_dataset
    """

    def __init__(self, pyg_dataset, phase):
        """
        Args:
            pyg_data: torch_geometric.data.data.Data对象
            phase: 'train', 'val' or 'test'
        """
        pyg_data = pyg_dataset[0]
        assert phase in ['train', 'val', 'test'], 'param "phase" must be in [train, val, test]'
        assert pyg_data.x is not None, 'Data must has a attribute "x"'
        assert pyg_data.y is not None, 'Data must has a attribute "y"'
        self.mask = getattr(pyg_data, f'{phase}_mask')
        assert self.mask is not None, f'Data must has a attribute "{phase}_mask"'

        self.data = pyg_data
        self.phase = phase

    @classmethod
    def loaders(cls, pyg_dataset, phases=('train', 'val', 'test')):
        return [cls(pyg_dataset, phase) for phase in phases]

    def __iter__(self):
        yield self.data, MaskedTensor(self.data.y, self.mask)

    def __len__(self):
        if torch.sum(self.mask.int()) > 0:
            return 1
        else:
            return 0
