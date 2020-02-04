from torch.utils.data import DataLoader as TrochDataLoader
import torch

class Dataset():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DataLoader(TrochDataLoader):
    def __init__(self, *p, **pd):
        super().__init__(*p, **pd)


class DataBunch():
    def __init__(self, train_dl, valid_dl=None):
        self.train_dl, self.valid_dl = train_dl, valid_dl

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        if self.valid_dl is None:
            return None
        return self.valid_dl.dataset


class DataProcess:
    """
    数据处理工具函数
    """
    @classmethod
    def normalize(cls, x, m, s):
        """
        标准化
        """
        return (x-m)/s

    @classmethod
    def stats(cls, x):
        """
        均值，标准差
        """
        return x.mean(), x.std()

    @classmethod
    def get_dls(cls, train_ds, valid_ds, bs, **kwargs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
                DataLoader(valid_ds, batch_size=bs*2, **kwargs))

    @classmethod
    def normalize_to(cls, train, valid, test=None):
        m, s = train.mean(), train.std()
        if test is None:
            return cls.normalize(train, m, s), cls.normalize(valid, m, s)
        else:
            return cls.normalize(train, m, s), cls.normalize(valid, m, s), cls.normalize(test, m, s)


    @classmethod
    def flatten(cls, x):
        return x.view(x.shape[0], -1)


class PyGDataLoaderWrapper(DataLoader):
    """
    用于图计算torch_geometric.data.DataLoader的Wrapper
    """

    def __init__(self, pyg_loader):
        self.loader = pyg_loader

    def __iter__(self):
        for data in self.loader:
            yield data, data.y

    def __getattr__(self, k):
        return getattr(self.loader, k)


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


class PyGDataWrapper:
    """
    用于节点计算，仅包含单一的torch_geometric.data.data.Data的Wrapper
    """

    def __init__(self, pyg_data, phase):
        """
        Args:
            pyg_data: torch_geometric.data.data.Data对象
            phase: 'train', 'val' or 'test'
        """
        assert phase in ['train', 'val', 'test'], 'param "phase" must be in [train, val, test]'
        assert pyg_data.x is not None, 'Data must has a member "x"'
        assert pyg_data.y is not None, 'Data must has a member "y"'
        self.mask = getattr(pyg_data, f'{phase}_mask')
        assert self.mask is not None, f'Data must has a member "{phase}_mask"'

        self.data = pyg_data
        self.phase = phase

    def __iter__(self):
        yield self.data, MaskedTensor(self.data.y, self.mask)

    def __len__(self):
        if torch.sum(self.mask.int()) > 0:
            return 1
        else:
            return 0
