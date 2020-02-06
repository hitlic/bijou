from torch.utils.data import DataLoader as TrochDataLoader


class Dataset():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DataLoaderBase:
    pass


class DataLoader(TrochDataLoader, DataLoaderBase):
    def __init__(self, *p, **pd):
        super().__init__(*p, **pd)

    @classmethod
    def loaders(cls, train_ds, val_ds, test_ds=None, batch_size=1):
        train_dl = cls(train_ds, batch_size=batch_size)
        val_dl = cls(val_ds, batch_size=batch_size)
        if test_ds is None:
            return train_dl, val_dl
        else:
            return train_dl, val_dl, cls(test_ds, batch_size=batch_size)


class DataBunch():
    def __init__(self, train_dl, valid_dl=None):
        self.train_dl, self.valid_dl = train_dl, valid_dl

    @classmethod
    def from_dataset(cls, train_ds, valid_ds=None, batch_size=1):
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        if valid_ds is not None:
            valid_dl = DataLoader(valid_dl, batch_size=batch_size)
        else:
            valid_dl = None
        return cls(train_dl, valid_dl)

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
