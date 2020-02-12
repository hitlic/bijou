from torch_geometric.data import DataLoader as DataLoader_pyg
from .data import DataLoaderBase

class PyGDataLoader(DataLoader_pyg, DataLoaderBase):

    def __iter__(self):
        for data in super().__iter__():
            yield data, data.y
