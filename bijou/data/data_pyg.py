from torch_geometric.data import DataLoader as DataLoader_pyg
from .data import DataLoaderBase, MaskedTensor, GraphLoader

class PyGDataLoader(DataLoader_pyg, DataLoaderBase):

    def __iter__(self):
        for data in super().__iter__():
            yield data, data.y


class PyGGraphLoader(GraphLoader):
    """
    用于节点计算，仅包含单一的torch_geometric.data.data.Data的pyg_dataset
    """

    def __init__(self, pyg_dataset, phase, label_name='y'):
        """
        Args:
            pyg_data: torch_geometric.data.data.Data对象
            phase: 'train', 'val' or 'test'
        """
        pyg_data = pyg_dataset[0]
        assert phase in ['train', 'val', 'test'], 'param "phase" must be in [train, val, test]'
        assert getattr(pyg_data, label_name) is not None, f'Data must has a attribute "{label_name}"'
        mask = getattr(pyg_data, f'{phase}_mask')
        assert mask is not None, f'Data must has a attribute "{phase}_mask"'

        # GraphLoader must have following four attributes
        self.data = pyg_data
        self.label = MaskedTensor(getattr(self.data, label_name), mask)
        self.mask = mask
