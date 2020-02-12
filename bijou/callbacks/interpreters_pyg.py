import networkx as nx
from .interpreters import InterpreterBase, GraphInterpreter, test_phase
import torch
import matplotlib.pyplot as plt


def pyg_data2g(data):
    if data.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    edge_index = zip(data.edge_index[0].numpy(), data.edge_index[1].numpy())
    g.add_edges_from(edge_index)
    return g


class PyGInterpreter(InterpreterBase):
    def __init__(self, task_type='classify', learner=None, multi_out=False):
        """
        Args:
            task_type: type of leaning task
            learner: learner
            multi_out: is the model have multi output
        """
        super().__init__(task_type, learner, multi_out)

    def cat(self, phase):
        if getattr(self, f'_x_{phase}', None) is None:
            databs = getattr(self, f'_xbs_{phase}')
            data_list = []
            for b in databs:
                data_list.extend(b[0].to_data_list())
            setattr(self, f'_x_{phase}', data_list)
        if getattr(self, f'_y_{phase}', None) is None:
            setattr(self, f'_y_{phase}', torch.cat(getattr(self, f'_ybs_{phase}')).detach().cpu())
        if getattr(self, f'_pred_{phase}', None) is None:
            if not self.multi_out:
                setattr(self, f'_pred_{phase}', torch.cat(getattr(self, f'_predbs_{phase}')).detach().cpu())
            else:
                predbs = getattr(self, f'_predbs_{phase}')
                predbs = [torch.cat(predb, 1) for predb in predbs]
                setattr(self, f'_pred_{phase}', torch.cat(predbs).detach().cpu())

    def plot_graph(self, data, **kwargs):
        g = pyg_data2g(data)
        nx.draw(g, **kwargs)


class PyGGraphInterpreter(GraphInterpreter):
    def get_features(self, data):
        return data[0].x

    def data2nxg(self, data):
        return pyg_data2g(data)
