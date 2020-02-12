from .interpreters import InterpreterBase, GraphInterpreter
import torch
import networkx as nx
import dgl


class DGLInterpreter(InterpreterBase):
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
                data_list.extend(dgl.unbatch(b[0]))
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

    def plot_graph(self, data, directed=False, **kwargs):
        if directed:
            g = data.to_networkx()
        else:
            g = data.to_networkx().to_undirected()
        nx.draw(g, **kwargs)



class DGLGraphInterpreter(GraphInterpreter):
    def get_features(self, data):
        return data[1]

    def data2nxg(self, data):
        return data.to_networkx()
