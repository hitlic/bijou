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



def get_size(value, max_value, min_value, max_size, min_size, largest):
    size = 0
    if largest:
        size = (value - min_value)/(max_value - min_value)
    else:
        size = (max_value - value)/(max_value - min_value)
    size = size * (max_size - min_size) + min_size
    return round(size)


class PyGGraphInterpreter(GraphInterpreter):
    def get_features(self, data):
        return data[0].x

    def plot_graph(self, metric, k=0, largest=True, phase='train',
                   layout=None, max_node_size=500, min_node_size=100,
                   label_id=False, label_score=True, dec=2, **kwargs):
        """
        plot the PyG Data as Networkx graph
        Args:
            metric: top_data metric. 计算指标，与Learner中的指标不同，需要返回每个数据的计算结果，即无需reduction（如均值）
            k: 多少数据
            largest: 返回最大还是最小的数据
            phase: train, val or test
            layout: networkx layout函数
            max_node_size: 最大节点大小
            min_node_size: 最小节点大小
            label_id: id是否出现在节点label中
            label_score: metric的值是否出现在节点label中
            dec: 若metric值的小数位数
        """
        test_phase(self, phase)

        scores, _, _, _, indecies = self.top_data(metric, phase, largest, 0)
        if k == 0:
            k = len(scores)
        scores = scores[:k]
        indecies = indecies['index'][:k]

        data = getattr(self, f'_xbs_{phase}')[0][0]
        if data.num_nodes > 1000:
            print('\nALERT! The network is larger than 1000. Drawing may take a long time!\n')

        min_score, max_score = min([s for s in scores if s > 0]), max(scores)

        node_size = {}
        label_dict = {}
        for s, i in zip(scores, indecies):
            node_size[i] = get_size(s, max_score, min_score, max_node_size, min_node_size, largest)
            if label_score:
                label_dict[i] = '' if s == 0 else f'{s:0.{dec}f}'
                label_dict[i] = f'{i}-{label_dict[i]}' if label_id else label_dict[i]
            else:
                label_dict[i] = i

        g = pyg_data2g(data)
        node_size = [20 if not node_size.get(i) else node_size[i] for i in g.nodes]

        if layout is None:
            layout = nx.random_layout
        plt.figure()
        pos = layout(g)
        nx.draw_networkx_nodes(g, pos=pos, node_size=node_size, **kwargs)
        nx.draw_networkx_edges(g, pos=pos, **kwargs)
        nx.draw_networkx_labels(g, pos, labels=label_dict, **kwargs)

        return g, node_size, label_dict
