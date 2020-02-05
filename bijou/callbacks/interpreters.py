from .basic_callbacks import Callback
from ..utils import ToolBox as tbox
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import networkx as nx


def test_phase(interp, phase):
    assert phase in ['train', 'val', 'test'], '"phase" must be "train", "val" or "test"'
    bs = getattr(interp, f'_ybs_{phase}')
    if not bs:
        raise Exception(f'!!! "{phase}" is not performed')


class InterpreterBase(Callback):
    _name = 'interpreter'

    def __init__(self, task_type, learner, multi_out):
        """
        Args:
            task_type: type of leaning task
            learner: learner
            multi_out: is the model have multi output
        """
        super().__init__(as_attr=True, learner=learner)
        assert task_type in ['classify', 'regress']  # 分类、回归任务
        self.task_type = task_type
        self.multi_out = multi_out
        self.cpu = torch.device('cpu')

        self._xbs_train = []
        self._xbs_val = []
        self._xbs_test = []
        self._ybs_train = []
        self._ybs_val = []
        self._ybs_test = []
        self._predbs_train = []
        self._predbs_val = []
        self._predbs_test = []

    def begin_fit(self):
        self.epoch_th = 0
        if self.task_type == 'classify':
            self.c_matrix_train = None
            self.c_matrix_val = None
            self.c_matrix_test = None
            self.c_dict_train = None
            self.c_dict_val = None
            self.c_dict_test = None

    def begin_epoch(self):
        self.epoch_th += 1

    def after_pred(self):
        if self.epoch_th == self.epochs or self.state == 'test':  # last epoch or test epoch
            getattr(self, f'_predbs_{self.state}').append(self.predb)
            getattr(self, f'_ybs_{self.state}').append(self.yb)
            getattr(self, f'_xbs_{self.state}').append(self.xb.to(self.cpu))  # 放入CPU中避免占用GPU存储

    def cat(self, phase):
        raise Exception("To be rewrited!!!")

    def create_top_indices(self, top_indices, phase):
        batch_size = len(getattr(self, f'_ybs_{phase}')[0])
        top_index = {
            'batch': [i // batch_size for i in top_indices],
            'index': [i % batch_size for i in top_indices]
        }
        return top_index

    def top_data(self, metric, phase='train', largest=True, k=0):
        """
        返回metric指标最大（largest=True）或最小（largest=False）的k个数据。
        Args:
            metric: 计算指标
            phase: 分析对象，'train', 'val' 或 'test'分别表示训练数据、验证数据或测试数据
            largest: 返回最大还是最小的k个数据
        Return: (top_scores, top_xs, top_ys, top_preds, top_index)即
                (metric最大值, 最大值对应的x, 最大值对应的y, 最大值对应的pred, {最大值所在batch及在batch中的位置})
        """
        test_phase(self, phase)

        self.cat(phase)
        scores = []
        with torch.no_grad():
            for preb, yb in zip(getattr(self, f'_predbs_{phase}'), getattr(self, f'_ybs_{phase}')):
                scores.append(metric(preb, yb).detach().cpu())
        scores = torch.cat(scores)
        if k == 0:
            k = len(scores)
        top_k = scores.topk(k, largest=largest)
        top_indices = top_k.indices.numpy()

        top_scores = top_k.values.numpy()
        # top_xs = getattr(self, f'_x_{phase}')[top_indices]
        top_xs = [getattr(self, f'_x_{phase}')[i] for i in top_indices]
        top_ys = getattr(self, f'_y_{phase}')[top_indices].numpy()
        top_preds = getattr(self, f'_pred_{phase}')[top_indices].numpy()

        return top_scores, top_xs, top_ys, top_preds, self.create_top_indices(top_indices, phase)

    def confusion_matrix(self, phase='train', return_dict=False):
        """
        混淆矩阵
        """
        if self.task_type != 'classify':
            raise Exception('Confusion matrix only in "classify" tasks!')
        test_phase(self, phase)
        # 若已存在，不必再计算
        if return_dict:
            c_dict = getattr(self, f'c_dict_{phase}')
            if c_dict is not None:
                return c_dict
        else:
            c_matrix = getattr(self, f'c_matrix_{phase}')
            if c_matrix is not None:
                return c_matrix

        self.cat(phase)

        pred_size = getattr(self, f'_pred_{phase}').size()
        if len(pred_size) == 1:  # for sigmoid output classification
            self.class_num = 2
        else:
            self.class_num = pred_size[1]

        c_dict = {}
        for x, y, pred in zip(getattr(self, f'_x_{phase}'),
                              getattr(self, f'_y_{phase}'),
                              getattr(self, f'_pred_{phase}')):
            if len(pred_size) == 1:
                key = (int(y.numpy().tolist()), pred.round().int().numpy().tolist())
            else:
                key = (int(y.numpy().tolist()), int(pred.argmax().numpy().tolist()))
            if key not in c_dict:
                c_dict[key] = []
            c_dict[key].append(x)

        c_matrix = np.zeros([self.class_num, self.class_num], dtype=int)
        for key, item in c_dict.items():
            c_matrix[key[0], key[1]] = len(item)

        setattr(self, f'c_dict_{phase}', c_dict)
        setattr(self, f'c_matrix_{phase}', c_matrix)

        if return_dict:
            return c_dict
        else:
            return c_matrix

    def plot_confusion(self, phase='train', title='Confusion matrix', class_names=None,
                       normalize=False, norm_dec=2, cmap='Blues', **kwargs):
        """
        画出混淆矩阵
        """
        if self.task_type != 'classify':
            raise Exception('Confusion matrix only in "classify" tasks!')
        test_phase(self, phase)

        c_matrix = self.confusion_matrix(phase)
        data_size = c_matrix.sum()
        if normalize:
            c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]

        fig = plt.figure(**kwargs)

        plt.imshow(c_matrix, interpolation='nearest', cmap=cmap)
        plt.title(f'{title} - {phase}({data_size})')
        if class_names and len(tbox.listify(class_names)) == self.class_num:
            tick_marks = np.arange(self.class_num)
            plt.xticks(tick_marks, class_names, rotation=90)
            plt.yticks(tick_marks, class_names, rotation=0)

        thresh = c_matrix.max() / 2.
        for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
            coeff = f'{c_matrix[i, j]:.{norm_dec}f}' if normalize else f'{c_matrix[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                     color="white" if c_matrix[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(self.class_num-.5, -.5)

        plt.tight_layout()
        plt.ylabel('Real')
        plt.xlabel('Pred')
        plt.grid(False)
        fig.subplots_adjust(bottom=0.1)
        return fig

    def most_confused(self, phase='train', k=5):
        """
        错分最多的情况
        """
        if self.task_type != 'classify':
            raise Exception('Confusion matrix only in "classify" tasks!')
        test_phase(self, phase)

        c_dict = self.confusion_matrix(phase, return_dict=True)
        c_items = [c for c in c_dict.items() if c[0][0] != c[0][1]]
        return sorted(c_items, key=lambda e: len(e[1]), reverse=True)[:k]


class Interpreter(InterpreterBase):
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
            setattr(self, f'_x_{phase}', torch.cat(getattr(self, f'_xbs_{phase}')))
        if getattr(self, f'_y_{phase}', None) is None:
            setattr(self, f'_y_{phase}', torch.cat(getattr(self, f'_ybs_{phase}')).detach().cpu())
        if getattr(self, f'_pred_{phase}', None) is None:
            if not self.multi_out:
                setattr(self, f'_pred_{phase}', torch.cat(getattr(self, f'_predbs_{phase}')).detach().cpu())
            else:
                predbs = getattr(self, f'_predbs_{phase}')
                predbs = [torch.cat(predb, 1) for predb in predbs]
                setattr(self, f'_pred_{phase}', torch.cat(predbs).detach().cpu())


def data2g(data):
    if data.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    edge_index = zip(data.edge_index[0].numpy(), data.edge_index[1].numpy())
    g.add_edges_from(edge_index)
    return g


class PyGGraphInterpreter(InterpreterBase):
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
                data_list.extend(b.to_data_list())
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
        g = data2g(data)
        nx.draw(g, **kwargs)


def get_size(value, max_value, min_value, max_size, min_size, largest):
    size = 0
    if largest:
        size = (value - min_value)/(max_value - min_value)
    else:
        size = (max_value - value)/(max_value - min_value)
    size = size * (max_size - min_size) + min_size
    return round(size)


class PyGNodeInterpreter(InterpreterBase):
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
            # batches of y
            ys = getattr(self, f'_ybs_{phase}')[0]
            mask = ys.mask
            ys = ys.data[mask]
            setattr(self, f'_y_{phase}', ys.detach().cpu())
            setattr(self, f'_mask_{phase}', mask)

            # batches of x
            data = getattr(self, f'_xbs_{phase}')[0]
            setattr(self, f'_x_{phase}', data.x[mask])

            # batches of pred
            if not self.multi_out:
                setattr(self, f'_pred_{phase}', torch.cat(getattr(self, f'_predbs_{phase}'))[mask].detach().cpu())
            else:
                predbs = getattr(self, f'_predbs_{phase}')
                predbs = [torch.cat(predb, 1)[mask] for predb in predbs]
                setattr(self, f'_pred_{phase}', torch.cat(predbs).detach().cpu())

    def create_top_indices(self, top_indices, phase):
        mask = getattr(self, f'_mask_{phase}').numpy()
        index = np.where(mask == True)[0]
        top_index = {
            'index': [index[i] for i in top_indices]
        }
        return top_index

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

        data = getattr(self, f'_xbs_{phase}')[0]
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

        g = data2g(data)
        node_size = [20 if not node_size.get(i) else node_size[i] for i in g.nodes]

        if layout is None:
            layout = nx.random_layout
        plt.figure()
        pos = layout(g)
        nx.draw_networkx_nodes(g, pos=pos, node_size=node_size, **kwargs)
        nx.draw_networkx_edges(g, pos=pos, **kwargs)
        nx.draw_networkx_labels(g, pos, labels=label_dict, **kwargs)

        return g, node_size, label_dict
