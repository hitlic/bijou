from .basic_callbacks import Callback, CancelTrainException
from ..hook import Hooks
from ..utils import ToolBox as tbox
import torch
import torch.nn as nn
import math
from functools import partial
import matplotlib.pyplot as plt


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10, **kwargs):
        super().__init__(**kwargs)
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if self.state != 'train':
            return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg['lr'] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs, **kwargs):
        super().__init__(**kwargs)
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.p_epochs/self.epochs)

    def begin_batch(self):
        if self.state == 'train':
            self.set_param()


def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)
    return _inner


@annealer
def sched_linear(start, end, pos):
    return start + pos*(end-start)


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2


@annealer
def sched_no(start, end, pos):  # pylint: disable=unused-argument
    return start


@annealer
def sched_exp(start, end, pos):
    return start * (end/start) ** pos


def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]  # pylint: disable=no-value-for-parameter


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + tbox.listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


class LayerAnalysisCallback(Callback):
    _order = 3

    def __init__(self, forward=True, hist_span=10, **kwargs):
        """
        forward: True, 各层forward输出分析; False, 各层backward输出梯度分析
        hist_span: histgram的分析范围。前向分析时应当大一些，例如10；后向分析时应当小一些，例如0.1。
        """
        super().__init__(**kwargs)
        self.forward = forward
        self.hist_span = [-hist_span, hist_span]

    def begin_fit(self):
        def append_stats(hook, module, inputs, outputs):  # pylint: disable=unused-argument
            if not hasattr(hook, 'stats'):
                hook.stats = ([], [], [])
            if isinstance(outputs, tuple):  # backward hook
                outputs = outputs[0]
            means, stds, hists = hook.stats
            means.append(outputs[0].data.mean().cpu())
            stds .append(outputs[0].data.std().cpu())
            hists.append(outputs[0].data.cpu().histc(40, *self.hist_span))

        self.hooks = Hooks(self.model, append_stats, self.forward)

    def after_fit(self):
        self.hooks.remove()
        self.plot()

    def plot(self):
        mode = 'FORWARD' if self.forward else 'BACKWARD'
        # 均值与标准差
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        for h in self.hooks:
            ms, ss, _ = h.stats
            ax0.plot(ms, label=h.name)
            ax1.plot(ss, label=h.name)
        ax0.legend(prop={'size': 6})
        ax0.set_title(f"{mode}: Mean", fontsize=16)
        ax1.legend(prop={'size': 6})
        ax1.set_title(f"{mode}: Standard deviation", fontsize=16)

        # 各层输出值的分布
        figsize = (15, int(len(self.hooks)*0.7))
        fig, axes = plt.subplots(int(math.ceil(len(self.hooks)/3)), 3, figsize=figsize)
        [ax.axis('off') for ax in axes.flatten()]  # pylint:disable=expression-not-assigned
        for ax, h in zip(axes.flatten(), self.hooks):
            ax.axis('on')
            hist_matrix = torch.stack(h.stats[2]).t().float().log1p()
            extent = [0, hist_matrix.size()[1], *self.hist_span]
            im = ax.imshow(hist_matrix, origin='lower', extent=extent, aspect='auto')
            ax.set_title(h.name)
            fig.colorbar(im, ax=ax, shrink=1.0)
        fig.subplots_adjust(hspace=0.6, top=1-0.75/figsize[1])
        fig.suptitle(f'{mode}: Histogram of values by "log(1+x)"', fontsize=16)
        # plt.tight_layout()

        # 各层输出值中接近0的值的比例
        figsize = (15, int(len(self.hooks)*0.7))
        fig, axes = plt.subplots(int(math.ceil(len(self.hooks)/3)), 3, figsize=figsize)
        [ax.axis('off') for ax in axes.flatten()]  # pylint:disable=expression-not-assigned
        for ax, h in zip(axes.flatten(), self.hooks):
            ax.axis('on')
            hist_matrix = torch.stack(h.stats[2]).t().float()
            tiny_ratio = hist_matrix[19:22].sum(0)/hist_matrix.sum(0)
            ax.plot(tiny_ratio)
            ax.set_ylim(0, 1.02)
            ax.set_title(h.name)
        fig.subplots_adjust(hspace=0.6, top=1-0.75/figsize[1])
        fig.suptitle(f'{mode}: Fraction of tiny values', fontsize=16)


LayerOutputAnalysisHookCallback = LayerAnalysisCallback


class EarlyStopping(Callback):
    _order = -10000  # 保证最后执行

    def __init__(self, monitor='train', patience=10, min_delta=0., **kwargs):
        """
        Args:
            monitor: train loss or val loss
            patience: max patience epochs of getting worse
            min_delta: 小于 min_delta 的提升被认为没有变化
        """
        super().__init__(**kwargs)
        assert monitor in ['train', 'val'], '"monitor" must be "train" or "val"'
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.patience_num = 0

    def after_epoch(self):
        loss = self.messages['metric_values_epoch'][self.monitor]['loss']
        if loss > self.best_loss - self.min_delta:
            self.patience_num += 1
        else:
            self.patience_num = 0
        if self.patience_num >= self.patience:
            print('\n ... Early stopping is triggered!\n')
            raise CancelTrainException()


class GradientClipping(Callback):
    def __init__(self, max_norm=0., **kwargs):
        super().__init__(**kwargs)
        self.max_norm = max_norm

    def after_backward(self):
        if self.max_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)



if __name__ == '__main__':
    pass
    # a = torch.arange(0, 100)
    # p = torch.linspace(0.01, 1, 100)

    # annealings = "NO LINEAR COS EXP".split()
    # fns = [sched_no, sched_lin, sched_cos, sched_exp]
    # for fn, t in zip(fns, annealings):
    #     f = fn(2, 1e-2)
    #     plt.plot(a, [f(o) for o in p], label=t)
    # plt.legend()
    # plt.show()

    # sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])  # pylint: disable=no-value-for-parameter
    # plt.plot(a, [sched(o) for o in p])
    # plt.show()
