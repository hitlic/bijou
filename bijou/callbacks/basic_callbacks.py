import torch
import re
import matplotlib.pyplot as plt
from bijou.utils import ToolBox as tbox
from tqdm import tqdm


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


def camel2snake(name):
    """
    生成callback的名字
    """
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')

    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def dictformat(dic):
    return str({k: f'{v:0.6f}' for k, v in dic.items()})


class Callback():
    """
    Callback的基类
    """
    _order = 0
    _name = None

    def __init__(self, as_attr=False, learner=None):
        """
        Args:
            as_attr: 是否将callback对象作为learner的属性
            learner: callback所属的Learner，若为空则后续可调用set_learner设置
        """
        self.as_attr = as_attr
        self.learner = learner

    def set_learner(self, learner):
        self.learner = learner

    def __getattr__(self, k):
        return getattr(self.learner, k)

    @property
    def name(self):
        if self._name is None:
            name = re.sub(r'Callback$', '', self.__class__.__name__)
            self._name = camel2snake(name or 'callback')
        return self._name

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class AvgStats:
    def __init__(self, metrics, state):
        self.metrics, self.state = tbox.listify(metrics), state
        self.metric_names = ['loss'] + [m.__name__ for m in self.metrics]  # 构造metrics 名称

    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        if self.count > 0:
            return dict(zip(self.metric_names, [o/self.count for o in self.all_stats]))
        else:
            return {}

    def __repr__(self):
        if not self.count:
            return ""
        return f"{self.state}: {self.avg_stats}"

    def accumulate(self, learner):
        bn = len(learner.xb)  # .shape[0]
        self.tot_loss += learner.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(learner.predb, learner.yb).item() * bn


class AvgStatsCallback(Callback):
    """
    该类型的回调最后只用一个，如果需要多个指标，则将多个指标组成列表，传到metrics参数即可
    """
    _order = 1

    def __init__(self, metrics, **kwargs):
        super().__init__(**kwargs)
        self.train_stats = AvgStats(metrics, 'train')
        self.val_stats = AvgStats(metrics, 'val')
        self.test_stats = AvgStats(metrics, 'test')

    def begin_fit(self):
        self.learner.messages['metric_values_batch'] = {'train': None, 'val': None, 'test': None}
        self.learner.messages['metric_values_epoch'] = {}

    def begin_epoch(self):
        self.train_stats.reset()
        self.val_stats.reset()

    def after_epoch(self):
        self.learner.messages['metric_values_epoch']['train'] = self.train_stats.avg_stats
        self.learner.messages['metric_values_epoch']['val'] = self.val_stats.avg_stats

        # update best loss
        loss = self.messages['metric_values_epoch']['train']['loss']
        if loss < self.best_loss:
            self.learner.best_loss = loss

    def begin_test(self):
        self.test_stats.reset()

    def after_loss(self):
        stats = getattr(self, f'{self.state}_stats')
        with torch.no_grad():
            stats.accumulate(self.learner)

    def after_batch(self):
        self.learner.messages['metric_values_batch'][self.state] = getattr(self, f'{self.state}_stats').avg_stats


class ProgressBarCallback(Callback):
    _order = 0
    train_message = ''
    val_message = ''
    test_message = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def begin_epoch(self):
        batch_num = len(self.data.train_dl)
        self.pbar = tqdm(total=batch_num,
                         bar_format=f'E {self.epoch + 1:<4} ' + 'B {n_fmt} {l_bar} {rate_fmt} | {postfix[0]}',
                         unit=' batch', postfix=[''])

    def begin_validate(self):
        self.pbar.reset(total=len(self.data.valid_dl))

    def begin_test(self):
        self.pbar = tqdm(total=len(self.test_dl),
                         bar_format=f'TEST   ' + 'B {n_fmt} {l_bar} {rate_fmt} | {postfix[0]}',
                         unit=' batch', postfix=[''])

    def after_batch(self):
        if self.learner.messages['metric_values_batch'][self.state]:
            message = f'{self.state}-'
            message += str(dictformat(self.messages['metric_values_batch'][self.state])).replace("'", '')
            setattr(self, f'{self.state}_message', message)
            self.learner.messages['metric_values_batch'][self.state] = None

        if self.state == 'train':
            self.pbar.postfix[0] = self.train_message
        elif self.state == 'val':
            self.pbar.postfix[0] = self.train_message + '  ' + self.val_message
        elif self.state == 'test':
            self.pbar.postfix[0] = self.test_message
        self.pbar.update()

    def after_epoch(self):
        self.pbar.close()

    def after_test(self):
        self.pbar.close()


class StatesCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def begin_fit(self):
        self.learner.p_epochs = 0.
        self.learner.n_iter = 0

    def after_batch(self):
        if self.state != 'train':
            return
        self.learner.p_epochs += 1./self.iters
        self.learner.n_iter += 1

    def begin_epoch(self):
        self.learner.p_epochs = self.epoch
        self.model.train()
        self.learner.state = 'train'

    def begin_validate(self):
        self.model.eval()
        self.learner.state = 'val'

    def begin_test(self):
        self.model.eval()
        self.learner.state = 'test'


class Recorder(Callback):
    metric_of_epochs = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if self.state != 'train':
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())

    def after_epoch(self):
        self.metric_of_epochs.append(self.messages['metric_values_epoch'].copy())

    def plot_lr(self, pgid=-1, **kwargs):
        plt.plot(self.lrs[pgid], **kwargs)

    def plot_loss(self, skip_last=0, **kwargs):
        plt.plot(self.losses[:len(self.losses)-skip_last], **kwargs)

    def plot(self):
        """
        plot learning and losses
        """
        fig = plt.figure(figsize=[10, 4])

        plt.subplot(121)
        self.plot_lr()
        plt.xlabel('n iter')
        plt.ylabel('learning rate')

        plt.subplot(122)
        self.plot_loss(c='r')
        plt.xlabel('n iter')
        plt.ylabel('loss')
        fig.subplots_adjust(wspace=0.3, bottom=0.18)

    def plot_lr_loss(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])

    def plot_metrics(self):
        trains_vals = ld2dl(self.metric_of_epochs)
        trains = ld2dl(trains_vals['train'])
        vals = trains_vals.get('val')  # 可能无验证集
        if vals:
            vals = ld2dl(vals)
        plt.figure(figsize=[len(trains) * 5, 4])
        for i, m in enumerate(trains):
            plt.subplot(1, len(trains), i + 1)
            plt.plot(trains[m], label='train')
            if vals:
                plt.plot(vals[m], label='val')
            plt.title(m, y=-0.2)
            plt.legend()
        plt.subplots_adjust(bottom=0.2)


def ld2dl(ld):
    dl = {}
    for d in ld:
        for k, v in d.items():
            if k not in dl:
                dl[k] = []
            dl[k].append(v)
    return dl


class CudaCallback(Callback):
    _order = 0

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        # if isinstance(self.xb, (list, tuple)):
        #     self.learner.xb = [xs.to(self.device) for xs in self.xb]  # multi-inputs
        # else:
        #     self.learner.xb = [self.xb.to(self.device)]  # single-inputs
        self.learner.xb = self.to_device(self.xb)
        self.learner.yb = self.yb.to(self.device)

    def begin_predict(self):
        self.model.to(self.device)
        self.learner.predict_data = self.to_device(self.predict_data)
        # self.learner.predict_data = self.predict_data.to(self.device)

    def to_device(self, batch_data):
        if isinstance(batch_data, (list, tuple)):
            xb = [xs.to(self.device) for xs in batch_data]  # multi-inputs
        else:
            xb = [batch_data.to(self.device)]  # single-inputs
        return xb



class Checkpoints(Callback):
    _order = -1

    def __init__(self, epochs=1, path='./checkpoints', skip_worse=False, **kwargs):
        """
        Args:
            epochs: save checkpoint each 'epochs' epochs.
            path: path
            skip_worse: skip at worse loss epoch
        """
        super().__init__(**kwargs)
        self.per_epochs = epochs
        self.path = path
        self.skip_worse = skip_worse
        self.best_check_loss = float('inf')

    def after_epoch(self):
        if (self.epoch+1) % self.per_epochs == 0:
            if self.skip_worse:
                epoch_loss = self.messages['metric_values_epoch']['train']['loss']
                if epoch_loss < self.best_check_loss:
                    self.checkpoint(path=self.path)
                    self.best_check_loss = epoch_loss
            else:
                self.checkpoint(path=self.path)
