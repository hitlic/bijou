"""
后续：
  数据预处理
  各种计算指标
  各种优化技术：参数初始化
"""

import torch
from bijou.data import DataLoaderBase
from bijou.utils import ToolBox as tbox
import bijou.callbacks as cbks
import matplotlib.pyplot as plt
from pathlib import Path
import time


class Learner():
    def __init__(self, model, opt, loss_func, data=None, metrics=None, callbacks=None, device=None):
        """
        Args:
            model: pytorch 模型
            opt: 优化器
            loss_func: 损失函数
            data: DataBunch
            metrics: 性能评价指标或评价指标列表
            callbacks: callbacks对象或类，或者其列表
            device: cpu or gpu device
        """

        self.model, self.opt, self.data, self.loss_func = model, opt, data, loss_func
        self.state = 'train'  # 'train', 'val', 'test'
        self.messages = {}    # 存放需要在不同callbacks之间共享的信息
        self.epoch = 0
        self.best_loss = float('inf')

        cbs = tbox.listify(callbacks)
        # 添加一些必要的回调
        cbs.append(cbks.Recorder(as_attr=True))
        cbs += [cbks.StatesCallback(),
                cbks.CudaCallback(device=device),
                cbks.ProgressBarCallback(),
                cbks.AvgStatsCallback(metrics=metrics)]

        cb_list = []
        for cb in cbs:
            if isinstance(cb, cbks.Callback):
                cb_obj = cb
            else:
                cb_obj = cb()

            if cb_obj.as_attr:
                setattr(self, cb_obj.name, cb_obj)
            cb_obj.set_learner(self)
            cb_list.append(cb_obj)

        self.cbs = cb_list

    def __call__(self, cb_name, reverse=False):
        res = False
        cb_list = sorted(self.cbs, key=lambda x: x._order)  # pylint: disable=protected-access
        if reverse:
            cb_list = reversed(cb_list)
        for cb in cb_list:
            res = cb(cb_name) or res
        return res

    def one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb
            self('begin_batch', reverse=False)
            self.predb = self.model(*self.xb)  # There may be multiple inputs
            self('after_pred', reverse=True)
            self.loss = self.loss_func(self.predb, self.yb)
            self('after_loss', reverse=True)
            if self.state != 'train':  # 若不是训练状态，则结束batch
                return
            self.loss.backward()
            self('after_backward', reverse=True)
            self.opt.step()
            self('after_step', reverse=True)
            self.opt.zero_grad()
        except cbks.CancelBatchException:
            self('after_cancel_batch', reverse=True)
        finally:
            self('after_batch', reverse=True)

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in dl:
                self.one_batch(xb, yb)
        except cbks.CancelEpochException:
            self('after_cancel_epoch', reverse=True)

    def fit(self, epochs):
        self.epochs, self.loss = epochs, torch.tensor(0.)
        start_epoch = self.epoch
        try:
            self('begin_fit', reverse=False)
            for epoch in range(epochs):
                if not self('begin_epoch', reverse=False):
                    self.all_batches(self.data.train_dl)

                if self.data.valid_dl is not None:
                    with torch.no_grad():
                        if not self('begin_validate', reverse=False):
                            self.all_batches(self.data.valid_dl)
                self('after_epoch', reverse=True)
                self.epoch = epoch + start_epoch + 1  # resume from checkpoint

        except cbks.CancelTrainException:
            self('after_cancel_train', reverse=True)
        finally:
            self('after_fit', reverse=True)
            # self.model, self.opt, self.loss_func, self.data = None, None, None, None

    def test(self, test_dl):
        if len(test_dl) == 0:
            return
        self.test_dl = test_dl
        if not self('begin_test', reverse=False):
            with torch.no_grad():
                self.all_batches(test_dl)
        self('after_test', reverse=True)
        self.test_dl = None

    def predict(self, dataset):
        """
        dataset: Dataloader或者可直接计算的数据如Tensor或PyG Data
        """

        if hasattr(dataset, 'to'):  # whether it has the "to(device)" method
            dataset = [dataset]
        is_dl = isinstance(dataset, DataLoaderBase)

        preds = []
        for batch in dataset:
            self.predict_data = batch[0] if is_dl else batch
            self('begin_predict', reverse=False)
            with torch.no_grad():
                preds.append(self.model(*self.predict_data))
        if preds and isinstance(preds[0], tuple):  # multiple output
            return preds # predict result of batches
        else:
            return torch.cat(preds, 0)

    def fit_one_cycle(self, epochs, stage=(0.3, 0.7), start_lr=0.01, high_lr=0.5, end_lr=0.01):
        sched = cbks.combine_scheds(stage,
                                    cbks.cos_1cycle_anneal(start_lr, high_lr, end_lr)
                                    )  # pylint: disable=no-value-for-parameter
        lr_shed_cb = cbks.ParamScheduler('lr', sched, learner=self)
        self.cbs += [lr_shed_cb]
        self.fit(epochs)

        self.cbs.remove(lr_shed_cb)

    def find_lr(self, max_iter=100, min_lr=1e-6, max_lr=10, skip_last=5):
        lr_find_cbk = cbks.LR_Find(max_iter, min_lr, max_lr, learner=self)
        self.cbs += [lr_find_cbk]
        self.fit(max_iter)

        self.recorder.plot_lr_loss(skip_last=skip_last)
        plt.xlabel('learning rate')
        plt.ylabel('loss')

        self.cbs.remove(lr_find_cbk)

    def save_model(self, path='./model.pkl'):
        """
        保存模型
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model, path)

    @classmethod
    def load_model(cls, path):
        """
        加载模型
        """
        model = torch.load(path)
        return model

    def load_checkpoint(self, check_name=None, check_latest=False, check_best=False,
                        check_opt=True, path='./checkpoints'):
        """
        load learner checkpoint
        Args:
            check_name: checkpoint file name
            check_latest: load latest checkpoint
            check_best: load best loss checkpoint
            check_opt: load opt state or not
            path: checkpoint folder path
        """
        path = Path(path)
        name_dict = {f.name: [float(e) for e in f.name[:-5].split('-')] for f in path.iterdir()
                     if not f.is_dir() and f.name.endswith('.ckpt')}
        # find latest or best checkpoint
        if not check_name:
            if check_name is None and not check_latest and not check_best:
                check_latest = True
            if check_latest:
                check_name = max(name_dict.items(), key=lambda e: e[1][0])[0]
                print('- Loadding the latest checkpoint ...')
            elif check_best:
                check_name = min(name_dict.items(), key=lambda e: e[1][1])[0]
                print('- Loadding the best checkpoint ...')
            if not check_name:
                print('\nNo checkpoint!')
                return
        else:
            if len(check_name) < 25:
                ckn = [f for f in name_dict.keys() if f.startswith(check_name)]
                if ckn:
                    check_name = ckn[0]
            elif not check_name.endswith('.ckpt'):
                check_name = check_name.strip() + '.ckpt'

        check_info = torch.load(path/check_name)

        # load learner state
        self.epoch = 0  # check_info['epoch'] + 1
        self.best_loss = check_info['best_loss']

        # load mode state
        model_dict = self.model.state_dict()
        update_state = {k: v for k, v in check_info['model_state'].items() if k in model_dict.keys()}
        model_dict.update(update_state)
        self.model.load_state_dict(model_dict)

        # load opt sate
        if check_opt:
            self.opt.load_state_dict(check_info['opt_state'])

    def checkpoint(self, path='./checkpoints'):
        """
        save learner checkpoint
        """
        path = Path(path)
        if not path.exists():
            path.mkdir()
        check_info = {
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'model_state': self.model.state_dict(),
            'opt_state': self.opt.state_dict(),
        }
        check_name = f'{int(time.time())}-{self.best_loss:0.10f}-{self.epoch+1}'
        torch.save(check_info, path/f'{check_name}.ckpt')
