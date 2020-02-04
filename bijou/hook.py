from bijou.utils import ToolBox as tbox
from functools import partial


class ListContainer():
    def __init__(self, items):
        self.items = tbox.listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):  # idx为bool列表，返回idx中值为true的位置对应的元素
            assert len(idx) == len(self)  # bool mask
            return [o for m, o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del(self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10:
            res = res[:-1] + '...]'
        return res


class Hook():
    def __init__(self, m, fun, forward):
        if forward:
            self.hook = m.register_forward_hook(partial(fun, self))
        else:
            self.hook = m.register_backward_hook(partial(fun, self))
        self.name = str(m).replace('\n', '').replace(' ', '')

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, model, fun, forward=True):
        """
        model: pytorch module
        fun: hook function
        forward: True is forward hook, Flase is backward hook
        """
        ms = [m for m in model.modules() if len(list(m.children())) == 0]
        super().__init__([Hook(m, fun, forward) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()
