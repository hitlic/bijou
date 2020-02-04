import torch
import operator
import math
from typing import Iterable


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


class ToolBox:
    """
    工具箱，乱七八糟的函数都放这里
    """
    @classmethod
    def test(cls, a, b, cmp, cname=None):
        if cname is None:
            cname = cmp.__name__
        assert cmp(a, b), f"{cname}:\n{a}\n{b}"

    @classmethod
    def test_eq(cls, a, b):
        cls.test(a, b, operator.eq, '==')

    @classmethod
    def near(cls, a, b):
        return torch.allclose(a, b, rtol=1e-3, atol=1e-5)

    @classmethod
    def test_near(cls, a, b):
        cls.test(a, b, cls.near)

    @classmethod
    def test_near_zero(cls, a, tol=1e-3):
        assert a.abs() < tol, f"Near zero: {a}"

    @classmethod
    def listify(cls, o):
        if o is None:
            return []
        if isinstance(o, list):
            return o
        if isinstance(o, str):
            return [o]
        if isinstance(o, Iterable):
            return list(o)
        return [o]

    @classmethod
    def setify(cls, o):
        return o if isinstance(o, set) else set(cls.listify(o))


class Initor:
    @classmethod
    def kaiming2(cls, x, a, use_fan_out=False):
        def gain(a):
            return math.sqrt(2.0 / (1 + a**2))
        nf, ni, *_ = x.shape
        rec_fs = x[0, 0].shape.numel()
        fan = nf*rec_fs if use_fan_out else ni*rec_fs
        std = gain(a) / math.sqrt(fan)
        bound = math.sqrt(3.) * std
        x.data.uniform_(-bound, bound)
