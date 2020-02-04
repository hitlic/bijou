import torch
from bijou.utils import rename
import torch.nn.functional as F
import numpy as np

# --- metrics


@rename('acc')
def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


@rename('acc')
def masked_accuracy(pred, target):
    _, pred = pred.max(dim=1)
    correct = pred[target.mask].eq(target.data[target.mask]).sum()
    acc = correct / target.mask.float().sum()
    return acc


@rename('mse')
def masked_mse(pred, target):
    return F.mse_loss(torch.squeeze(pred[target.mask]), target.data[target.mask])


@rename('mae')
def masked_mae(pred, target):
    pred = torch.squeeze(pred[target.mask])
    target = target.data[target.mask]
    return torch.mean(torch.abs(pred - target))


def ordinal(pred, target):
    """
    真实排序下，对应的预测值的序号
    """
    frame = np.array([pred, target]).transpose()
    frame = frame[(-frame[:, 0]).argsort()]
    frame = np.concatenate([frame, np.expand_dims(np.arange(1, len(pred)+1), 1)], 1)
    frame = frame[(-frame[:, 1]).argsort()]
    pred_ids = frame[:, 2]
    return pred_ids


@rename('map')
def MAP(pred, target):
    """
    Mean average precision(MAP)
    """
    n = len(pred)
    target_ids = np.arange(1, len(pred)+1)
    pred_ids = ordinal(pred, target)

    def p_at_n(p_ids, t_ids, n):
        return len(set(p_ids[:n]).intersection(set(t_ids[:n])))/n
    return np.average([p_at_n(pred_ids, target_ids, i) for i in range(1, n+1)])


@rename('map')
def masked_MAP(pred, target):
    """
    Mean average precision(MAP)
    """
    pred = torch.squeeze(pred[target.mask]).detach().cpu().numpy()
    target = target.data[target.mask].detach().cpu().numpy()
    return MAP(pred, target)


def DCG_at_n(pred, target, n):
    """
    Discount Cumulative Gain (DCG@n)
    """
    frame = np.array([pred, target]).transpose()
    frame = frame[(-frame[:, 0]).argsort()]
    frame = frame[:n]
    return np.sum([t/np.log2(i+2) for i, (_, t) in enumerate(frame)])


def NDCG_at_n(n, pred, target):
    """
    Normalized discount cumulative gain (NDCG@n)
    """
    return DCG_at_n(pred, target, n)/DCG_at_n(target, target, n)


def masked_NDCG_at_n(n, pred, target):
    """
    Masked normalized discount cumulative gain (NDCG@n)
    """
    pred = torch.squeeze(pred[target.mask]).detach().cpu().numpy()
    target = target.data[target.mask].detach().cpu().numpy()
    return NDCG_at_n(n, pred, target)


# --- losses

def masked_nll_loss(pred, target):
    return F.nll_loss(pred[target.mask], target.data[target.mask])

def masked_cross_entropy(pred, target):
    return F.cross_entropy(pred[target.mask], target.data[target.mask])

def masked_TOP_loss(pred, target):
    """
    Top One Probability(TOP) loss，from <<Learning to Rank: From Pairwise Approach to Listwise Approach>>
    """
    pred = torch.squeeze(pred[target.mask])
    target = target.data[target.mask]
    pred_p = torch.softmax(pred, 0)
    target_p = torch.softmax(target, 0)
    loss = torch.mean(-torch.sum(target_p*torch.log(pred_p)))
    return loss
