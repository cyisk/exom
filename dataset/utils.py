import torch as th
import warnings
from torch.masked import as_masked_tensor
from typing import *


def gather_fn(fn, *elements) -> Any:
    # Gather bottom data by 'fn' while keeping data structures
    if len(elements) == 0:
        raise ValueError(
            f"Expect at least one element, but none is given.")
    if not all(type(element) == type(elements[0]) for element in elements):
        raise ValueError(
            f"Expect all elements have the same type.")

    if isinstance(elements[0], dict):
        if not all(set(element.keys()) == set(elements[0].keys()) for element in elements):
            raise ValueError(
                f"Expect all dict elements have the same keys.")
        return {
            gather_fn(
                fn=fn,
                *[element[k] for element in elements],
            ) for k in elements[0].keys()
        }
    elif isinstance(elements[0], list):
        if not all(len(element) == len(elements[0]) for element in elements):
            raise ValueError(
                f"Expect all list elements have the same length.")
        return [
            gather_fn(
                fn=fn,
                *[element[i] for element in elements],
            ) for i in range(len(elements[0]))
        ]
    elif isinstance(elements[0], tuple):
        if not all(len(element) == len(elements[0]) for element in elements):
            raise ValueError(
                f"Expect all list elements have the same length.")
        return (
            gather_fn(
                fn=fn,
                *[element[i] for element in elements],
            ) for i in range(len(elements[0]))
        )
    elif isinstance(elements[0], th.Tensor):
        if not all(element.shape == elements[0].shape for element in elements):
            raise ValueError(
                f"Expect all tensor elements have the same shape.")
        return fn(elements)
    else:
        return elements


def indicator_to_set(indicator: th.Tensor,
                     features: Dict[str, int],
                     ) -> Set[str]:
    assert indicator.dim() == 1 and indicator.size(0) == sum(features.values())
    res = set()
    last_dim = 0
    for v in features:
        if th.any(indicator[last_dim:last_dim + features[v]]):
            res.add(v)
        last_dim += features[v]
    return res


def to_float_tensor(x: Any) -> th.Tensor:
    return th.tensor(x).float() if not isinstance(x, th.Tensor) else x


def batch_expand(x: th.Tensor, shape: th.Size):
    x_dim = x.dim()
    x = x[*([None]*len(shape)), ...]
    return x.expand(*shape, *([-1]*x_dim))


def feature_expand(x: th.Tensor, shape: th.Size):
    x_dim = x.dim()
    x = x[..., *([None]*len(shape))]
    return x.expand(*([-1]*x_dim), *shape)


def feature_masked_select(x: th.Tensor, mask: th.Tensor) -> th.Tensor:
    assert x.shape[:mask.dim()] == mask.shape
    if x.dim()-mask.dim() > 0:
        feat_shape = x.shape[-(x.dim()-mask.dim()):]
    else:
        feat_shape = th.Size()
    mask = feature_expand(mask, feat_shape)
    x = th.masked_select(x, mask)
    return x.reshape(-1, *feat_shape)


def feature_masked_scatter(x: th.Tensor, mask: th.Tensor, default: float = 0) -> th.Tensor:
    assert x.size(0) == mask.float().sum()
    feat_shape = x.shape[1:]
    mask = feature_expand(mask, feat_shape)
    feat = th.full_like(mask, default).to(x.dtype)
    return th.masked_scatter(feat, mask, x)


def masked_mean(input: th.Tensor, mask: Optional[th.BoolTensor] = None, dim: Any = None):
    with warnings.catch_warnings(action='ignore'):
        masked_input = as_masked_tensor(input, mask)
        mean = masked_input.mean(dim=dim)
        mean = mean.to_tensor(0)
    return mean


def masked_std(input: th.Tensor, mask: Optional[th.BoolTensor] = None, dim: Any = None):
    with warnings.catch_warnings(action='ignore'):
        masked_input = as_masked_tensor(input, mask)
        std = masked_input.std(dim=dim)
        std = std.to_tensor(0)
    return std


def masked_logsumexp(input: th.Tensor, mask: Optional[th.BoolTensor] = None, dim: Any = None):
    with warnings.catch_warnings(action='ignore'):
        masked_input = as_masked_tensor(input, mask)
        logsumexp = masked_input.exp().sum(dim=dim).log()
        logsumexp = logsumexp.to_tensor(0)
    return logsumexp


def masked_reduce(reduce: str,
                  x: th.Tensor,
                  mask: th.BoolTensor,
                  dim: int | th.Size | None = None,
                  ) -> th.Tensor:
    if mask is None:
        mask = th.ones_like(x).bool()
    elif mask.dim() < x.dim():
        assert mask.shape == x.shape[:mask.dim()]
        feat_shape = x.shape[-(x.dim()-mask.dim()):]
        mask = feature_expand(mask, feat_shape)

    if reduce == 'sum':
        x[~mask] = 0
        x = x.sum(dim=dim)
    elif reduce == 'mean':
        x[~mask] = 0
        x = x.sum(dim=dim)
        x = x / mask.float().sum(dim=dim)
    elif reduce == 'min':
        x[~mask] = th.inf
        x = x.amin(dim=dim)
    elif reduce == 'max':
        x[~mask] = -th.inf
        x = x.amax(dim=dim)
    else:
        raise ValueError("Unsupported reduce.")

    return x
