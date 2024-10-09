import torch as th
from math import prod
from torch.distributions import Distribution, Independent, Normal, Uniform
from typing import *


def shapeit(shape: th.Size | int | Iterable = None) -> th.Size:
    if shape is None:
        return th.Size()
    if isinstance(shape, th.Size):
        return shape
    if isinstance(shape, int):
        return th.Size([shape])
    return th.Size(list(shape))


def batchshape(X: Dict[str, th.Tensor], dim_X: Dict[str, th.Size]) -> th.Size:
    unknown = set(X.keys()) - (set(dim_X.keys()))
    assert len(unknown) == 0, \
        f"There are unknowns in given variables: {list(unknown)}"

    batch_shape = None
    for x in X:
        dim = dim_X[x]
        batch_dim = X[x].shape
        assert len(batch_dim) >= len(dim), \
            f"Dimensions mismatch for {x}: {batch_dim}, {dim}"
        feat_dim = th.Size([]) if len(dim) == 0 else batch_dim[-len(dim):]
        assert feat_dim == dim, f"Dimensions mismatch for {x}: {batch_dim}, {dim}"
        batch_shape_i = batch_dim if len(dim) == 0\
            else th.Size([]) if len(dim) == len(batch_dim)\
            else batch_dim[:-len(dim)]
        if batch_shape is None or batch_shape == batch_shape_i:
            batch_shape = batch_shape_i
        else:
            assert f"Batch size mismatch for {x}: {batch_shape}, {batch_shape_i}"

    return batch_shape


def batch(X: Dict[str, th.Tensor], dim_X: Dict[str, th.Size]) -> th.Tensor:
    batch_shape = batchshape(X, dim_X)
    xs = sorted(list(dim_X.keys()))
    return th.cat([X[x].reshape(*batch_shape, -1) for x in xs], dim=-1)


def unbatch(X: th.Tensor, dim_X: Dict[str, th.Size]) -> Dict[str, th.Tensor]:
    batch_shape = X.shape[:-1]
    xs = sorted(list(dim_X.keys()))
    X_dict, start_i = {}, 0
    for x in xs:
        start_j = int(start_i + prod(dim_X[x]))
        X_dict[x] = X[..., start_i:start_j].reshape((*batch_shape, *dim_X[x]))
        start_i = start_j
    return X_dict


def send_distribution_to(distr: Distribution, device: Optional[str | th.device | int] = None) -> Distribution:
    # For pytorch distribution
    if isinstance(distr, Independent):
        return Independent(send_distribution_to(distr.base_dist, device), distr.reinterpreted_batch_ndims)
    elif isinstance(distr, Normal):
        return Normal(distr.loc.to(device=device), distr.scale.to(device=device))
    elif isinstance(distr, Uniform):
        return Uniform(distr.low.to(device=device), distr.high.to(device=device))
    else:
        raise NotImplementedError("Unsupported distribution.")


def send_distributions_to(distrs: Dict[str, Distribution], device: Optional[str | th.device | int] = None) -> Distribution:
    return {
        u: send_distribution_to(distr, device)
        for u, distr in distrs.items()
    }
