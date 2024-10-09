import torch as th
import torch.nn as nn
from functools import partial
from typing import *

from zuko.nn import MLP

from model.zuko.nn import *
from dataset.utils import *


class ConcatAggregator(nn.Module):
    def __init__(self,
                 hyper: nn.Module,
                 **kwargs,
                 ) -> None:
        super().__init__()

        # From multicontext
        self._hyper = hyper

    def preprocess(self,
                   x: th.Tensor,
                   c: th.Tensor,
                   c_adj: th.BoolTensor,
                   w_j: th.BoolTensor,
                   bake_mask: Callable,
                   ):
        max_joint_size = c.size(-2)
        # c: [..., feature_size, multi_context_size * context_size]
        c = th.cat([
            c[..., i, :]
            for i in range(max_joint_size)
        ], dim=-1)
        expand_shape = x.shape[:-c.dim()]
        c = batch_expand(c, expand_shape)
        x = th.cat((x, c), dim=-1)
        # adj: [..., feature_size, feature_size + multi_context_size * context_size]
        c_adj = th.cat([
            c_adj[..., i, :, :]
            for i in range(max_joint_size)
        ], dim=-1)
        adj = bake_mask(c_adj=c_adj)
        return x, adj, None

    def forward(self,
                x: th.Tensor,
                adj: th.BoolTensor,
                _=None,
                ) -> th.Tensor:
        # Phi: [..., multi_context_size, out_features]
        phi = self._hyper(x, adj)
        return phi


class ReduceAggregator(nn.Module):
    def __init__(self,
                 hyper: nn.Module,
                 reduce: str = 'sum',
                 **kwargs,
                 ) -> None:
        super().__init__()

        # From multicontext
        self._hyper = hyper

        # Reduce function
        if reduce == 'sum':
            self._reduce_fn = partial(masked_reduce, reduce='sum', dim=-2)
        elif reduce == 'mean':
            self._reduce_fn = partial(masked_reduce, reduce='mean', dim=-2)
        elif reduce == 'min':
            self._reduce_fn = partial(masked_reduce, reduce='min', dim=-2)
        elif reduce == 'max':
            self._reduce_fn = partial(masked_reduce, reduce='max', dim=-2)
        else:
            raise ValueError("Unsupported reduce function.")

    def preprocess(self,
                   x: th.Tensor,
                   c: th.Tensor,
                   c_adj: th.Tensor,
                   w_j: th.BoolTensor,
                   bake_mask: Callable,
                   ):
        # c: [..., multi_context_size, context_size]
        x = x[..., None, :].expand(*([-1]*(x.dim()-1)), c.size(-2), -1)
        expand_shape = x.shape[:-c.dim()]
        c = batch_expand(c, expand_shape)
        x = th.cat((x, c), dim=-1)
        # adj: [..., feature_size + multi_context_size * context_size]
        adj = bake_mask(c_adj=c_adj)
        # w_j: [..., multi_context_size]
        w_j = batch_expand(w_j, expand_shape)
        return x, adj, w_j

    def scatter_gather(self,
                       x: th.Tensor,
                       adj: th.BoolTensor,
                       fn: Callable,
                       ):
        return th.stack([
            fn(
                x=x[..., i, :],
                adjacency=adj[..., i, :, :],
            ) for i in range(x.size(-2))
        ], dim=-2)

    def forward(self,
                x: th.Tensor,
                adj: th.BoolTensor,
                w_j: th.Tensor = None,
                ) -> th.Tensor:
        # Phi: [..., multi_context_size, out_features]
        Phi = self.scatter_gather(x, adj, self._hyper)
        # phi: [..., out_features]
        phi = self._reduce_fn(x=Phi, mask=w_j)
        return phi


def safe_softmax(weight: th.Tensor, w_j: BoolTensor, dim: int = -1):
    weight = weight - weight.amax(dim=dim, keepdim=True)  # Avoid big number
    weight[~w_j] = weight[~w_j] - weight[~w_j].detach()  # Avoid noise
    return th.softmax(weight, dim=dim)


class WeightedReduceAggregator(ReduceAggregator):
    def __init__(self,
                 hyper: nn.Module,
                 out_features: int,
                 reduce: str = 'wsum',
                 hidden_features: Sequence[int] = tuple(),
                 **kwargs,
                 ) -> None:
        super().__init__(
            hyper=hyper,
            reduce={
                'wsum': 'sum',
                'wmean': 'mean',
                'wmin': 'min',
                'wmax': 'max',
            }[reduce],
        )

        self._weight_hyper = MLP(
            in_features=out_features,
            out_features=1,
            hidden_features=hidden_features,
        )  # Weight derive from phi: w(f(x)) * f(x)

    def forward(self,
                x: th.Tensor,
                adj: th.BoolTensor,
                w_j: th.Tensor = None,
                ) -> th.Tensor:
        Phi = self.scatter_gather(x, adj, self._hyper)
        weight = self._weight_hyper(Phi).squeeze(-1)
        # Softmax
        weight = safe_softmax(weight, w_j, dim=-1)
        weight = feature_expand(weight, th.Size((Phi.size(-1), )))

        # Weighted sum
        Phi = weight * Phi
        phi = self._reduce_fn(x=Phi, mask=w_j)
        return phi


class AttentionReduceAggregator(ReduceAggregator):
    def __init__(self,
                 hyper: nn.Module,
                 context: int,
                 out_features: int,
                 hidden_features: Sequence[int] = tuple(),
                 **kwargs,
                 ) -> None:
        super().__init__(
            hyper=hyper,
            reduce='sum',
        )

        self.context = context
        self._weight_hyper = MLP(
            in_features=self.context,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        def _emb(x, fn):
            return th.stack([
                fn(x[..., i, -self.context:]) for i in range(x.size(-2))
            ], dim=-2)
        self._emb = _emb

    def forward(self,
                x: th.Tensor,
                adj: th.BoolTensor,
                w_j: th.Tensor = None,
                ) -> th.Tensor:
        Phi = self.scatter_gather(x, adj, self._hyper)
        weight = self._emb(x, self._weight_hyper)
        # Softmax
        weight = safe_softmax(weight, w_j, dim=-2)

        # Weighted sum
        Phi = weight * Phi
        phi = self._reduce_fn(x=Phi, mask=w_j)
        return phi
