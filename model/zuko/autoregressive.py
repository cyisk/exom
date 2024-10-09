import torch

from functools import partial
from math import ceil, prod
from torch import BoolTensor, LongTensor, Size
from torch.distributions import Transform
from typing import *

from zuko.flows.core import *
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import broadcast, unpack

from model.zuko.nn import *
from dataset.utils import batch_expand


class BatchedMaskedAutoregressiveTransform(MaskedAutoregressiveTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super(LazyTransform, self).__init__()

        # Copied from zuko.autoregressive.MaskedAutoregressiveTransform
        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)
        self.register_buffer('order', None)
        if passes is None:
            passes = features
        if order is None:
            order = torch.arange(features)
        else:
            order = torch.as_tensor(order)
        self.passes = min(max(passes, 1), features)
        self.order = torch.div(order, ceil(
            features / self.passes), rounding_mode='floor'
        )

        # Other parameters
        self.context = context

        # BatchedMaskedMLP
        self.in_features = len(self.order) + self.context
        self.out_features = len(self.order) * self.total
        self.hyper = BatchedMaskedMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            **kwargs,
        )

        # Prepare unique and inverse for hyper
        self.register_buffer(
            "unique",
            torch.arange(len(self.order)) * self.total,
        )
        self.register_buffer(
            "inverse",
            torch.arange(len(self.order)).repeat_interleave(self.total, dim=0),
        )

        # Default autoregressive feature adjacency
        self.register_buffer(
            "autoregressive_adjacency",
            self.order[:, None] > self.order,
        )
        # Default full context adjacency
        self.register_buffer(
            "full_context_adj",
            torch.ones(len(self.order), self.context).bool(),
        )

    def meta(self,
             x: Tensor,
             c: Tensor,
             f_adj: Tensor,
             c_adj: Tensor,
             ) -> Transform:
        if c is not None:
            x = torch.cat(broadcast(x, c, ignore=1), dim=-1)

        # Adjacency of this transform
        adj = self.bake_mask(f_adj, c_adj)

        # Inference
        phi = self.hyper(x, adj)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)

    def forward(self,
                c: Tensor = None,
                f_adj: Tensor = None,
                c_adj: Tensor = None,
                ) -> Transform:
        meta = partial(self.meta, c=c, f_adj=f_adj, c_adj=c_adj)
        return AutoregressiveTransform(meta, self.passes)

    def bake_mask(self,
                  f_adj: BoolTensor = None,
                  c_adj: BoolTensor = None,
                  ) -> BoolTensor:
        # Either is None, use default
        if f_adj is None:
            f_adj = self.autoregressive_adjacency
        if c_adj is None:
            c_adj = self.full_context_adj

        # Repeat for batch size
        if f_adj.dim() > c_adj.dim():
            dim_diff = f_adj.dim() - c_adj.dim()
            c_adj = batch_expand(c_adj, f_adj.shape[:dim_diff])
        if c_adj.dim() > f_adj.dim():
            dim_diff = c_adj.dim() - f_adj.dim()
            f_adj = batch_expand(f_adj, c_adj.shape[:dim_diff])

        # Concat adjacencies: [..., feature_size, feature_size + context_size]
        adjacency = torch.cat((
            f_adj, c_adj
        ), dim=-1).repeat_interleave(
            self.total, dim=f_adj.dim() - 2
        )

        return adjacency


class BMAF(LazyTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        base: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # Orders
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        # Transforms
        self.n_transforms = transforms
        transforms = [
            BatchedMaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(self.n_transforms)
        ]

        # Base distribution
        base = base or Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        # Copied from LazyComposedTransform
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.base = base

    def forward(self,
                c: Tensor = None,
                f_adjs: List[BoolTensor] | BoolTensor = None,
                c_adjs: List[BoolTensor] | BoolTensor = None,
                ) -> NormalizingFlow:
        f_adjs, c_adjs = self.bake_masks(f_adjs, c_adjs)

        transforms, i = [], 0
        for t in self.transforms:
            if isinstance(t, BatchedMaskedAutoregressiveTransform):  # MAT
                transforms.append(t(c, f_adjs[i], c_adjs[i]))
                i += 1
            else:  # For other transforms
                transforms.append(t(c))

        transform = ComposedTransform(*transforms)

        if c is None:
            base = self.base(c)
        else:
            base = self.base(c).expand(c.shape[:-1])

        return NormalizingFlow(transform, base)

    def bake_masks(self,
                   f_adjs: List[BoolTensor] | BoolTensor = None,
                   c_adjs: List[BoolTensor] | BoolTensor = None,
                   ) -> List[BoolTensor]:
        masks1, masks2 = [], []
        for i in range(self.n_transforms):
            f_adj = f_adjs[i] if isinstance(f_adjs, list) else f_adjs
            masks1.append(f_adj)
            c_adj = c_adjs[i] if isinstance(c_adjs, list) else c_adjs
            masks2.append(c_adj)
        return masks1, masks2
