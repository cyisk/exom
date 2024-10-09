from functools import partial
from math import prod
from torch import Tensor, BoolTensor, Size
from torch.distributions import Transform
from typing import *

from zuko.flows.core import *
from zuko.flows.coupling import GeneralCouplingTransform
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import broadcast, unpack

from model.zuko.nn import *
from dataset.utils import batch_expand


class BatchedGeneralCouplingTransform(GeneralCouplingTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        mask: BoolTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super(LazyTransform, self).__init__()

        # Univariate transformation
        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        # Mask
        self.register_buffer('mask', None)
        if mask is None:
            self.mask = torch.arange(features) % 2 == 1
        else:
            self.mask = mask

        # Hyper
        self.context = context
        self.features_a = self.mask.sum().item()
        self.features_b = features - self.features_a
        self.in_features = self.features_a + context
        self.out_features = self.features_b * self.total
        self.hyper = BatchedMaskedMLP(self.features_a + context,
                                      self.features_b * self.total, **kwargs)

        # Prepare unique and inverse for hyper
        self.register_buffer(
            "unique",
            torch.arange(self.features_b) * self.total,
        )
        self.register_buffer(
            "inverse",
            torch.arange(self.features_b).repeat_interleave(self.total, dim=0),
        )

        # Default feature adjacency
        self.register_buffer(
            "feature_adjacency",
            torch.ones(self.features_b, self.features_a).bool(),
        )
        # Default full context adjacency
        self.register_buffer(
            "full_context_adj",
            torch.ones(self.features_b, self.context).bool(),
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
        return CouplingTransform(meta, self.mask)

    def bake_mask(self,
                  f_adj: BoolTensor = None,
                  c_adj: BoolTensor = None,
                  ) -> BoolTensor:
        # Either is None, use default
        if f_adj is None:
            f_adj = self.feature_adjacency
        if c_adj is None:
            c_adj = self.full_context_adj
        c_adj = c_adj[..., ~self.mask, :]

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


class BNICE(LazyTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randmask: bool = False,
        base: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # Transforms
        self.n_transforms = transforms
        transforms = []
        for i in range(self.n_transforms):
            if randmask:
                mask = torch.randperm(features) % 2 == i % 2
            else:
                mask = torch.arange(features) % 2 == i % 2
            transforms.append(
                BatchedGeneralCouplingTransform(
                    features=features,
                    context=context,
                    mask=mask,
                    **kwargs,
                )
            )

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
            if isinstance(t, BatchedGeneralCouplingTransform):  # GCT
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
