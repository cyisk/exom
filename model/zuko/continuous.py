from functools import partial
from torch import Tensor, BoolTensor
from torch.distributions import Transform
from typing import *

from zuko.flows.core import *
from zuko.flows.continuous import FFJTransform
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import broadcast

from model.zuko.nn import *
from dataset.utils import batch_expand


class BatchedFFJTransform(FFJTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        freqs: int = 3,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
        **kwargs,
    ):
        super().__init__(features, context, freqs, atol, rtol, exact, **kwargs)

        # Hyper
        self.context = context
        self.features = features
        self.freqs = freqs
        self.in_features = features + context + 2 * freqs
        self.out_features = features
        self.ode = BatchedMaskedMLP(
            features + context + 2 * freqs, features, **kwargs
        )

        # Prepare unique and inverse for hyper
        self.register_buffer(
            "unique",
            torch.arange(features),
        )
        self.register_buffer(
            "inverse",
            torch.arange(features),
        )

        # Default feature adjacency
        self.register_buffer(
            "feature_adjacency",
            torch.ones(self.features, self.features).bool(),
        )
        # Default full context adjacency
        self.register_buffer(
            "full_context_adj",
            torch.ones(self.features, self.context).bool(),
        )
        # Default frequency adjacency
        self.register_buffer(
            "frequency_adjacency",
            torch.ones(self.features, 2 * self.freqs).bool(),
        )

    def f(self,
          t: Tensor,
          x: Tensor,
          c: Tensor,
          f_adj: Tensor,
          c_adj: Tensor,
          ) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        if c is None:
            x = torch.cat(broadcast(t, x, ignore=1), dim=-1)
        else:
            x = torch.cat(broadcast(t, x, c, ignore=1), dim=-1)

        # Adjacency of this transform
        adj = self.bake_mask(f_adj, c_adj)
        return self.ode(x, adj)

    def bake_mask(self,
                  f_adj: BoolTensor = None,
                  c_adj: BoolTensor = None,
                  ) -> BoolTensor:
        # Either is None, use default
        if f_adj is None:
            f_adj = self.feature_adjacency
        if c_adj is None:
            c_adj = self.full_context_adj
        fr_adj = self.frequency_adjacency

        # Repeat for batch size
        if f_adj.dim() > c_adj.dim():
            dim_diff = f_adj.dim() - c_adj.dim()
            c_adj = batch_expand(c_adj, f_adj.shape[:dim_diff])
        if c_adj.dim() > f_adj.dim():
            dim_diff = c_adj.dim() - f_adj.dim()
            f_adj = batch_expand(f_adj, c_adj.shape[:dim_diff])
            fr_adj = batch_expand(fr_adj, c_adj.shape[:dim_diff])

        # Concat adjacencies: [..., feature_size, feature_size + context_size]
        adjacency = torch.cat((
            f_adj, c_adj, fr_adj
        ), dim=-1).repeat_interleave(
            self.total, dim=f_adj.dim() - 2
        )

        return adjacency

    def forward(self,
                c: Tensor = None,
                f_adj: Tensor = None,
                c_adj: Tensor = None,
                ) -> Transform:
        return FreeFormJacobianTransform(
            f=partial(self.f, c=c, f_adj=f_adj, c_adj=c_adj),
            t0=self.times[0],
            t1=self.times[1],
            phi=self.parameters() if c is None else (c, *self.parameters()),
            atol=self.atol,
            rtol=self.rtol,
            exact=self.exact,
        )


class BCNF(LazyTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        base: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # Transforms
        self.n_transforms = 1
        transforms = [
            FFJTransform(
                features=features,
                context=context,
                **kwargs,
            )
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
            if isinstance(t, BatchedFFJTransform):  # FFJT
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
