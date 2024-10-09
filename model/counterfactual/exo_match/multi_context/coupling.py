import torch as th
import torch.nn as nn
from torch.distributions import Transform
from typing import *

from zuko.flows.core import *
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import unpack

from model.zuko import BatchedGeneralCouplingTransform, BNICE
from model.counterfactual.exo_match.multi_context.reduce import *
from dataset.utils import *


class MultiContextGeneralCouplingTransform(BatchedGeneralCouplingTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        max_context_num: int = 1,
        reduce: str = 'concat',
        mask: BoolTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[th.Size] = ((), ()),
        **kwargs,
    ) -> None:
        # Copied from model.counterfactual.exo_match.multi_context.autoregressive
        self.reduce = reduce
        if reduce == 'concat':
            context_for_hyper = max_context_num * context
        else:
            context_for_hyper = context
        super().__init__(
            features=features,
            context=context_for_hyper,
            mask=mask,
            univariate=univariate,
            shapes=shapes,
            **kwargs,
        )
        self.context = context
        if reduce in ['concat']:
            self.aggregator = ConcatAggregator(
                hyper=self.hyper,
            )
        if reduce in ['sum', 'mean', 'min', 'max']:
            self.aggregator = ReduceAggregator(
                hyper=self.hyper,
                reduce=reduce,
            )
        if reduce in ['wsum', 'wmean', 'wmin', 'wmax']:
            self.aggregator = WeightedReduceAggregator(
                hyper=self.hyper,
                reduce=reduce,
                out_features=self.out_features,
                **kwargs,
            )
        if reduce in ['attn']:
            self.aggregator = AttentionReduceAggregator(
                hyper=self.hyper,
                reduce=reduce,
                context=self.context,
                out_features=self.out_features,
                **kwargs,
            )

    def reduced_phi(self,
                    x: th.Tensor,
                    c: th.Tensor,
                    adj: th.BoolTensor,
                    w_j: th.BoolTensor
                    ) -> th.Tensor:
        x, adj, w_j = self.aggregator.preprocess(
            x, c, adj, w_j, self.bake_mask
        )
        return self.aggregator(x, adj, w_j)

    def meta(self,
             x: th.Tensor,
             c: th.Tensor,
             adj: th.BoolTensor,
             w_j: th.BoolTensor,
             ) -> Transform:
        phi = self.reduced_phi(x, c, adj, w_j)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)
        return DependentTransform(self.univariate(*phi), 1)

    def forward(self,
                c: th.Tensor,
                adj: th.BoolTensor,
                w_j: th.BoolTensor,
                ) -> Transform:
        meta = partial(self.meta, c=c, adj=adj, w_j=w_j)
        return CouplingTransform(meta, self.mask)


class MultiContextNICE(BNICE):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randmask: bool = False,
        base: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # Copied from model.zuko.coupling.BNICE
        self.n_transforms = transforms
        transforms = []
        for i in range(self.n_transforms):
            if randmask:
                mask = torch.randperm(features) % 2 == i % 2
            else:
                mask = torch.arange(features) % 2 == i % 2
            transforms.append(
                MultiContextGeneralCouplingTransform(
                    features=features,
                    context=context,
                    mask=mask,
                    **kwargs,
                )
            )
        base = base or Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )
        LazyTransform.__init__(self)
        self.transforms = nn.ModuleList(transforms)
        self.base = base

    def forward(self,
                c: th.Tensor = None,
                adj: th.BoolTensor = None,
                w_j: th.BoolTensor = None,
                ) -> NormalizingFlow:
        # Copied from model.zuko.autoregressive.MAF
        transforms, i = [], 0
        for t in self.transforms:
            if isinstance(t, MultiContextGeneralCouplingTransform):  # MCGCT
                transforms.append(t(c, adj, w_j))
                i += 1
            else:  # For other transforms
                transforms.append(t(c))
        transform = ComposedTransform(*transforms)
        if c is None:
            base = self.base(c)
        else:
            # However, context size is not expanded
            base = self.base(c).expand(c.shape[:-2])
        return NormalizingFlow(transform, base)
