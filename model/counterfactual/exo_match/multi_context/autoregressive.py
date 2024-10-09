import torch as th
import torch.nn as nn
from torch.distributions import Transform
from typing import *

from zuko.flows.core import *
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import unpack

from model.zuko import BatchedMaskedAutoregressiveTransform, BMAF
from model.counterfactual.exo_match.multi_context.reduce import *
from dataset.utils import *


class MultiContextMaskedAutoregressiveTransform(BatchedMaskedAutoregressiveTransform):
    def __init__(
        self,
        features: int,
        context: int = 0,
        max_context_num: int = 1,
        reduce: str = 'concat',
        passes: int = None,
        order: th.LongTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[th.Size] = ((), ()),
        **kwargs,
    ) -> None:
        self.reduce = reduce

        # Concatenate
        if reduce == 'concat':
            context_for_hyper = max_context_num * context
        else:
            context_for_hyper = context

        super().__init__(
            features=features,
            context=context_for_hyper,
            passes=passes,
            order=order,
            univariate=univariate,
            shapes=shapes,
            **kwargs,
        )

        # Reset context size if concatenate
        self.context = context

        # Concatenate
        if reduce in ['concat']:
            self.aggregator = ConcatAggregator(
                hyper=self.hyper,
            )

        # Reduce
        if reduce in ['sum', 'mean', 'min', 'max']:
            self.aggregator = ReduceAggregator(
                hyper=self.hyper,
                reduce=reduce,
            )

        # Weighted Reduce
        if reduce in ['wsum', 'wmean', 'wmin', 'wmax']:
            self.aggregator = WeightedReduceAggregator(
                hyper=self.hyper,
                reduce=reduce,
                out_features=self.out_features,
                **kwargs,
            )

        # Attention Reduce
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
        # x: [..., feature_size]
        # c: [..., multi_context_size, context_size]
        # adj: [..., multi_context_size, feature_size, feature_size + context_size]

        # Get preprocessed tensors
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
        # Phi from x and c
        phi = self.reduced_phi(x, c, adj, w_j)

        # Unpack fro each parameter
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)

    def forward(self,
                c: th.Tensor,
                adj: th.BoolTensor,
                w_j: th.BoolTensor,
                ) -> Transform:
        meta = partial(self.meta, c=c, adj=adj, w_j=w_j)
        return AutoregressiveTransform(meta, self.passes)


class MultiContextMAF(BMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        base: Optional[th.nn.Module] = None,
        aggregator_type: Type[nn.Module] = ReduceAggregator,
        aggregator_kwargs: Dict[Any, str] = {},
        **kwargs,
    ):
        # Copied from model.zuko.autoregressive.MAF
        orders = [
            th.arange(features),
            th.flipud(th.arange(features)),
        ]
        self.n_transforms = transforms
        transforms = [
            MultiContextMaskedAutoregressiveTransform(  # Use MultiContextMAT instead
                features=features,
                context=context,
                order=th.randperm(features) if randperm else orders[i % 2],
                aggregator_type=aggregator_type,
                aggregator_kwargs=aggregator_kwargs,
                **kwargs,
            )
            for i in range(self.n_transforms)
        ]
        base = base or Unconditional(
            DiagNormal,
            th.zeros(features),
            th.ones(features),
            buffer=True,
        )

        # Copied from LazyComposedTransform
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
            if isinstance(t, BatchedMaskedAutoregressiveTransform):  # MCMAT
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
