import torch as th
from torch.distributions import Transform
from torch.distributions import MultivariateNormal
from typing import *

from zuko.flows.core import *
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import unpack

from model.zuko import BGMM, SafeMixture
from model.counterfactual.exo_match.multi_context.reduce import *
from dataset.utils import *


class MultiContextGMM(BGMM):
    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        max_context_num: int = 1,
        reduce: str = 'concat',
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
            components=components,
            context=context_for_hyper,
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
                    c: th.Tensor,
                    adj: th.BoolTensor,
                    w_j: th.BoolTensor,) -> Transform:
        x = th.empty((*c.shape[:-2], 0), device=c.device)  # dummy x
        # Copied from model.counterfactual.exo_match.multi_context.autoregressive
        x, adj, w_j = self.aggregator.preprocess(
            x, c, adj, w_j, self.bake_mask
        )
        return self.aggregator(x, adj, w_j)

    def forward(self,
                c: th.Tensor = None,
                adj: th.BoolTensor = None,
                w_j: th.BoolTensor = None,
                ) -> NormalizingFlow:
        if c is None:
            phi = self.phi
        else:
            phi = self.reduced_phi(c, adj, w_j)
            phi = unpack(phi, self.shapes)

        logits, loc, diag, tril = phi

        scale = torch.diag_embed(diag.exp() + 1e-5)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)

        return SafeMixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)
