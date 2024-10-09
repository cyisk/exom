import torch
import torch.nn as nn

from math import prod
from torch import Tensor, BoolTensor
from torch.distributions import Distribution, MultivariateNormal
from typing import *

from zuko.flows.core import *
from zuko.flows.mixture import Mixture
from zuko.distributions import *
from zuko.transforms import *
from zuko.utils import unpack

from dataset.utils import *
from model.zuko.nn import BatchedMaskedMLP


class SafeMixture(Mixture):
    def log_prob(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=-(len(self.event_shape) + 1))

        logits = self.logits - self.logits.amax(dim=-1, keepdim=True)
        log_w = torch.log_softmax(logits, dim=-1)  # Safe softmax
        log_p = self.base.log_prob(x)

        mixed_log_p = torch.logsumexp(log_w + log_p, dim=-1)
        mask = log_p.isinf().any(dim=-1)  # -inf cannot be backpropagated
        return mixed_log_p.masked_fill(mask, -th.inf)  # Block -inf logprob


class BGMM(LazyDistribution):
    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Copied from zuko.flows.mixture.GMM
        shapes = [
            (components,),  # probabilities
            (components, features),  # mean
            (components, features),  # diagonal
            (components, features * (features - 1) // 2),  # off diagonal
        ]
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)
        self.components = components
        self.features = features
        self.context = context
        self.in_features = context
        self.out_features = self.total

        if context > 0:
            self.hyper = BatchedMaskedMLP(
                self.in_features, self.out_features, **kwargs
            )
        else:
            self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)

        # Prepare unique and inverse for hyper
        self.register_buffer("adjacency_unique", None)
        self.adjacency_unique = torch.tensor([i for i in range(self.total)])
        self.register_buffer("adjacency_inverse", None)
        self.adjacency_inverse = torch.tensor([i for i in range(self.total)])
        assert self.adjacency_inverse.size(0) == self.total

    def bake_mask(self,
                  c_adj: BoolTensor = None,
                  ) -> BoolTensor:
        if c_adj is None:
            # adjacency: [..., total_size, context_size]
            adjacency = torch.ones(
                self.total, self.context
            ).bool()
        else:
            # c_adj: [..., feature_size, context_size]
            batch_dim = c_adj.dim() - 2
            batch_ones = [-1] * batch_dim

            # p_adjacency: [..., components, context_size]
            p_adjacency = torch.ones_like(c_adj[..., 0, :])
            p_adjacency = p_adjacency[..., None, :].expand(
                *batch_ones, self.components, -1
            ).bool()

            # m_adjacency: [..., components, feature_size, context_size]
            m_adjacency = c_adj[..., None, :, :].expand(
                *batch_ones, self.components, -1, -1
            )

            # d_adjacency: [..., components, feature_size, context_size]
            d_adjacency = torch.ones_like(m_adjacency).bool()

            # o_adjacency: [..., components, feature_size * (feature_size - 1) // 2, context_size]
            o_adjacency = torch.ones_like(m_adjacency[..., 0, :])
            o_adjacency = o_adjacency[..., None, :].expand(
                *batch_ones, -1, self.features * (self.features - 1) // 2, -1
            ).bool()

            # adjacency = [..., total_size, context_size]
            adjacency = torch.cat([
                p_adjacency,
                m_adjacency.flatten(start_dim=-3, end_dim=-2),
                d_adjacency.flatten(start_dim=-3, end_dim=-2),
                o_adjacency.flatten(start_dim=-3, end_dim=-2),
            ], dim=-2)

            assert adjacency.size(-2) == self.total

        return adjacency

    def forward(self,
                c: Tensor = None,
                c_adj: Tensor = None,
                ) -> Distribution:
        if c is None:
            phi = self.phi
        else:
            mask = self.bake_mask(c_adj=c_adj)
            phi = self.hyper(c, mask)
            phi = unpack(phi, self.shapes)

        logits, loc, diag, tril = phi

        scale = torch.diag_embed(diag.exp() + 1e-5)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)

        return SafeMixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)
