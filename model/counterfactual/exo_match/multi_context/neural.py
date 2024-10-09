import torch as th
import torch.nn as nn
from typing import *

from zuko.flows.core import *
from zuko.distributions import *
from zuko.transforms import *
from zuko.flows.neural import MNN, UMNN

from model.zuko import BNAF, BUNAF
from model.counterfactual.exo_match.multi_context.autoregressive import *


class MultiContextNAF(BNAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        signal: int = 16,
        network: Dict[str, Any] = {},
        base: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # Copied from model.zuko.neural.BNAF
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]
        self.n_transforms = transforms
        transforms = [
            MultiContextMaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=MNN(signal=signal, stack=features, **network),
                shapes=[(signal,)],
                **kwargs,
            )
            for i in range(self.n_transforms)
        ]

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=11.0))
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
            if isinstance(t, MultiContextMaskedAutoregressiveTransform):  # MCMAT
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


class MultiContextUNAF(BUNAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        signal: int = 16,
        network: Dict[str, Any] = {},
        base: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # Copied from model.zuko.neural.BUNAF
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]
        self.n_transforms = transforms
        transforms = [
            MultiContextMaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=UMNN(signal=signal, stack=features, **network),
                shapes=[(signal,), ()],
                **kwargs,
            )
            for i in range(self.n_transforms)
        ]
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
            if isinstance(t, MultiContextMaskedAutoregressiveTransform):  # MCMAT
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
