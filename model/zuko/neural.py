import torch
import torch.nn as nn
from torch import BoolTensor, Tensor
from typing import *

from zuko.flows.core import *
from zuko.flows.neural import MNN, UMNN
from zuko.distributions import *
from zuko.transforms import *

from model.zuko.autoregressive import BatchedMaskedAutoregressiveTransform


class BNAF(LazyTransform):
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
                univariate=MNN(signal=signal, stack=features, **network),
                shapes=[(signal,)],
                **kwargs,
            )
            for i in range(self.n_transforms)
        ]

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=11.0))

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


class BUNAF(LazyTransform):
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
                univariate=UMNN(signal=signal, stack=features, **network),
                shapes=[(signal,), ()],
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
