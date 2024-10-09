import torch

from math import pi
from typing import *

from zuko.flows.core import *
from zuko.flows.spline import CircularRQSTransform
from zuko.distributions import BoxUniform
from zuko.transforms import *

from model.counterfactual.exo_match.multi_context.autoregressive import *


class MultiContextNSF(MultiContextMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )


class MultiContextNCSF(MultiContextMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=CircularRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = Unconditional(
            BoxUniform,
            torch.full((features,), -pi - 1e-5),
            torch.full((features,), pi + 1e-5),
            buffer=True,
        )