from functools import partial
from typing import *

from zuko.flows.core import *
from zuko.transforms import *

from model.zuko.autoregressive import BMAF


class BSOSPF(BMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        degree: int = 4,
        polynomials: int = 3,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=SOSPolynomialTransform,
            shapes=[(polynomials, degree + 1), ()],
            **kwargs,
        )

        transforms = self.transforms

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=11.0))


class BBPF(BMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        degree: int = 16,
        linear: bool = False,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=partial(BernsteinTransform, linear=linear),
            shapes=[(degree + 1,)],
            **kwargs,
        )

        transforms = self.transforms

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=11.0))
