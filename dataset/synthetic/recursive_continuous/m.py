import math
import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='x')
def f1(ux, uxz):
    return ux + uxz


@Equation(v='z')
def f2(uz, uxz, uyz):
    return th.exp((uxz - uyz) / 4) + math.sqrt(10 / 3) * uz


@Equation(v='y', pV='x')
def f3(x, uy, uyz):
    return th.tanh(2 * x) + 1.5 * uy - 1 + th.tanh(uyz)


def m_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='m',
        equations=[
            f1,
            f2,
            f3,
        ],
        exogenous_distrs={
            'ux': Normal(loc=0, scale=1),
            'uy': Normal(loc=0, scale=1),
            'uz': Normal(loc=0, scale=1),
            'uxz': Normal(loc=0, scale=1),
            'uyz': Normal(loc=0, scale=1),
        },
    )
