import math
import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='x1')
def f1(u1):
    return u1


@Equation(v='x2', pV='x1')
def f2(x1, u2):
    return 2 * th.tanh(2 * x1) + 1 / math.sqrt(10) * u2


@Equation(v='x3', pV=['x1', 'x2'])
def f3(x1, x2, u3):
    return 0.5 * x1 * x2 + 1 / math.sqrt(2) * u3


@Equation(v='x4', pV='x1')
def f4(x1, u4):
    return th.tanh(1.5 * x1) + math.sqrt(3 / 10) * u4


def simpson_symprod_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='simpson_symprod',
        equations=[
            f1,
            f2,
            f3,
            f4,
        ],
        exogenous_distrs={
            'u1': Normal(loc=0, scale=1),
            'u2': Normal(loc=0, scale=1),
            'u3': Normal(loc=0, scale=1),
            'u4': Normal(loc=0, scale=1),
        },
    )
