import math
import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


def s(x):
    return th.log(1 + th.exp(x))


@Equation(v='x1')
def f1(u1):
    return u1


@Equation(v='x2', pV='x1')
def f2(x1, u2):
    return s(1 - x1) + math.sqrt(3 / 20) * u2


@Equation(v='x3', pV=['x1', 'x2'])
def f3(x1, x2, u3):
    return th.tanh(2 * x2) + 1.5 * x1 - 1 + th.tanh(u3)


@Equation(v='x4', pV='x3')
def f4(x3, u4):
    return (x3 - 4) / 5 + 3 + 1 / math.sqrt(10) * u4


def simpson_nlin_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='simpson_nlin',
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
