import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='x1')
def f1(u1):
    return u1


@Equation(v='x2')
def f2(u2):
    return u2


@Equation(v='x3', pV=['x1', 'x2'])
def f3(x1, x2, u3):
    return 4 / (1 + th.exp(-x1 - x2)) - x2 ** 2 + 0.5 * u3


@Equation(v='x4', pV=['x3'])
def f4(x3, u4):
    return 20 / (1 + th.exp(0.5 * x3 ** 2 - x3)) + u4


def fork_nlin_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='fork_nlin',
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
