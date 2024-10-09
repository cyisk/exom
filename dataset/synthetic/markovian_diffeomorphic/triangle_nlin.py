import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='x1')
def f1(u1):
    return u1 + 1


@Equation(v='x2', pV='x1')
def f2(x1, u2):
    return 2 * x1 ** 3 + u2


@Equation(v='x3', pV=['x1', 'x2'])
def f3(x1, x2, u3):
    return 20 / (1 + th.exp(-x2 ** 2 + x1)) + u3


def triangle_nlin_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='triangle_nlin',
        equations=[
            f1,
            f2,
            f3,
        ],
        exogenous_distrs={
            'u1': Normal(loc=0, scale=1),
            'u2': Normal(loc=0, scale=1),
            'u3': Normal(loc=0, scale=1),
        },
    )
