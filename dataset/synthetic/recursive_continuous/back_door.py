import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='z')
def f1(uz):
    return uz


@Equation(v='x', pV='z')
def f2(z, ux):
    return th.exp(z / 2) + ux / 4


@Equation(v='y', pV=['x', 'z'])
def f3(x, z, uy):
    return th.tanh(2 * z) + 1.5 * x - 1 + th.tanh(uy)


def back_door_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='back_door',
        equations=[
            f1,
            f2,
            f3,
        ],
        exogenous_distrs={
            'uz': Normal(loc=0, scale=1),
            'ux': Normal(loc=0, scale=1),
            'uy': Normal(loc=0, scale=1),
        },
    )
