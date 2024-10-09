import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='x')
def f1(ux, uxy):
    return ux + uxy


@Equation(v='z', pV='x')
def f2(x, uz):
    return th.exp(x / 2) + uz / 4


@Equation(v='y', pV='z')
def f3(z, uy, uxy):
    return (z - 5) / 15 + 2 * uy + 0.5 * uxy


def front_door_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='front_door',
        equations=[
            f1,
            f2,
            f3,
        ],
        exogenous_distrs={
            'ux': Normal(loc=0, scale=1),
            'uz': Normal(loc=0, scale=1),
            'uy': Normal(loc=0, scale=1),
            'uxy': Normal(loc=0, scale=1),
        },
    )
